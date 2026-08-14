"""
Extract (decode) the ROS 2 bag data recorded by the robot into readable /
numpy-friendly files.

The bags are sqlite3 "rosbag2" storage containing *serialized* CDR messages.
This script reads them with the pure-Python `rosbags` library (no ROS 2 shell
environment required), deserializes every message and writes, for each bag:

    <out-dir>/<bag_name>/
        summary.json          # counts, duration and per-field statistics
        csv/<topic>.csv       # human-readable flat tables per topic (default output)
                              # (data.npz is legacy; use --format npz/both only if needed)

Why a custom typestore?
------------------------
These bags were recorded when `interface_protocol` used an *older* message
definition than the one currently compiled in the workspace.  In particular:

    MotorDebug  (recorded era):  float64[] tau_cmd, mos_temperature, motor_temperature
    MotorDebug  (current):       7 fields  -> deserializing old data with the
                                current type fails (CDR layout mismatch)

    MotionState (recorded era):  string current_motion_task
    MotionState (current):      + string[] available_transition_motions

Therefore this script registers the *recording-era* definitions (they match the
bytes actually stored in the bags).  `rclpy.serialization` cannot be used for
those two topics because the compiled typesupport only knows the current layout.

Usage
-----
Run with the interpreter that has `rosbags` installed (the workspace venv):

    cd /home/xiaoran/workspaces/engineai_ros2_workspace
    ./.venv/bin/python src/identification/identification/extract_bag_data.py

Options
-------
    bags ...            bag dirs / a directory containing bag dirs / a .db3 file
                        (default: the package's `bags/` directory)
    --out-dir DIR       output root (default: src/identification/bag_data)
    --topics a,b,c      only extract these topics
    --format FMT        npz | csv | both  (default: csv)
    --max-messages N    stop after N messages per topic (quick checks)
    --summary-only      print the summary only, write no data files
    --msg-dir DIR       use .msg files from DIR instead of the embedded
                        recording-era definitions (e.g. for newly recorded bags)

Dependencies: `rosbags` (pip install rosbags) and `numpy`.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Recording-era interface_protocol message definitions
# ---------------------------------------------------------------------------
# MotorDebug / MotionState below are the OLD (recorded) layouts.  All other
# topics are unchanged since recording, so their definitions equal the current
# ones in src/interface_protocol/msg/.
MSG_DEFS_RECORDED: dict[str, str] = {
    "interface_protocol/msg/JointState": (
        "std_msgs/Header header\n"
        "float64[] position\n"
        "float64[] velocity\n"
        "float64[] torque\n"
    ),
    "interface_protocol/msg/JointCommand": (
        "std_msgs/Header header\n"
        "float64[] position\n"
        "float64[] velocity\n"
        "float64[] feed_forward_torque\n"
        "float64[] torque\n"
        "float64[] stiffness\n"
        "float64[] damping\n"
        "uint8 parallel_parser_type\n"
    ),
    "interface_protocol/msg/ImuInfo": (
        "std_msgs/Header header\n"
        "geometry_msgs/Quaternion quaternion\n"
        "geometry_msgs/Vector3 rpy\n"
        "geometry_msgs/Vector3 linear_acceleration\n"
        "geometry_msgs/Vector3 angular_velocity\n"
    ),
    "interface_protocol/msg/GamepadKeys": (
        "std_msgs/Header header\n"
        "bool hardware_connected\n"
        "int32[12] digital_states\n"
        "float64[6] analog_states\n"
    ),
    "interface_protocol/msg/Heartbeat": (
        "std_msgs/Header header\n"
        "string node_name\n"
        "int64 startup_timestamp\n"
        "string node_status\n"
        "int64 error_code\n"
        "string error_message\n"
    ),
    # --- recorded-era (old) layouts, see module docstring ---
    "interface_protocol/msg/MotorDebug": (
        "float64[] tau_cmd\nfloat64[] mos_temperature\nfloat64[] motor_temperature\n"
    ),
    "interface_protocol/msg/MotionState": ("string current_motion_task\n"),
}


def _topic_key(topic: str) -> str:
    """Turn '/hardware/joint_state' into a filesystem/npz friendly key."""
    return topic.lstrip("/").replace("/", "_")


# ---------------------------------------------------------------------------
# Typestore construction
# ---------------------------------------------------------------------------
def build_typestore(msg_dir: str | Path | None = None):
    """Build a `rosbags` typestore able to deserialize the bag messages.

    If `msg_dir` is given, the message definitions are read from the `.msg`
    files there instead of the embedded recording-era definitions (useful for
    bags recorded with the current/new message layout).
    """
    from rosbags.typesys import Stores, get_types_from_msg, get_typestore

    store = get_typestore(Stores.ROS2_HUMBLE)
    defs = load_defs_from_dir(msg_dir) if msg_dir else dict(MSG_DEFS_RECORDED)
    for type_name, text in defs.items():
        store.register(get_types_from_msg(text, type_name))
    return store, defs


def load_defs_from_dir(msg_dir: str | Path) -> dict[str, str]:
    """Read `<package>/msg/*.msg` (or a bare msg dir) into type definitions."""
    msg_dir = Path(msg_dir)
    if msg_dir.name == "msg":
        pkg, search = msg_dir.parent.name, msg_dir
    else:
        pkg = msg_dir.name
        search = msg_dir / "msg" if (msg_dir / "msg").exists() else msg_dir
    defs: dict[str, str] = {}
    for f in sorted(search.glob("*.msg")):
        defs[f"{pkg}/msg/{f.stem}"] = f.read_text()
    return defs


# ---------------------------------------------------------------------------
# Generic CDR-message -> columns flattening
# ---------------------------------------------------------------------------
def _extract_header(msg) -> tuple[int, str]:
    """Pull the std_msgs/Header stamp (ns) and frame_id out of a message."""
    t_stamp_ns, frame_id = 0, ""
    header = getattr(msg, "header", None)
    if header is not None and hasattr(header, "__dict__"):
        frame_id = getattr(header, "frame_id", "") or ""
        stamp = getattr(header, "stamp", None)
        if stamp is not None and hasattr(stamp, "__dict__"):
            sec = int(getattr(stamp, "sec", 0) or 0)
            nsec = int(getattr(stamp, "nanosec", 0) or 0)
            t_stamp_ns = sec * 1_000_000_000 + nsec
    return t_stamp_ns, frame_id


def _flatten(obj, prefix: str, scalars: dict, arrays: dict, strings: dict):
    """Recursively flatten a deserialized message into column buckets.

    * numeric scalar   -> `scalars[prefix]`
    * numeric sequence -> `arrays[prefix]` (1-D numpy array)
    * nested message   -> recurse with `prefix.<field>`
    * string / string[]-> `strings[prefix]` (joined with '|' for arrays)
    """
    if isinstance(obj, (str, bytes)):
        strings[prefix] = (
            obj.decode(errors="replace") if isinstance(obj, bytes) else obj
        )
        return
    if isinstance(obj, (int, float, bool, np.integer, np.floating, np.bool_)):
        scalars[prefix] = float(obj)
        return
    if isinstance(obj, (list, tuple, np.ndarray)):
        if len(obj) == 0:
            arrays[prefix] = np.array([], dtype=float)
            return
        if all(
            isinstance(x, (int, float, bool, np.integer, np.floating, np.bool_))
            for x in obj
        ):
            arrays[prefix] = np.asarray(obj, dtype=float)
            return
        if all(isinstance(x, str) for x in obj):
            strings[prefix] = "|".join(obj)
            return
        strings[prefix] = repr(obj)
        return
    if hasattr(obj, "__dict__"):  # nested ROS message
        for name, val in vars(obj).items():
            if name.startswith("__"):
                continue
            _flatten(
                val, f"{prefix}.{name}" if prefix else name, scalars, arrays, strings
            )
        return
    strings[prefix] = str(obj)


class TopicAccumulator:
    """Accumulates every message of one topic into per-field column lists."""

    def __init__(self, topic: str, msgtype: str):
        self.topic = topic
        self.msgtype = msgtype
        self.t_recv_ns: list[int] = []
        self.t_stamp_ns: list[int] = []
        self.frame_id: list[str] = []
        self.has_header = False
        self.scalars: dict[str, list[float]] = {}
        self.arrays: dict[str, list[np.ndarray]] = {}
        self.strings: dict[str, list[str]] = {}
        self.n = 0

    def add(self, msg, t_recv_ns: int):
        t_stamp_ns, frame_id = _extract_header(msg)
        self.t_recv_ns.append(int(t_recv_ns))
        self.t_stamp_ns.append(t_stamp_ns)
        if self.n == 0:
            self.has_header = (
                frame_id != ""
                or t_stamp_ns != 0
                or getattr(msg, "header", None) is not None
            )
        if self.has_header:
            self.frame_id.append(frame_id)

        scalars, arrays, strings = {}, {}, {}
        for name, val in vars(msg).items():
            if name.startswith("__") or name == "header":
                continue
            _flatten(val, name, scalars, arrays, strings)

        if self.n == 0:  # establish schema from the first message
            self.scalars = {k: [] for k in scalars}
            self.arrays = {k: [] for k in arrays}
            self.strings = {k: [] for k in strings}
        for k, v in scalars.items():
            if k in self.scalars:
                self.scalars[k].append(v)
        for k, v in arrays.items():
            if k in self.arrays:
                self.arrays[k].append(np.asarray(v, dtype=float))
        for k, v in strings.items():
            if k in self.strings:
                self.strings[k].append(v)
        self.n += 1

    def to_arrays(self) -> dict[str, np.ndarray]:
        """Convert all columns to numpy arrays (padding ragged arrays with NaN)."""
        out: dict[str, np.ndarray] = {}
        t_recv = np.asarray(self.t_recv_ns, dtype=np.int64)
        out["t_recv_ns"] = t_recv
        out["t_stamp_ns"] = np.asarray(self.t_stamp_ns, dtype=np.int64)
        out["t_s"] = (t_recv - t_recv[0]) / 1e9 if len(t_recv) else np.zeros(0)
        if self.has_header:
            out["frame_id"] = np.asarray(self.frame_id, dtype="U")
        for k, v in self.scalars.items():
            out[k] = np.asarray(v, dtype=float)
        for k, v in self.arrays.items():
            out[k] = _stack_arrays(v)
        for k, v in self.strings.items():
            out[k] = np.asarray(v, dtype="U")
        return out


def _stack_arrays(rows: list[np.ndarray]) -> np.ndarray:
    """Stack 1-D rows into a 2-D array, padding ragged rows with NaN."""
    if not rows:
        return np.zeros((0, 0))
    width = max(len(r) for r in rows)
    out = np.full((len(rows), width), np.nan)
    for i, r in enumerate(rows):
        if len(r):
            out[i, : len(r)] = r
    return out


# ---------------------------------------------------------------------------
# Bag reading
# ---------------------------------------------------------------------------
def read_bag(
    bag_dir: str | Path, topics: set[str], store, max_messages: int | None = None
):
    """Read one bag and deserialize every requested topic into accumulators."""
    from rosbags.rosbag2 import Reader

    bag_dir = Path(bag_dir)
    reader = Reader(str(bag_dir))
    reader.open()
    try:
        conns = [c for c in reader.connections if c.topic in topics]
        accs = {c.topic: TopicAccumulator(c.topic, c.msgtype) for c in conns}
        for read_total, (connection, ts, raw) in enumerate(
            reader.messages(connections=conns)
        ):
            acc = accs[connection.topic]
            acc.add(store.deserialize_cdr(raw, connection.msgtype), ts)
            read_total += 1
            # `--max-messages` caps the total number of decoded messages
            # (a quick sanity check; leave None to decode everything).
            if max_messages and read_total >= max_messages:
                break
        meta = {
            "start_time_ns": reader.start_time,
            "end_time_ns": reader.end_time,
            "duration_ns": reader.duration,
        }
        return accs, meta
    finally:
        reader.close()


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def _stack_summary(accs, meta) -> dict:
    summary = {
        "bag": None,
        "start_time_ns": meta.get("start_time_ns"),
        "end_time_ns": meta.get("end_time_ns"),
        "duration_s": round(
            (meta.get("end_time_ns", 0) - meta.get("start_time_ns", 0)) / 1e9, 6
        )
        if meta.get("end_time_ns")
        else None,
        "topics": {},
    }
    for topic, acc in accs.items():
        arrays = acc.to_arrays()
        fields = {}
        for k, a in arrays.items():
            fields[k] = _field_stats(k, a)
        n = len(arrays["t_recv_ns"])
        rate = None
        if n > 1:
            dt = (arrays["t_recv_ns"][-1] - arrays["t_recv_ns"][0]) / 1e9
            rate = round((n - 1) / dt, 2) if dt > 0 else None
        summary["topics"][topic] = {
            "msgtype": acc.msgtype,
            "count": n,
            "rate_hz": rate,
            "fields": fields,
        }
    return summary


def _field_stats(name: str, a: np.ndarray) -> dict:
    info = {"shape": list(a.shape), "dtype": str(a.dtype), "kind": a.dtype.kind}
    if a.dtype.kind in "fciu":  # float / complex / signed & unsigned int
        if a.size == 0:
            info.update(min=None, max=None, mean=None, nan=0)
        else:
            info.update(
                min=float(np.nanmin(a)),
                max=float(np.nanmax(a)),
                mean=float(np.nanmean(a)),
                nan=int(np.isnan(a).sum()),
            )
    else:  # string / unicode columns -> show the distinct values
        vals = np.unique(a)
        info["values"] = [str(v) for v in vals[:8]]
        if len(vals) > 8:
            info["values"].append(f"... ({len(vals)} distinct)")
    return info


def save_npz(accs, out_path: Path):
    """Save every topic's columns into one .npz archive."""
    payload = {}
    for topic, acc in accs.items():
        tk = _topic_key(topic)
        for k, a in acc.to_arrays().items():
            payload[f"{tk}.{k}"] = a
    np.savez(out_path, **payload)


# ruff: noqa: UP031
def _fmt8(x) -> str:
    return "%.8g" % x


def save_csv(accs, out_dir: Path):
    """Write one flat CSV per topic."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for topic, acc in accs.items():
        arrays = acc.to_arrays()
        tk = _topic_key(topic)
        path = out_dir / f"{tk}.csv"
        _write_topic_csv(path, tk, arrays)


def _write_topic_csv(path: Path, tk: str, arrays: dict[str, np.ndarray]):
    n = len(arrays["t_recv_ns"])
    if n == 0:
        path.write_text("")
        return
    header: list[str] = []
    cols: list[list[str]] = []
    used: set[str] = set()

    def add_col(name: str, fmt, arr: np.ndarray):
        if name in used:
            return
        used.add(name)
        header.append(name)
        cols.append([fmt(v) for v in arr])

    add_col("t_recv_ns", lambda v: str(int(v)), arrays["t_recv_ns"])
    add_col("t_s", _fmt8, arrays["t_s"])
    add_col("t_stamp_ns", lambda v: str(int(v)), arrays["t_stamp_ns"])
    if "frame_id" in arrays:
        add_col("frame_id", str, arrays["frame_id"])
    for k in sorted(arrays):
        a = arrays[k]
        if k in used or a.ndim != 1 or a.dtype.kind not in "fc":
            continue
        add_col(k, _fmt8, a)
    for k in sorted(arrays):
        a = arrays[k]
        if a.ndim != 2:
            continue
        for j in range(a.shape[1]):
            add_col(f"{k}_{j}", _fmt8, a[:, j])
    for k in sorted(arrays):
        a = arrays[k]
        if k in used or a.ndim != 1 or a.dtype.kind not in "US":
            continue
        add_col(k, str, a)

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(n):
            w.writerow([c[i] for c in cols])


# ---------------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------------
def extract_bag(
    bag_dir,
    store,
    topics: set[str] | None = None,
    out_root=None,
    fmt="csv",
    time_coeffs: float | None = None,
    max_messages: int | None = None,
    summary_only: bool = False,
):
    """Extract a single bag; print a summary and write outputs."""
    from rosbags.rosbag2 import Reader

    bag_dir = Path(bag_dir)
    bag_name = bag_dir.name
    out_root = (
        Path(out_root)
        if out_root
        else Path(__file__).resolve().parent.parent / "bag_data"
    )
    out_bag = out_root / bag_name

    # --- summary-only path (no deserialization needed) ---
    if summary_only:
        reader = Reader(str(bag_dir))
        reader.open()
        try:
            meta = {
                "start_time_ns": reader.start_time,
                "end_time_ns": reader.end_time,
                "duration_ns": reader.duration,
            }
            counts = {c.topic: (c.msgcount, c.msgtype) for c in reader.connections}
        finally:
            reader.close()
        _print_summary_only(bag_dir, meta, counts, topics)
        return None

    accs, meta = read_bag(bag_dir, topics or set(), store, max_messages)
    summary = _stack_summary(accs, meta)
    summary["bag"] = bag_name
    # 录制时的 time_coeffs（fourier_trajectory 时间系数）写入 summary.json，
    # 供下游（fourier_fit / plot_residual / compare_torque）直接读取。
    if time_coeffs is not None:
        summary["time_coeffs"] = float(time_coeffs)

    # console summary
    _print_summary(bag_name, summary)

    if fmt in ("npz", "both"):
        out_bag.mkdir(parents=True, exist_ok=True)
        save_npz(accs, out_bag / "data.npz")
    if fmt in ("csv", "both"):
        save_csv(accs, out_bag / "csv")
    if fmt in ("npz", "csv", "both"):
        out_bag.mkdir(parents=True, exist_ok=True)
        with open(out_bag / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"  -> wrote outputs to {out_bag}")

    return accs


def _print_summary_only(bag_dir, meta, counts, topics):
    print(f"\n=== {bag_dir.name} (summary only) ===")
    if meta.get("start_time_ns") and meta.get("end_time_ns"):
        print(
            f"  duration: {(meta['end_time_ns'] - meta['start_time_ns']) / 1e9:.3f} s"
        )
    for topic, (count, msgtype) in sorted(counts.items()):
        if topics and topic not in topics:
            continue
        print(f"  {topic:38s} {msgtype:42s} {count:8d} msgs")


def _print_summary(bag_name, summary):
    print(f"\n=== {bag_name} ===")
    print(f"  duration: {summary['duration_s']} s")
    for topic, info in summary["topics"].items():
        rate = f"{info['rate_hz']:.1f} Hz" if info["rate_hz"] else "-"
        print(f"  {topic:38s} {info['msgtype']:42s} n={info['count']:7d}  {rate}")
        for k, fi in info["fields"].items():
            if fi["kind"] in "fciu":
                print(
                    f"      {k:28s} {fi['dtype']:10s} shape={fi['shape']} "
                    f"min={fi['min']:.5g} max={fi['max']:.5g} mean={fi['mean']:.5g} nan={fi['nan']}"
                )
            else:
                print(
                    f"      {k:28s} {fi['dtype']:10s} shape={fi['shape']} values={fi.get('values')}"
                )


# ---------------------------------------------------------------------------
# Discovery / CLI
# ---------------------------------------------------------------------------
def discover_bags(path) -> list[Path]:
    """Return bag directories (those containing a metadata.yaml)."""
    p = Path(path).expanduser().resolve()
    if p.is_file():
        if p.name == "metadata.yaml":
            return [p.parent]
        if p.suffix == ".db3":
            return [p.parent]
        raise FileNotFoundError(f"Unrecognised file: {p}")
    if (p / "metadata.yaml").exists():
        return [p]
    bags = sorted(d for d in p.iterdir() if (d / "metadata.yaml").exists())
    if not bags:
        raise FileNotFoundError(f"No rosbag2 directories found under {p}")
    return bags


def _parse_topics(arg: str | None) -> set[str] | None:
    if not arg:
        return None
    return {t.strip() for t in arg.split(",") if t.strip()}


def main(argv=None):
    default_bags = Path(__file__).resolve().parent.parent / "bags"
    parser = argparse.ArgumentParser(
        description="Decode recorded rosbag2 data into npz/csv files."
    )
    parser.add_argument(
        "bags",
        nargs="*",
        default=[str(default_bags)],
        help="bag dirs, a directory of bags, or a .db3 file (default: %(default)s)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="output root directory (default: src/identification/bag_data)",
    )
    parser.add_argument(
        "--topics",
        default=None,
        help="comma separated topic subset, e.g. /hardware/joint_state",
    )
    parser.add_argument("--format", choices=["npz", "csv", "both"], default="csv")
    parser.add_argument(
        "--time-coeffs",
        type=float,
        default=None,
        help="录制时 fourier_trajectory 的时间系数（如 0.75），写入 summary.json 供下游读取",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=None,
        help="stop after N messages per topic (for quick checks)",
    )
    parser.add_argument(
        "--summary-only", action="store_true", help="print summary only, write nothing"
    )
    parser.add_argument(
        "--msg-dir",
        default=None,
        help="read .msg definitions from this directory instead of the embedded recording-era ones",
    )
    args = parser.parse_args(argv)

    # expand bag inputs into concrete bag directories
    bag_dirs: list[Path] = []
    for raw in args.bags:
        bag_dirs.extend(discover_bags(raw))
    bag_dirs = list(dict.fromkeys(bag_dirs))  # de-duplicate, keep order

    topics = _parse_topics(args.topics)

    # Default topic selection: every interface_protocol topic present in the bag.
    if topics is None:
        topics = set()
        from rosbags.rosbag2 import Reader

        for bd in bag_dirs:
            r = Reader(str(bd))
            r.open()
            try:
                topics.update(
                    c.topic
                    for c in r.connections
                    if c.msgtype.startswith("interface_protocol/")
                )
            finally:
                r.close()

    if args.summary_only:
        for bd in bag_dirs:
            extract_bag(bd, store=None, topics=topics, summary_only=True)
        return 0

    # Building the typestore requires `rosbags`.
    try:
        store, defs = build_typestore(args.msg_dir)
    except ImportError as exc:  # pragma: no cover
        print(
            "Missing dependency 'rosbags'. Install it with:  pip install rosbags",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    missing = [
        t
        for t in topics
        if not any(t == c.topic for bd in bag_dirs for c in _conns(bd))
    ]
    if missing:
        print(f"Warning: topics not found in any bag: {missing}", file=sys.stderr)

    for bd in bag_dirs:
        extract_bag(
            bd,
            store,
            topics,
            args.out_dir,
            args.format,
            args.max_messages,
            time_coeffs=args.time_coeffs,
        )

    print("\nDone.")
    return 0


def _conns(bag_dir):
    from rosbags.rosbag2 import Reader

    r = Reader(str(bag_dir))
    r.open()
    try:
        return r.connections
    finally:
        r.close()


if __name__ == "__main__":
    raise SystemExit(main())
