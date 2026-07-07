#!/usr/bin/env python3
import argparse
import sys
import re
from pathlib import Path
import numpy as np
import yaml
from datetime import datetime

RAW_INPUT = """
 [ 0.34303679 -0.34303679 -0.12809646  0.64178438  1.02911036 -1.02911036
 -1.37214714 -1.37214714  0.1661353   1.71518393  1.2776     -0.16757255
  0.16757255 -0.3351451  -0.3351451   0.50271766 -0.50271766 -0.67029021
  0.67029021 -0.83786276 -0.83786276  1.53945    -0.23269492 -0.31013803
 -0.62027605 -0.62027605 -0.93041408 -0.93041408  1.18238874 -1.24055211
  1.55069013  0.13958418 -1.234       0.165386   -0.165386    0.27772398
 -0.33077201 -0.49615801 -0.30215685 -0.66154401  0.66154401 -0.66281975
  0.82693002 -0.07446467  0.31013803  0.31013803  0.62027605  0.62010561
 -0.93041408 -0.93041408 -1.24055211  0.49849833  1.55069013 -1.55069013
 -1.234     ]

"""

OUTPUT_NAME = f"pso_friction_{datetime.now().strftime('%y%m%d_%H%M%S')}"

OUTPUT_DESC = "Iter: 89, Best fit: [-101.87915795]"

DEFAULT_DIM = 5
DEFAULT_N_HARMONICS = 5


def parse_flat_array(text: str) -> np.ndarray:
    """
    Parse a flat array from a string, handling various input formats:
      - "[0.1, -0.2, ...]"
      - "0.1 -0.2 0.3 ..."
      - "0.1, -0.2, 0.3 ..."
      - Mixed content with "Best fit:" etc.
    """
    # Remove everything before and including '[' or 'at [' or 'Best fit:'
    # Then find the actual numeric content
    text = text.strip()

    # Try to extract content within square brackets
    bracket_match = re.search(r"\[([^\]]+)\]", text)
    if bracket_match:
        text = bracket_match.group(1)

    # Replace commas and newlines with spaces, then split
    text = re.sub(r"[\n\r,]+", " ", text)
    tokens = text.strip().split()

    values = []
    for t in tokens:
        try:
            values.append(float(t))
        except ValueError:
            pass  # skip non-numeric tokens

    return np.array(values)


def flat_to_yaml(
    flat_array: np.ndarray,
    dim: int = DEFAULT_DIM,
    n_harmonics: int = DEFAULT_N_HARMONICS,
    description: str = "",
) -> dict:
    """
    Convert a flat Fourier coefficient array into a structured dict
    suitable for YAML serialization.

    The flat array layout for each joint: [a1, b1, a2, b2, ..., aN, bN, q0]
    """
    expected = dim * (n_harmonics * 2 + 1)
    if flat_array.size != expected:
        raise ValueError(
            f"Expected {expected} values for dim={dim}, n_harmonics={n_harmonics}, "
            f"but got {flat_array.size}"
        )

    params = flat_array.reshape(dim, n_harmonics * 2 + 1)
    data = {}
    for i in range(dim):
        a = params[i, 0 : n_harmonics * 2 : 2]
        b = params[i, 1 : n_harmonics * 2 : 2]
        q0 = params[i, -1]
        data[f"joint_{i}"] = {
            "a": [round(float(v), 8) for v in a.tolist()],
            "b": [round(float(v), 8) for v in b.tolist()],
            "q0": round(float(q0), 8),
        }
    return data


def array_to_yaml_text(
    flat_array: np.ndarray,
    dim: int = DEFAULT_DIM,
    n_harmonics: int = DEFAULT_N_HARMONICS,
    description: str = "",
) -> str:
    """Convert a flat array to YAML string with header comment."""
    data = flat_to_yaml(flat_array, dim, n_harmonics, description)
    lines = [f"# dim={dim}, n_harmonics={n_harmonics}"]
    if description:
        lines.append(f"# {description}")
    lines.append("")
    lines.append(yaml.dump(data, default_flow_style=None, sort_keys=False).strip())
    return "\n".join(lines) + "\n"


def save_yaml(
    flat_array: np.ndarray,
    name: str,
    dim: int = DEFAULT_DIM,
    n_harmonics: int = DEFAULT_N_HARMONICS,
    description: str = "",
    output_dir: str | None = None,
) -> Path:
    """Save a flat array as a structured YAML file in trajectory_coefficients/."""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not name.endswith(".yaml"):
        name += ".yaml"
    out_path = output_dir / name

    yaml_text = array_to_yaml_text(flat_array, dim, n_harmonics, description)
    out_path.write_text(yaml_text)
    print(f"Saved to: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert flat Fourier coefficient arrays to structured YAML."
    )
    parser.add_argument(
        "values",
        nargs="?",
        help="Flat array values (comma/space separated, with or without brackets)",
    )
    parser.add_argument(
        "--name", default=None, help="Output YAML filename (without .yaml)"
    )
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM, help="Number of joints")
    parser.add_argument(
        "--n_harmonics",
        type=int,
        default=DEFAULT_N_HARMONICS,
        help="Number of harmonics",
    )
    parser.add_argument("--desc", default="", help="Optional description comment")
    parser.add_argument(
        "--dir",
        default=None,
        help="Output directory (default: this script's directory)",
    )
    args = parser.parse_args()

    # --- Determine input source: RAW_INPUT > CLI arg > stdin ---
    if RAW_INPUT:
        flat = parse_flat_array(RAW_INPUT)
    elif args.values:
        flat = parse_flat_array(args.values)
    else:
        print("Paste the flat array (end with Ctrl+D or Ctrl+Z):")
        text = sys.stdin.read()
        flat = parse_flat_array(text)

    if flat.size == 0:
        print("Error: no numeric values found.", file=sys.stderr)
        sys.exit(1)

    # --- Determine output name: OUTPUT_NAME > CLI --name > prompt ---
    name = OUTPUT_NAME or args.name
    if not name:
        name = input("Output filename (without .yaml): ").strip()
    if not name:
        name = "trajectory"
        print(f"Using default name: {name}")

    # --- Description: OUTPUT_DESC > CLI --desc ---
    desc = OUTPUT_DESC or args.desc

    expected = args.dim * (args.n_harmonics * 2 + 1)
    if flat.size != expected:
        print(
            f"Warning: got {flat.size} values, expected {expected} "
            f"(dim={args.dim}, n_harmonics={args.n_harmonics})",
            file=sys.stderr,
        )

    save_yaml(flat, name, args.dim, args.n_harmonics, desc, args.dir)
    print(f"\nPreview of {name}.yaml:")
    print(array_to_yaml_text(flat, args.dim, args.n_harmonics, desc))


if __name__ == "__main__":
    main()
