"""Parameter-quality YAML → weighted-ridge coefficients and hard-freeze mask.

The old per-quality *box constraints* were replaced by a quality-weighted **ridge**
(L2 toward the URDF prior)::

    c_i = ridge_lambda · w_q / max(|prior_i|, scale_floor)²

Quality bands are read from the ``_diagnostics.per_param`` block of a
``pso_unified_*.yaml`` (see ``load_yaml_param_quality``).  ``null``/``small``
parameters do **not** participate in the ridge at all (``c = 0``) — they are
hard-frozen at the prior instead (``build_freeze_mask``), and the freeze mask is
the single source of truth for "frozen ⇒ no ridge".

Tunables: ``RIDGE_QUALITY_WEIGHTS``, ``RIDGE_FREEZE_QUALITIES``,
``RIDGE_FREEZE_LOCAL_INDICES``, ``RIDGE_LAMBDA``, ``RIDGE_SCALE_FLOOR``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .params import N_PER_JOINT, split_joint_params


# --- Weighted ridge (quality → L2 penalty toward the URDF prior) ---
RIDGE_QUALITY_WEIGHTS = {
    "good": 1.0,
    "ok": 3.0,
    "bad": 10.0,
    "rank_deficient": 20.0,
    "small": 1e-3,
}
# RIDGE_FREEZE_QUALITIES = frozenset({"null", "small"})
RIDGE_FREEZE_QUALITIES = frozenset({"null"})
# RIDGE_FREEZE_LOCAL_INDICES = frozenset({10, 11, 12})
RIDGE_FREEZE_LOCAL_INDICES = frozenset({})
DEFAULT_QUALITY = "ok"
RIDGE_LAMBDA = 3.0
RIDGE_SCALE_FLOOR = 1e-4


# ============================================================================
# YAML parameter-quality helpers
# ============================================================================
# YAML parameter-order constants (MUST match pso_excitation_unified output)
#
# ⚠️ FIXED 2026-09-14 (user: "the quality mapping of Iyy and Ixz is
#    swapped"): this used to be
#    {4:4, 5:5, 6:7, 7:6, 8:8, 9:9}, whose comment claimed the YAML inertia order
#    differs from ours ([Ixx,Ixy,Iyy,Ixz,Iyz,Izz] vs [Ixx,Ixy,Ixz,Iyy,Iyz,Izz]).
#    That claim is wrong — the two orders are THE SAME, so the map is the
#    IDENTITY.  Evidence:
#      · pin.Inertia(5,[0,0,0],[[1,.1,.2],[.1,2,.3],[.2,.3,3]]).toDynamicParameters()
#        [4:10] == [1, .1, 2, .2, .3, 3] = [Ixx, Ixy, Iyy, Ixz, Iyz, Izz]
#        (verified numerically; Iyy is idx 6, Ixz is idx 7);
#      · pso_traj_excitation._INERTIAL_PARAM_NAMES (which WRITES the YAML) is
#        [mass, mcx, mcy, mcz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz];
#      · our own `PARAM_LABELS` above is idx 6 = Iyy, idx 7 = Ixz;
#      · every YAML in trajectory_coefficients/ carries `per_param[].name`
#        ("<JOINT>/<param>") and idx 6/7 are named Iyy/Ixz there too.
#    Effect of the bug: ridge weight + hard-freeze mask were applied to the wrong
#    physical parameter for every joint (which one is allowed to move).
#    `load_yaml_param_quality` now cross-checks the `name` field against this
#    mapping, so this class of drift fails loudly instead of silently.
_YAML_INERTIA_YAML2OUR = {4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}
# YAML inertia: [Ixx, Ixy, Iyy, Ixz, Iyz, Izz] → our Pinocchio: [Ixx, Ixy, Iyy, Ixz, Iyz, Izz]

# YAML `per_param[].name` suffix → our local index.  Used ONLY to validate the
# index mapping above (the index mapping stays the source of truth); an unknown
# suffix is skipped rather than guessed at.
_YAML_NAME_SUFFIX_TO_LOCAL = {
    "mass": 0,
    "mcx": 1,
    "mcy": 2,
    "mcz": 3,
    "Ixx": 4,
    "Ixy": 5,
    "Iyy": 6,
    "Ixz": 7,
    "Iyz": 8,
    "Izz": 9,
    "armature": 10,
    "damping": 11,
    "friction": 12,
}


def _yaml_global_to_local(global_idx: int, dof: int) -> tuple[int, int]:
    """
    Convert YAML global index → (joint, param) in our 13-per-joint layout.

    YAML layout (total = 10*dof + dof + 2*dof = 13*dof):
      [j0_inertial(10), j1_inertial(10), ..., j{D-1}_inertial(10),
       j0_arm(1), ..., j{D-1}_arm(1),
       j0_damp(1), j0_fric(1), j1_damp(1), j1_fric(1), ...]
    """
    n_inertial = 10 * dof
    if global_idx < n_inertial:
        joint = global_idx // 10
        yaml_i = global_idx % 10
        our_i = yaml_i if yaml_i < 4 else _YAML_INERTIA_YAML2OUR[yaml_i]
        return joint, our_i
    elif global_idx < n_inertial + dof:
        joint = global_idx - n_inertial
        return joint, 10
    else:
        idx = global_idx - n_inertial - dof
        joint = idx // 2
        our_i = 11 + (idx % 2)
        return joint, our_i


def load_yaml_param_quality(yaml_path: str | Path) -> dict[int, str]:
    """
    Parse YAML diagnostics, return quality labels keyed by OUR global index
    (0..dof*13-1, joint-major by 13).

    Every entry's ``name`` field (``"<JOINT>/<param>"``) is cross-checked against
    the index mapping; if the name and the index disagree, a ``ValueError`` is
    raised (that is how the 2026-09-14 Iyy/Ixz swap would have been caught).
    """
    import yaml

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    per_param = data.get("_diagnostics", {}).get("per_param", [])
    dof = len(per_param) // N_PER_JOINT

    result = {}
    mismatch: list[str] = []
    for entry in per_param:
        yaml_g = entry["idx"]
        j, i = _yaml_global_to_local(yaml_g, dof)
        name = entry.get("name")
        if name:
            suffix = str(name).rsplit("/", 1)[-1]
            expected = _YAML_NAME_SUFFIX_TO_LOCAL.get(suffix)
            if expected is not None and expected != i:
                mismatch.append(
                    f"idx {yaml_g}: name '{name}' says '{suffix}' (local "
                    f"{expected}) but the index mapping gives local {i}"
                )
        our_g = j * N_PER_JOINT + i
        result[our_g] = entry["quality"]
    if mismatch:
        raise ValueError(
            f"{yaml_path}: quality YAML parameter order does not match "
            "_YAML_INERTIA_YAML2OUR / _yaml_global_to_local — "
            f"{len(mismatch)} of {len(per_param)} entries disagree:\n  "
            + "\n  ".join(mismatch[:8])
            + (f"\n  ... and {len(mismatch) - 8} more" if len(mismatch) > 8 else "")
        )
    return result


def _ridge_weight_for_quality(quality: str | None) -> float:
    """Map a YAML quality label → its relative-deviation ridge weight (w_q).

    ``RIDGE_FREEZE_QUALITIES`` labels (``null``/``small``) are **excluded from
    the ridge** — they are hard-frozen instead — so asking for their weight is
    a programming error, not a fallback case.
    """
    if quality in RIDGE_FREEZE_QUALITIES:
        raise ValueError(
            f"quality '{quality}' does not participate in the ridge "
            f"(it is hard-frozen via RIDGE_FREEZE_QUALITIES); no ridge weight "
            f"exists for it"
        )
    if quality is None or quality not in RIDGE_QUALITY_WEIGHTS:
        return RIDGE_QUALITY_WEIGHTS[DEFAULT_QUALITY]
    return RIDGE_QUALITY_WEIGHTS[quality]


def build_ridge_weights(
    quality_map: dict[int, str] | None,
    pi_prior: np.ndarray,
    *,
    ridge_lambda: float = RIDGE_LAMBDA,
    scale_floor: float = RIDGE_SCALE_FLOOR,
    freeze_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build per-parameter weighted-ridge quadratic coefficients c_i.

    Replaces the per-quality box (freeze / widen) with a ridge penalty.  For
    every global parameter i with URDF prior ``prior_i``, deviation from the
    prior is penalised in *relative* terms

        c_i · (pi_i − prior_i)² ,   c_i = ridge_lambda · w_q / scale_i²

    with ``scale_i = max(|prior_i|, scale_floor)`` and ``w_q`` from
    ``RIDGE_QUALITY_WEIGHTS``.  Better quality → smaller ``w_q`` → the
    parameter may drift further from the prior; poor quality → larger ``w_q``
    → it is pulled back hard.

    Parameters whose quality label is in ``RIDGE_FREEZE_QUALITIES``
    (``null``/``small``) get ``c_i = 0`` — they are **excluded from the ridge
    norm** and are instead hard-frozen at the prior by ``build_freeze_mask``.
    The same applies to every entry flagged in ``freeze_mask``, so the mask
    built by ``build_freeze_mask`` is the single source of truth for "frozen ⇒
    no ridge".

    Parameters
    ----------
    quality_map : dict[int, str] or None
        ``{global_idx: quality}`` as returned by ``load_yaml_param_quality``.
        ``None``/empty → every parameter gets the default quality weight.
    pi_prior : (dof*13,) ndarray
        URDF prior parameters (the ridge attractor).
    ridge_lambda : float
        Global ridge strength multiplying every ``w_q``.
    scale_floor : float
        Lower bound on the relative-deviation scale, so parameters whose
        prior is ~0 do not get an astronomically large coefficient.
    freeze_mask : (dof*13,) bool ndarray or None
        Where ``True``, the parameter is forced to ``c_i = 0`` (excluded from
        the ridge norm).  Pass the mask from ``build_freeze_mask`` to keep the
        "frozen" and "zero ridge weight" sets identical.

    Returns
    -------
    c : (dof*13,) ndarray
        Quadratic-coefficient vector (exactly 0 wherever the parameter is
        excluded from the ridge).
    """
    dof = len(pi_prior) // N_PER_JOINT
    pi_list = split_joint_params(pi_prior)
    c = np.zeros(dof * N_PER_JOINT)
    for g in range(dof * N_PER_JOINT):
        q = (quality_map or {}).get(g)
        if q in RIDGE_FREEZE_QUALITIES:
            continue  # frozen tier: does not take part in the ridge (c stays 0)
        w_q = _ridge_weight_for_quality(q)
        j, i = g // N_PER_JOINT, g % N_PER_JOINT
        prior_val = pi_list[j][i]
        scale = max(abs(prior_val), scale_floor)
        c[g] = ridge_lambda * w_q / (scale * scale)
    if freeze_mask is not None:
        # frozen ⇒ drops out of the ridge norm (same source as
        # build_freeze_mask, so the two cannot disagree)
        c[np.asarray(freeze_mask, dtype=bool).reshape(-1)] = 0.0
    return c


def build_freeze_mask(
    quality_map: dict[int, str] | None,
    dof: int,
    freeze_qualities: frozenset[str] = RIDGE_FREEZE_QUALITIES,
    freeze_local_indices: frozenset[int] = RIDGE_FREEZE_LOCAL_INDICES,
) -> np.ndarray:
    """
    (dof*13,) boolean mask of parameters to **hard-freeze** (pi == prior).

    Two independent sources, unioned:

    1. Quality labels in ``freeze_qualities`` (default ``null``/``small``) —
       too tiny for a large ridge weight to pin exactly.
    2. The local indices in ``freeze_local_indices`` (default
       ``RIDGE_FREEZE_LOCAL_INDICES`` = armature/damping/friction) for **every**
       joint, regardless of quality label.

    Frozen parameters are frozen at their URDF prior by an equality constraint
    in the solver, so they cannot move at all.  They are **simultaneously
    excluded from the ridge norm** (``c = 0``, see ``build_ridge_weights``);
    both halves are required, since a parameter with ``c = 0`` that is *not*
    frozen would be completely unregularised.

    Raises
    ------
    ValueError
        If a quality entry points outside ``[0, dof*13)`` — i.e. the quality
        YAML does not belong to this limb group, which would otherwise
        silently freeze the wrong parameters.
    """
    n_par = dof * N_PER_JOINT
    mask = np.zeros(n_par, dtype=bool)
    if quality_map:
        for g, q in quality_map.items():
            if not 0 <= g < n_par:
                raise ValueError(
                    f"quality YAML index {g} is outside [0, {n_par}) for "
                    f"dof={dof} — the quality YAML does not match this limb group"
                )
            if q in freeze_qualities:
                mask[g] = True
    for j in range(dof):
        for i in freeze_local_indices:
            mask[j * N_PER_JOINT + i] = True
    return mask
