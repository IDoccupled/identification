"""Paths and global setup constants (default URDF/YAML, sim gravity, …).

``PKG_ROOT`` is the ament package directory ``src/identification``; the former
single-file module lived directly in it (``Path(__file__).parent.parent``), so
every path below is now expressed relative to this constant.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[2]
_URDF_DIR = PKG_ROOT / "resource" / "robot" / "urdf"
_COEFFS_DIR = PKG_ROOT / "trajectory_coefficients"

DEFAULT_URDF_PATH = (_URDF_DIR / "serial_pm_v2_identify.urdf").resolve()

TRUE_URDF_PATH = (_URDF_DIR / "serial_pm_v2_identify_true.urdf").resolve()

TRAJ_YAML_PATH = (_COEFFS_DIR / "recovered_exc_arm_1.yaml").resolve()

VAL_YAML_PATH = (_COEFFS_DIR / "verify_left_arm.yaml").resolve()

QUALITY_YAML_PATH = (_COEFFS_DIR / "excite_left_arm.yaml").resolve()

SIM_GRAVITY = np.array([-9.76935626, 0.39412975, -0.98615969])
SIM_WAIST_YAW_OFFSET = 0.02079

# CSV topics of the bag extract: the IMU readings give the gravity vector
# and the joint state gives the fixed waist-joint angle.
IMU_CSV_TOPIC = "hardware_imu_info"

# [meas] Delay of the measured torque channel relative to the state
# (seconds, --tau-delay; 0 = no compensation).  See data_from_measurement:
# > 0 means the torque lags the state, so the regressor is evaluated at t − τ.
TAU_DELAY = 0.0
