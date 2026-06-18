import numpy as np

from identification.fourier_trajectory import FourierTrajectory
from identification.target_limb_regressor import TargetLimbRegressor
from identification.bayesianJFA import VariationalBayesianJFA

from sko.PSO import PSO
from sko.tools import set_run_mode
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

LEFT_LEG_Q_INDICES  = [0, 1, 2, 3, 4, 5]
RIGHT_LEG_Q_INDICES = [6, 7, 8, 9, 10, 11]
WAIST_Q_INDICES     = [12]
LEFT_ARM_Q_INDICES  = [13, 14, 15, 16, 17]
RIGHT_ARM_Q_INDICES = [18, 19, 20, 21, 22]
NECK_Q_INDICES      = [23]

VALID_LIMB_GROUPS = {
    'left_leg': LEFT_LEG_Q_INDICES,
    'right_leg': RIGHT_LEG_Q_INDICES,
    'left_arm': LEFT_ARM_Q_INDICES,
    'right_arm': RIGHT_ARM_Q_INDICES,
    'waist': WAIST_Q_INDICES,
    'neck': NECK_Q_INDICES
}

GROUP_TO_IDENTIFY = 'left_arm' 

URDF_PATH = (
    Path(get_package_share_directory('identification'))
    / "resource"
    / "robot"
    / "urdf"
    / "serial_pm_v2_identify.urdf"
).resolve()

# ==============================
# Tunable trajectory configuration
# ==============================

# Series expansion parameters
N_HARMONICS = 5
TRAJ_PERIOD = 10.0
SAMPLE_RATE   = 50.0

# Soft constraint parameters
REG_EPS = 1e-6
RANK_REL_TOL = 1e-4
RANK_ABS_TOL = 1e-10

# ==============================
# PSO parameters
# ==============================
POP_SIZE = 100
MAX_ITER = 500
PSO_W    = 0.7
PSO_C1   = 1.5
PSO_C2   = 1.5

# Normalized penalty weights
PENALTY_W_Q = 20
PENALTY_W_V = 10
PENALTY_W_TAU = 20
PENALTY_W_MAX = 50
PENALTY_W_COLLISION = 1000

REWARD_ACTIVE = 30.0
REWARD_WEAKLY_ACTIVE = 20.0

# Progressive penalty schedule: lambda(k) = lambda0 * (1 + alpha * progress)
PENALTY_LAMBDA0 = 400.0
PENALTY_LAMBDA_ALPHA = 2.0

# Limit buffers
Q_LIMIT_BUFFER = 0.1
V_LIMIT_BUFFER = 0.1
TAU_LIMIT_BUFFER = 0.1

RNG_SEED = 114

class DCondition:
    def __init__(self, X, Lambda):
        self.X = np.array(X)
        self.M = self.X.T @ self.X
        self.Lambda = Lambda * np.eye(self.M.shape[0])
        self.M_reg = self.M + self.Lambda