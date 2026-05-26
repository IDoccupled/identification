#!/usr/bin/env python3

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np # type: ignore
import pinocchio as pin

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
    Path(__file__).resolve().parent
    / ".."
    / ".."
    / "simulation"
    / "mujoco"
    / "assets"
    / "resource"
    / "robot"
    / "pm_v2"
    / "urdf"
    / "serial_pm_v2_identify.urdf"
).resolve()

RNG_SEED = 42


class TargetLimbRegressor:
    def __init__(
        self,
        urdf_path: Path = URDF_PATH,
        group_to_identify = GROUP_TO_IDENTIFY
    ):
        if group_to_identify not in VALID_LIMB_GROUPS:
            raise ValueError(
                f"Invalid group_to_identify: {group_to_identify}. "
                f"Must be one of: {list(VALID_LIMB_GROUPS.keys())}"
            )
        if not urdf_path.is_file():
            raise FileNotFoundError(f"URDF file not found at: {urdf_path}")
        
        self.urdf_path = Path(urdf_path).resolve()
        self.group_to_identify = list(VALID_LIMB_GROUPS[group_to_identify])
        self.rng_seed = int(RNG_SEED)

        self.model = self._load_model(self.urdf_path)
        self.data = self.model.createData()
        self.urdf_dynamics = self._load_urdf_joint_dynamics(self.urdf_path)
        self.joint_infos, self.target_v_indices = self.collect_target_limb_info()

        self.limits = {
            "q_lower": self.model.lowerPositionLimit[self.group_to_identify],
            "q_upper": self.model.upperPositionLimit[self.group_to_identify],
            "v_limit": self.model.velocityLimit[self.target_v_indices],
            "effort_limit": self.model.effortLimit[self.target_v_indices],
        }

    @staticmethod
    def _fmt_array(arr: np.ndarray) -> str:
        arr = np.asarray(arr, dtype=float).reshape(-1)
        return "[" + ", ".join(f"{x:.6g}" for x in arr) + "]"

    @staticmethod
    def _load_model(urdf_path: Path) -> pin.Model:
        if not urdf_path.is_file():
            raise FileNotFoundError(f"URDF file not found at: {urdf_path}")
        return pin.buildModelFromUrdf(str(urdf_path))

    @staticmethod
    def _load_urdf_joint_dynamics(urdf_path: Path):
        """Load damping/friction values from URDF joint dynamics tags."""
        tree = ET.parse(str(urdf_path))
        root = tree.getroot()

        dynamics_by_joint = {}
        for joint_elem in root.findall("joint"):
            name = joint_elem.attrib.get("name")
            if not name:
                continue

            dyn_elem = joint_elem.find("dynamics")
            damping = 0.0
            friction = 0.0
            if dyn_elem is not None:
                damping = float(dyn_elem.attrib.get("damping", "0.0") or 0.0)
                friction = float(dyn_elem.attrib.get("friction", "0.0") or 0.0)

            dynamics_by_joint[name] = {
                "damping": damping,
                "friction": friction,
            }
        return dynamics_by_joint
    
    def state_size_check_and_form(self, q, v, a):
        n = len(self.group_to_identify)
        if len(q) != n:
            raise ValueError(f"Expected q of length {n}, got {len(q)}")
        if len(v) != n:
            raise ValueError(f"Expected v of length {n}, got {len(v)}")
        if len(a) != n:
            raise ValueError(f"Expected a of length {n}, got {len(a)}")
        formed_q = np.zeros(self.model.nq)
        formed_v = np.zeros(self.model.nv)
        formed_a = np.zeros(self.model.nv)
        for i, q_idx in enumerate(self.group_to_identify):
            formed_q[q_idx] = q[i]
        for i, v_idx in enumerate(self.target_v_indices):
            formed_v[v_idx] = v[i]
            formed_a[v_idx] = a[i]
        return formed_q, formed_v, formed_a
        

    def collect_target_limb_info(self):
        target_q_set = set(self.group_to_identify)
        infos = []
        target_v_indices = []

        for joint_id in range(1, self.model.njoints):
            joint = self.model.joints[joint_id]
            q_range = list(range(joint.idx_q, joint.idx_q + joint.nq))
            if target_q_set.intersection(q_range):
                v_range = list(range(joint.idx_v, joint.idx_v + joint.nv))
                infos.append(
                    {
                        "joint_id": joint_id,
                        "name": self.model.names[joint_id],
                        "idx_q": joint.idx_q,
                        "nq": joint.nq,
                        "idx_v": joint.idx_v,
                        "nv": joint.nv,
                        "q_range": q_range,
                        "v_range": v_range,
                        "q_lower": self.model.lowerPositionLimit[joint.idx_q : joint.idx_q + joint.nq].copy(),
                        "q_upper": self.model.upperPositionLimit[joint.idx_q : joint.idx_q + joint.nq].copy(),
                        "v_limit": self.model.velocityLimit[joint.idx_v : joint.idx_v + joint.nv].copy(),
                        "effort_limit": self.model.effortLimit[joint.idx_v : joint.idx_v + joint.nv].copy(),
                    }
                )
                target_v_indices.extend(v_range)

        target_v_indices = sorted(set(target_v_indices))
        infos.sort(key=lambda x: x["idx_q"])
        return infos, target_v_indices

    def sample_state(self, target_v_indices):
        rng = np.random.default_rng(self.rng_seed)
        
        q = pin.neutral(self.model)
        for q_idx in self.group_to_identify:
            if q_idx < 0 or q_idx >= self.model.nq:
                raise IndexError(f"Target q index out of range: {q_idx} (nq={self.model.nq})")

            low = self.model.lowerPositionLimit[q_idx]
            high = self.model.upperPositionLimit[q_idx]
            q[q_idx] = rng.uniform(low, high)

        v = np.zeros(self.model.nv)
        a = np.zeros(self.model.nv)

        if target_v_indices:
            v[target_v_indices] = rng.normal(0.0, 0.3, size=len(target_v_indices))
            a[target_v_indices] = rng.normal(0.0, 0.5, size=len(target_v_indices))

        return q, v, a

    def build_augmented_target_regressor(
        self,
        joint_infos,
        target_v_indices,
        Y_target_limb,
        v,
    ):
        """
        Build target-limb regressor with 12 columns per 1-DoF joint:
        - 10 inertial columns from Pinocchio's per-joint inertial block
        - 2 friction columns [v_i, sign(v_i)]
        """

        # Remove unrelated trunk columns by keeping only target-joint inertial blocks.
        inertial_blocks = []
        target_joint_names = []
        for info in joint_infos:
            joint_id = int(info["joint_id"])
            col_begin = 10 * (joint_id - 1)
            col_end = col_begin + 10
            inertial_blocks.append(Y_target_limb[:, col_begin:col_end])
            target_joint_names.append(info["name"])

        Y_target_inertial = np.hstack(inertial_blocks)

        n_rows = len(target_v_indices)
        n_target_joints = len(joint_infos)
        Y_friction = np.zeros((n_rows, 2 * n_target_joints))

        row_by_v_idx = {v_idx: row_idx for row_idx, v_idx in enumerate(target_v_indices)}
        friction_params_from_urdf = np.zeros(2 * n_target_joints)

        for joint_local_idx, info in enumerate(joint_infos):
            v_idx = int(info["idx_v"])
            row_idx = row_by_v_idx[v_idx]

            # Friction model: tau_f = fv * v + fc * sign(v)
            Y_friction[row_idx, 2 * joint_local_idx] = v[v_idx]
            Y_friction[row_idx, 2 * joint_local_idx + 1] = np.tanh(v[v_idx]*100)

            dyn = self.urdf_dynamics.get(info["name"], {"damping": 0.0, "friction": 0.0})
            friction_params_from_urdf[2 * joint_local_idx] = float(dyn["damping"])
            friction_params_from_urdf[2 * joint_local_idx + 1] = float(dyn["friction"])

        Y_aug = np.hstack([Y_target_inertial, Y_friction])
        return Y_aug, Y_target_inertial, Y_friction, target_joint_names, friction_params_from_urdf

    def compute_regressor(self, 
                          q=None, 
                          v=None, 
                          a=None, 
                          print_info=False
                          ):

        if q is None or v is None or a is None:
            print("\nNo state provided, sampling random state within limits for target limb...") if print_info else None
            q, v, a = self.sample_state(self.target_v_indices)
        else:
            print("\nUsing provided state for regressor computation...") if print_info else None
            q, v, a = self.state_size_check_and_form(q, v, a)

        # Y satisfies tau = Y * pi, where pi is the stacked inertial parameter vector.
        Y = pin.computeJointTorqueRegressor(self.model, self.data, q, v, a)
        Y_target_limb = Y[self.target_v_indices, :]
        (
            Y_aug,
            Y_target_inertial,
            Y_friction,
            target_joint_names,
            friction_params_from_urdf,
        ) = self.build_augmented_target_regressor(
            joint_infos=self.joint_infos,
            target_v_indices=self.target_v_indices,
            Y_target_limb=Y_target_limb,
            v=v,
        )

        tau_inertia = pin.rnea(self.model, self.data, q, v, a)[self.target_v_indices]
        tau_friction_from_urdf = Y_friction @ friction_params_from_urdf

        tau_aug = tau_inertia + tau_friction_from_urdf

        if print_info:

            np.set_printoptions(precision=6, suppress=True, linewidth=220, threshold=np.inf)
            print(f"Loading URDF: {self.urdf_path}")
            print(
                f"Model loaded: nq={self.model.nq}, nv={self.model.nv}, njoints={self.model.njoints}"
            )

            print("\n=== Target limb joint parameters ===")
            for info in self.joint_infos:
                print(
                    f"joint_id={info['joint_id']:>2d} name={info['name']} "
                    f"idx_q={info['idx_q']} nq={info['nq']} idx_v={info['idx_v']} nv={info['nv']}"
                )
                print(f"  q_range      = {info['q_range']}")
                print(f"  v_range      = {info['v_range']}")
                print(f"  q_lower      = {self._fmt_array(info['q_lower'])}")
                print(f"  q_upper      = {self._fmt_array(info['q_upper'])}")
                print(f"  velocity_lim = {self._fmt_array(info['v_limit'])}")
                print(f"  effort_lim   = {self._fmt_array(info['effort_limit'])}")

            print("\n=== Regressor ===")
            print(
                f"Current state (q, v, a) for target limb: \n"
                f"  q={self._fmt_array(q)} \n"
                f"  v={self._fmt_array(v)} \n"
                f"  a={self._fmt_array(a)} \n"
            )
            print(f"Full regressor shape: {Y.shape}")
            print(f"Target limb row indices: {self.target_v_indices}")
            print(f"Target joint names (column blocks order): {target_joint_names}")
            print(f"Target limb raw regressor shape: {Y_target_limb.shape}")
            print(
                f"Target inertial-only shape: "
                f"{Y_target_inertial.shape}"
            )
            print(f"Target friction regressor shape: {Y_friction.shape}")
            print(f"Per-joint columns: 12 (= 10 inertial + 2 friction)")
            print(
                f"URDF friction params [fv1, fc1, fv2, fc2, ...]: "
                f"{self._fmt_array(friction_params_from_urdf)}"
            )
            
            print(
                "URDF friction torque contribution on target rows: "
                f"{self._fmt_array(tau_friction_from_urdf)}"
            )
            print(
                "Inertial torque contribution on target rows: "
                f"{self._fmt_array(tau_inertia)}"
            )
            print(
                "Augmented torque contribution on target rows: "
                f"{self._fmt_array(tau_aug)}"
            )

            print(f"Augmented regressor shape: {Y_aug.shape}")
            print("Augmented regressor (inertial columns + friction columns):")
            print(Y_aug)

        return Y_aug, tau_aug
    
'''
VALID_LIMB_GROUPS = {
    'left_leg': LEFT_LEG_Q_INDICES,
    'right_leg': RIGHT_LEG_Q_INDICES,
    'left_arm': LEFT_ARM_Q_INDICES,
    'right_arm': RIGHT_ARM_Q_INDICES,
    'waist': WAIST_Q_INDICES,
    'neck': NECK_Q_INDICES
}
'''

def main():
    regressor = TargetLimbRegressor(
        urdf_path=URDF_PATH,
        group_to_identify=GROUP_TO_IDENTIFY
    )
    regressor.compute_regressor(print_info=True)


if __name__ == "__main__":
    main()
