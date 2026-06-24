#!/usr/bin/env python3

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin

from ament_index_python.packages import get_package_share_directory

from identification.collision_test import CollisionTest

LEFT_LEG_Q_INDICES = [0, 1, 2, 3, 4, 5]
RIGHT_LEG_Q_INDICES = [6, 7, 8, 9, 10, 11]
WAIST_Q_INDICES = [12]
LEFT_ARM_Q_INDICES = [13, 14, 15, 16, 17]
RIGHT_ARM_Q_INDICES = [18, 19, 20, 21, 22]
NECK_Q_INDICES = [23]

VALID_LIMB_GROUPS = {
    "left_leg": LEFT_LEG_Q_INDICES,
    "right_leg": RIGHT_LEG_Q_INDICES,
    "left_arm": LEFT_ARM_Q_INDICES,
    "right_arm": RIGHT_ARM_Q_INDICES,
    "waist": WAIST_Q_INDICES,
    "neck": NECK_Q_INDICES,
}

GROUP_TO_IDENTIFY = "left_arm"

COLLISION_PAIRS = [
    ("LINK_ELBOW_YAW_L_0", "LINK_BASE_0"),
    ("LINK_ELBOW_YAW_L_0", "LINK_TORSO_YAW_0"),
    ("LINK_ELBOW_YAW_L_0", "LINK_HEAD_YAW_0"),
    ("LINK_ELBOW_YAW_L_0", "LINK_HIP_PITCH_L_0"),
    ("LINK_ELBOW_YAW_L_0", "LINK_HIP_ROLL_L_0"),
    ("LINK_ELBOW_YAW_L_0", "LINK_HIP_YAW_L_0"),
    ("LINK_ELBOW_PITCH_L_0", "LINK_TORSO_YAW_0"),
    ("LINK_SHOULDER_YAW_L_0", "LINK_TORSO_YAW_0"),
]

URDF_PATH = (
    Path(get_package_share_directory("identification"))
    / "resource"
    / "robot"
    / "urdf"
    / "serial_pm_v2_identify.urdf"
).resolve()
PKG_DIR = Path(get_package_share_directory("identification")).resolve()

RNG_SEED = 114


class TargetLimbRegressor:
    def __init__(
        self,
        urdf_path: Path = URDF_PATH,
        group_to_identify=GROUP_TO_IDENTIFY,
        print_info=False,
    ):

        if not urdf_path.is_file():
            raise FileNotFoundError(f"URDF file not found at: {urdf_path}")
        if group_to_identify not in VALID_LIMB_GROUPS:
            raise ValueError(
                f"Invalid group_to_identify: {group_to_identify}. "
                f"Must be one of: {list(VALID_LIMB_GROUPS.keys())}"
            )

        urdf_path = Path(urdf_path).resolve()
        print(f"\033[93mUsing URDF path: {urdf_path}\033[0m") if print_info else None
        self.group_to_identify = list(VALID_LIMB_GROUPS[group_to_identify])

        self.model = self._model_from_urdf(urdf_path)
        self.data = self.model.createData()
        self.urdf_dynamics = self._load_urdf_joint_dynamics(urdf_path)
        self.all_joint_infos, self.target_joint_infos = self.collect_target_limb_info()
        self.dof = len(self.group_to_identify)

        self.limits = {
            "q_lower": self.model.lowerPositionLimit[self.group_to_identify],
            "q_upper": self.model.upperPositionLimit[self.group_to_identify],
            "v_limit": self.model.velocityLimit[self.group_to_identify],
            "effort_limit": self.model.effortLimit[self.group_to_identify],
        }
        self.q_upper_limit, self.q_lower_limit, self.v_limit, self.tau_limit = (
            [],
            [],
            [],
            [],
        )
        for idx in self.group_to_identify:
            self.q_upper_limit.append(self.model.upperPositionLimit[idx])
            self.q_lower_limit.append(self.model.lowerPositionLimit[idx])
            self.v_limit.append(self.model.velocityLimit[idx])
            self.tau_limit.append(self.model.effortLimit[idx])

        np.random.seed(int(RNG_SEED))

        self.ct = CollisionTest(
            model=self.model, urdf_path=URDF_PATH, pkg_dir=PKG_DIR, performance=True
        )
        self.ct.add_collision_pairs(COLLISION_PAIRS)

    @staticmethod
    def _fmt_array(arr: np.ndarray) -> str:
        arr = np.asarray(arr, dtype=float).reshape(-1)
        return "[" + ", ".join(f"{x:.6g}" for x in arr) + "]"

    @staticmethod
    def _fmt_array_lines(arr: np.ndarray, per_line: int = 10) -> str:
        arr = np.asarray(arr, dtype=float).reshape(-1)
        if arr.size == 0:
            return "[]"
        lines = []
        for i in range(0, arr.size, per_line):
            chunk = ", ".join(f"{x:.6g}" for x in arr[i : i + per_line])
            lines.append("[" + chunk + "]")
        return "\n".join(lines)

    @staticmethod
    def _model_from_urdf(urdf_path: Path) -> pin.Model:
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
            elif name == "LAY_DOWN":
                continue  # LAY_DOWN is used only to tune the pose when identify

            dyn_elem = joint_elem.find("dynamics")
            damping = 0.0
            friction = 0.0
            if dyn_elem is not None:
                damping = float(dyn_elem.attrib.get("damping", "0.0"))
                friction = float(dyn_elem.attrib.get("friction", "0.0"))

            dynamics_by_joint[name] = {
                "damping": damping,
                "friction": friction,
            }
        return dynamics_by_joint

    def collect_target_limb_info(self):
        target_q_set = set(self.group_to_identify)
        all_infos = []
        target_infos = []

        for joint_id in range(1, self.model.njoints):
            joint = self.model.joints[joint_id]

            if joint.nq != 1 or joint.nv != 1:
                raise ValueError(
                    f"Only 1-DoF joints are supported. Joint {joint_id}: '{self.model.names[joint_id]}' has nq={joint.nq}, nv={joint.nv}."
                )
            if joint_id - 1 != joint.idx_q or joint_id - 1 != joint.idx_v:
                raise ValueError(
                    f"Expected joint {joint_id} to have idx_q and idx_v equal to joint_id-1. "
                    f"Got idx_q={joint.idx_q}, idx_v={joint.idx_v}."
                )

            all_infos.append(
                {
                    "joint_id": joint_id - 1,  # skip LAY_DOWN
                    "name": self.model.names[joint_id],
                    "idx_q": joint.idx_q,
                    "nq": joint.nq,
                    "idx_v": joint.idx_v,
                    "nv": joint.nv,
                    "q_lower": self.model.lowerPositionLimit[
                        joint.idx_q : joint.idx_q + joint.nq
                    ].copy(),
                    "q_upper": self.model.upperPositionLimit[
                        joint.idx_q : joint.idx_q + joint.nq
                    ].copy(),
                    "v_limit": self.model.velocityLimit[
                        joint.idx_v : joint.idx_v + joint.nv
                    ].copy(),
                    "effort_limit": self.model.effortLimit[
                        joint.idx_v : joint.idx_v + joint.nv
                    ].copy(),
                    "damping": self.urdf_dynamics.get(
                        self.model.names[joint_id], {}
                    ).get("damping", 0.0),
                    "friction": self.urdf_dynamics.get(
                        self.model.names[joint_id], {}
                    ).get("friction", 0.0),
                }
            )
            if target_q_set.intersection([joint.idx_q]):
                target_infos.append(
                    {
                        "joint_id": joint_id - 1,  # skip LAY_DOWN
                        "name": self.model.names[joint_id],
                        "idx_q": joint.idx_q,
                        "nq": joint.nq,
                        "idx_v": joint.idx_v,
                        "nv": joint.nv,
                        "q_lower": self.model.lowerPositionLimit[
                            joint.idx_q : joint.idx_q + joint.nq
                        ].copy(),
                        "q_upper": self.model.upperPositionLimit[
                            joint.idx_q : joint.idx_q + joint.nq
                        ].copy(),
                        "v_limit": self.model.velocityLimit[
                            joint.idx_v : joint.idx_v + joint.nv
                        ].copy(),
                        "effort_limit": self.model.effortLimit[
                            joint.idx_v : joint.idx_v + joint.nv
                        ].copy(),
                        "damping": self.urdf_dynamics.get(
                            self.model.names[joint_id], {}
                        ).get("damping", 0.0),
                        "friction": self.urdf_dynamics.get(
                            self.model.names[joint_id], {}
                        ).get("friction", 0.0),
                    }
                )

        return all_infos, target_infos

    def state_size_check_and_form(
        self, q: list | np.ndarray, v: list | np.ndarray, a: list | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        for i, v_idx in enumerate(self.group_to_identify):
            formed_v[v_idx] = v[i]
            formed_a[v_idx] = a[i]
        return formed_q, formed_v, formed_a

    def sample_state(self, target_v_indices):
        q = pin.neutral(self.model)
        for q_idx in self.group_to_identify:
            low = self.model.lowerPositionLimit[q_idx]
            high = self.model.upperPositionLimit[q_idx]
            q[q_idx] = np.random.uniform(low, high)

        v = np.zeros(self.model.nv)
        a = np.zeros(self.model.nv)

        if target_v_indices:
            v[target_v_indices] = np.random.normal(0.0, 0.3, size=len(target_v_indices))
            a[target_v_indices] = np.random.normal(0.0, 0.5, size=len(target_v_indices))

        return q, v, a

    def get_subtree_mask(self) -> np.ndarray:
        """
        Return a boolean matrix of shape (dof, dof) where mask[d, j] = True
        if joint group_to_identify[j] is in the kinematic subtree of
        group_to_identify[d].

        For a serial chain (no branching), this is simply:
            mask[d, j] = True  iff  j >= d
        because each joint's torque only depends on parameters of its
        descendants in the chain.

        For branching trees, we use the parent array to walk up from each
        potential descendant.
        """
        dof = self.dof
        mask = np.zeros((dof, dof), dtype=bool)

        for d in range(dof):
            joint_d = self.group_to_identify[d]
            for j in range(dof):
                joint_j = self.group_to_identify[j]
                # Walk up from joint_j to see if joint_d is an ancestor.
                cur = joint_j
                while cur != 0:  # 0 = universe (root)
                    if cur == joint_d:
                        mask[d, j] = True
                        break
                    cur = self.model.parents[cur]
                # Also true if joint_j == joint_d (a joint is in its own subtree).
                if joint_j == joint_d:
                    mask[d, j] = True

        return mask

    def build_augmented_target_regressor(
        self,
        Y_target_limb,
        v_target_limb,
    ):
        """
        Build target-limb regressor with 12 columns per 1-DoF joint:
        - 10 inertial columns from Pinocchio's per-joint inertial block
        - 2 friction columns [v_i, sign(v_i)]
        """

        # Remove unrelated trunk columns by keeping only target-joint inertial blocks.
        inertial_blocks = []
        Y_target_friction = np.zeros(
            (len(self.group_to_identify), 2 * len(self.group_to_identify))
        )
        friction_params_from_urdf = []

        for idx, joint in enumerate(self.group_to_identify):
            col_begin = 10 * joint
            col_end = col_begin + 10
            inertial_blocks.append(Y_target_limb[:, col_begin:col_end])
            Y_target_friction[idx, 2 * idx] = v_target_limb[idx]
            Y_target_friction[idx, 2 * idx + 1] = np.tanh(v_target_limb[idx] * 1e2)
            friction_params_from_urdf.extend(
                [
                    self.target_joint_infos[idx]["damping"],
                    self.target_joint_infos[idx]["friction"],
                ]
            )

        pi_inertia = np.hstack(
            [
                self.model.inertias[joint_id + 1].toDynamicParameters()
                for joint_id in self.group_to_identify
            ]
        )
        # +1 because universal joint took joint_id = 0
        pi_friction = np.hstack(friction_params_from_urdf)
        pi_aug = np.hstack([pi_inertia, pi_friction])

        Y_target_inertial = np.hstack(inertial_blocks)
        Y_aug = np.hstack([Y_target_inertial, Y_target_friction])

        tau_inertia = Y_target_inertial @ pi_inertia
        tau_friction = Y_target_friction @ pi_friction
        tau_aug = tau_inertia + tau_friction

        return (
            Y_aug,
            Y_target_inertial,
            Y_target_friction,
            tau_aug,
            tau_inertia,
            tau_friction,
            pi_inertia,
            pi_friction,
            pi_aug,
        )

    def compute_regressor(
        self,
        q: list | np.ndarray | None = None,
        v: list | np.ndarray | None = None,
        a: list | np.ndarray | None = None,
        print_info: bool = False,
    ):

        if q is None or v is None or a is None:
            print(
                "\n\033[92mNo state provided, sampling random state within limits for target limb...\033[0m"
            ) if print_info else None
            q, v, a = self.sample_state(self.group_to_identify)
        else:
            if print_info:
                print(
                    "\033[92mUsing provided state for regressor computation...\033[0m"
                )
                print(f"q: {self._fmt_array_lines(q)}")
                print(f"v: {self._fmt_array_lines(v)}")
                print(f"a: {self._fmt_array_lines(a)}")
            q, v, a = self.state_size_check_and_form(q, v, a)

        # Y satisfies tau = Y * pi, where pi is the stacked inertial parameter vector.
        Y = pin.computeJointTorqueRegressor(self.model, self.data, q, v, a)
        Y_target_limb = Y[self.group_to_identify, :]
        v_target_limb = v[self.group_to_identify]
        (
            self.Y_aug,
            self.Y_target_inertial,
            self.Y_target_friction,
            self.tau_aug,
            self.tau_inertia,
            self.tau_friction,
            self.pi_inertia,
            self.pi_friction,
            self.pi_aug,
        ) = self.build_augmented_target_regressor(
            Y_target_limb=Y_target_limb,
            v_target_limb=v_target_limb,
        )

        q_excess = 0.0
        v_excess = 0.0
        tau_excess = 0.0

        for i, joint_id in enumerate(self.group_to_identify):
            q_i = q[joint_id]
            v_i = v[joint_id]
            tau_i = self.tau_aug[i]

            q_lower = self.model.lowerPositionLimit[joint_id]
            q_upper = self.model.upperPositionLimit[joint_id]
            v_limit = self.model.velocityLimit[joint_id]
            tau_limit = self.model.effortLimit[joint_id]

            if q_i < q_lower:
                dq = q_lower - q_i
                q_excess += dq * dq
            elif q_i > q_upper:
                dq = q_i - q_upper
                q_excess += dq * dq
            if q_upper > q_lower:
                q_excess_normalized = q_excess / (q_upper - q_lower)
            else:
                raise ValueError(
                    f"Invalid position limits for joint {joint_id}: q_lower={q_lower}, q_upper={q_upper}"
                )

            av = v_i if v_i >= 0.0 else -v_i
            dv = av - v_limit
            if dv > 0.0:
                v_excess += dv * dv
            if v_limit > 0:
                v_excess_normalized = v_excess / v_limit
            else:
                raise ValueError(
                    f"Invalid velocity limit for joint {joint_id}: v_limit={v_limit}"
                )

            at = tau_i if tau_i >= 0.0 else -tau_i
            dt = at - tau_limit
            if dt > 0.0:
                tau_excess += dt
            if tau_limit > 0:
                tau_excess_normalized = tau_excess / tau_limit
            else:
                raise ValueError(
                    f"Invalid torque limit for joint {joint_id}: tau_limit={tau_limit}"
                )

        self.q_excess = q_excess
        self.v_excess = v_excess
        self.tau_excess = tau_excess
        self.q_excess_normalized = q_excess_normalized
        self.v_excess_normalized = v_excess_normalized
        self.tau_excess_normalized = tau_excess_normalized

        self.collided = self.ct.check_collisions(q)

        return (
            self.Y_aug,
            self.tau_aug,
            self.pi_aug,
            self.pi_inertia,
            self.pi_friction,
            self.q_excess,
            self.v_excess,
            self.tau_excess,
            self.q_excess_normalized,
            self.v_excess_normalized,
            self.tau_excess_normalized,
            self.collided,
        )

    def print_joint_info(self, selected_group=True):
        print(
            "\n"
            + (
                "\033[92mTarget limb joint parameters\033[0m"
                if selected_group
                else "\033[92mAll limb joint parameters\033[0m"
            ).center(60, "=")
        )
        printed = self.group_to_identify if selected_group else "All joints"
        print(f"Printing joint info for: {printed}\n")
        joint_infos = (
            self.target_joint_infos if selected_group else self.all_joint_infos
        )
        for info in joint_infos:
            print(
                f"joint_id = {info['joint_id']:<2d} name = {info['name']} \n"
                f"  idx_q = {info['idx_q']}; idx_v = {info['idx_v']}"
            )
            print(f"  q_lower      = {self._fmt_array_lines(info['q_lower'])}")
            print(f"  q_upper      = {self._fmt_array_lines(info['q_upper'])}")
            print(f"  velocity_lim = {self._fmt_array_lines(info['v_limit'])}")
            print(f"  effort_lim   = {self._fmt_array_lines(info['effort_limit'])}")
            print(f"  damping      = {info['damping']:.6g}")
            print(f"  friction     = {info['friction']:.6g}")

    def print_regressor_info(
        self,
        aug=True,
        parameters=False,
        inertial=False,
        friction=False,
        computed_torques=False,
        excess=False,
    ):

        print(
            "\n"
            + "\033[92mRegressor and state info for target limb\033[0m".center(80, "=")
        )

        print(f"Selected group to identify: {self.group_to_identify}")
        if self.collided:
            print("\033[91mCollision detected for the given state!\033[0m")
        else:
            print("\033[92mNo collision detected for the given state.\033[0m")

        print("\033[94mRegressor Infos\033[0m".center(80, "="))

        if aug:
            print(
                "\033[93mAugmented regressor (inertia + friction)\033[0m".center(
                    80, "-"
                )
            )
            print(f"Shape: {self.Y_aug.shape}")
            for i in range(self.Y_aug.shape[0]):
                print(
                    f"Joint {self.target_joint_infos[i]['joint_id']} ({self.target_joint_infos[i]['name']}): \n"
                    f"{self._fmt_array_lines(self.Y_aug[i, :], per_line=10)} \n"
                )

        if parameters:
            print(
                "\033[93mOriginal inertial parameters from URDF\033[0m".center(80, "-")
            )
            for i in range(self.Y_target_inertial.shape[0]):
                print(
                    f"Joint {self.target_joint_infos[i]['joint_id']} ({self.target_joint_infos[i]['name']}): \n"
                    f"{self._fmt_array_lines(self.pi_aug[i * 12 : (i + 1) * 12], per_line=12)} \n"
                )

        if inertial:
            print("\033[93mInertia-only regressor\033[0m".center(80, "-"))
            print(f"Shape: {self.Y_target_inertial.shape}")
            for i in range(self.Y_target_inertial.shape[0]):
                print(
                    f"Joint {self.target_joint_infos[i]['joint_id']} ({self.target_joint_infos[i]['name']}): \n"
                    f"{self._fmt_array_lines(self.Y_target_inertial[i, :], per_line=10)} \n"
                )

        if friction:
            print("\033[93mFriction-only regressor\033[0m".center(80, "-"))
            print(f"Shape: {self.Y_target_friction.shape}")
            for i in range(self.Y_target_friction.shape[0]):
                print(
                    f"Joint {self.target_joint_infos[i]['joint_id']} ({self.target_joint_infos[i]['name']}): \n"
                    f"{self._fmt_array_lines(self.Y_target_friction[i, :], per_line=10)} \n"
                )

        if computed_torques:
            print("\033[95mComputed torques for target limb\033[0m".center(80, "*"))
            print(f"Torques:{self._fmt_array(self.tau_aug)}")
        for i in range(len(self.group_to_identify)):
            print(
                f"Joint {self.target_joint_infos[i]['joint_id']} ({self.target_joint_infos[i]['name']}): \n"
                f"  tau_inertia  = {self.tau_aug[i] - self.tau_friction[i]:.6g} \n"
                f"  tau_friction = {self.tau_friction[i]:.6g} \n"
                f"  tau_total    = {self.tau_aug[i]:.6g}"
            )

        if excess:
            print(
                "\n"
                + "\033[92mExcess state/torque beyond limits\033[0m".center(80, "*")
            )
            print(f"q excess: {self.q_excess:.2g}")
            print(f"v excess: {self.v_excess:.2g}")
            print(f"tau excess: {self.tau_excess:.2g}")


"""
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
"""


def main():
    regressor = TargetLimbRegressor(
        urdf_path=URDF_PATH, group_to_identify="left_arm", print_info=True
    )

    (
        Y_aug,
        tau_aug,
        pi_aug,
        pi_inertia,
        pi_friction,
        q_excess,
        v_excess,
        tau_excess,
        q_excess_normalized,
        v_excess_normalized,
        tau_excess_normalized,
        collided,
    ) = regressor.compute_regressor(
        q=[-1.6, 1.5, 0.0, 0.0, 0.0],
        v=[0.5, 0.4, 0.3, 0.2, 0.1],
        a=[10.0, 8.0, 5.0, 3.0, 1.0],
        print_info=True,
    )
    regressor.print_regressor_info(computed_torques=True, parameters=True, excess=True)
    regressor.print_joint_info(selected_group=True)


if __name__ == "__main__":
    main()
