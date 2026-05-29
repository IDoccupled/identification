from pathlib import Path

import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

from ament_index_python.packages import get_package_share_directory

import time

LEFT_LEG_Q_INDICES  = [0, 1, 2, 3, 4, 5]
RIGHT_LEG_Q_INDICES = [6, 7, 8, 9, 10, 11]
WAIST_Q_INDICES     = [12]
LEFT_ARM_Q_INDICES  = [13, 14, 15, 16, 17]
RIGHT_ARM_Q_INDICES = [18, 19, 20, 21, 22]
NECK_Q_INDICES      = [23]

URDF_PATH = (
    Path(get_package_share_directory('identification'))
    / "resource"
    / "robot"
    / "urdf"
    / "serial_pm_v2_identify.urdf"
).resolve()
MESH_DIR = (
    Path(get_package_share_directory('identification'))
    / "resource"
    / "robot"
    / "meshes"
).resolve()
PKG_DIR = Path(get_package_share_directory('identification')).resolve()

class CollisionTest:
    def _geometry_id(self, collision_model, geometry_name):
        geom_id = collision_model.getGeometryId(geometry_name)
        if geom_id < 0 or geom_id >= len(collision_model.geometryObjects):
            raise ValueError(f"Geometry '{geometry_name}' was not found in collision model")
        return geom_id
    def _joint_id_idx(self, model, joint_name):
        joint_id = model.joints[model.getJointId(joint_name)].idx_q
        if joint_id < 0 or joint_id >= len(model.joints):
            raise ValueError(f"Joint '{joint_name}' was not found in model")
        return joint_id

    def _add_collision_pairs_by_name(self, collision_model, collision_pairs):
        for first_name, second_name in collision_pairs:
            collision_model.addCollisionPair(
                pin.CollisionPair(
                    self._geometry_id(collision_model, first_name),
                    self._geometry_id(collision_model, second_name),
                )
            )
    def _pair_geometry_names(self, collision_model, pair_index):
        pair = collision_model.collisionPairs[pair_index]
        first_name = collision_model.geometryObjects[pair.first].name
        second_name = collision_model.geometryObjects[pair.second].name
        return first_name, second_name

    def __init__(
            self, 
            urdf_path: Path = URDF_PATH,
            pkg_dir: Path = PKG_DIR,
            model: pin.Model = None,
            visualize: bool = False
            ):
        
        if not urdf_path.is_file():
            raise FileNotFoundError(f"\033[91mURDF file not found at {urdf_path}\033[0m")
        if not pkg_dir.is_dir():
            raise NotADirectoryError(f"\033[91mPackage directory not found at {pkg_dir}\033[0m")
        
        if model:
            self.model = model
            print(f'\033[92mUsing provided model:\033[0m')
        else:
            self.model = pin.buildModelFromUrdf(str(urdf_path))
            print(f'\033[92mBuilding model from URDF at {urdf_path}:\033[0m')
        
        print(self.model)

        self.data = self.model.createData()

        print(f'\033[92mBuilding collision model...\033[0m')
        self.collision_model = pin.buildGeomFromUrdf(
            self.model,
            str(urdf_path),
            pin.GeometryType.COLLISION,
            package_dirs=[str(pkg_dir)],
        )
        print(f'\033[92mCollision model built:\033[0m')
        for geom in self.collision_model.geometryObjects:
            print(f"  - {geom.name}, parentJoint: {geom.parentJoint}, parentFrame: {geom.parentFrame}")

        if visualize:
            print(f'\033[92mBuilding visual model...\033[0m')
            self.visual_model = pin.buildGeomFromUrdf(
                self.model,
                str(urdf_path),
                pin.GeometryType.VISUAL,
                package_dirs=[str(pkg_dir)],
            )
            self.visual_data = pin.GeometryData(self.visual_model)
            print(f'\033[92mVisual model built.\033[0m')

        self.pair_added = False

    def add_collision_pairs(self, collision_pairs: list[tuple[str, str]]):

        if self.pair_added:
            raise Exception('Add collision pairs only once before any collision computation')

        self._add_collision_pairs_by_name(self.collision_model, collision_pairs)
        self.pair_added = True

        self.collision_data = pin.GeometryData(self.collision_model)

    def check_collisions(self, q: np.ndarray, visualize: bool = False):

        if not self.pair_added:
            raise Exception('Add collision pairs before checking collisions')
        
        print(f'\033[92mCollision computation at configuration:\033[0m')
        print(q)

        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        if visualize:
            pin.updateGeometryPlacements(self.model, 
                                         self.data,
                                         self.visual_model, 
                                         self.visual_data,
                                         q)
        
        pin.updateGeometryPlacements(self.model, 
                                     self.data,
                                     self.collision_model, 
                                     self.collision_data,
                                     q)
        start_time = time.time()
        
        print('\n\033[92mComputing collisions...\033[0m')
        for k in range(len(self.collision_model.collisionPairs)):
            try:
                pin.computeCollision(self.collision_model, self.collision_data, k)
                result = self.collision_data.collisionResults[k]
                if result.isCollision():
                    first_name, second_name = self._pair_geometry_names(self.collision_model, k)
                    print(f"Collision detected for pair {k}: {first_name} <-> {second_name}")
            except Exception as e:
                first_name, second_name = self._pair_geometry_names(self.collision_model, k)
                print(
                    "Error occurred while computing collision for pair "
                    f"{k} ({first_name} <-> {second_name}): {e}"
                )
        print(f"Time taken: {(time.time() - start_time) * 1000:.2f} ms")

        if visualize:
            viz = MeshcatVisualizer(self.model, self.collision_model, self.visual_model) 
            viz.initViewer()        
            viz.loadViewerModel()        
            while True:
                viz.display(q)

if __name__ == "__main__":

    model = pin.buildModelFromUrdf(str(URDF_PATH))
    
    ct = CollisionTest(
        model=model,
        urdf_path=URDF_PATH,
        pkg_dir=PKG_DIR,
        visualize=True
    )

    collision_pairs = [
        ("LINK_ELBOW_YAW_L_0", "LINK_BASE_0"),
        ("LINK_ELBOW_YAW_L_0", "LINK_TORSO_YAW_0"),
        ("LINK_ELBOW_YAW_L_0", "LINK_HEAD_YAW_0"),
        ("LINK_ELBOW_YAW_L_0", "LINK_HIP_PITCH_L_0"),
        ("LINK_ELBOW_YAW_L_0", "LINK_HIP_ROLL_L_0"),
        ("LINK_ELBOW_YAW_L_0", "LINK_HIP_YAW_L_0"),
        ("LINK_ELBOW_PITCH_L_0", "LINK_TORSO_YAW_0"),
        ("LINK_SHOULDER_YAW_L_0", "LINK_TORSO_YAW_0"),
    ]

    ct.add_collision_pairs(collision_pairs)

    q = np.zeros(model.nq)
    q[ct._joint_id_idx(ct.model, "J14_SHOULDER_ROLL_L")] = 0.5
    q[ct._joint_id_idx(ct.model, "J15_SHOULDER_YAW_L")] = -1.2
    q[ct._joint_id_idx(ct.model, "J16_ELBOW_PITCH_L")] = -1

    ct.check_collisions(q, visualize=True)