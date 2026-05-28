from pathlib import Path

import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

from ament_index_python.packages import get_package_share_directory

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
    def _geometry_id(self, geometry_name):
        geom_id = self.collision_model.getGeometryId(geometry_name)
        if geom_id < 0 or geom_id >= len(self.collision_model.geometryObjects):
            raise ValueError(f"Geometry '{geometry_name}' was not found in collision model")
        return geom_id

    def _add_collision_pairs_by_name(self, collision_pairs):
        for first_name, second_name in collision_pairs:
            self.collision_model.addCollisionPair(
                pin.CollisionPair(
                    self._geometry_id(first_name),
                    self._geometry_id(second_name),
                )
            )

    def _pair_geometry_names(self, pair_index):
        pair = self.collision_model.collisionPairs[pair_index]
        first_name = self.collision_model.geometryObjects[pair.first].name
        second_name = self.collision_model.geometryObjects[pair.second].name
        return first_name, second_name

    def __init__(
            self, 
            urdf_path=URDF_PATH,
            pkg_dir=PKG_DIR,
            ):
        
        print(f"\n\033[92mLoading URDF from: {urdf_path}\033\n[0m")
        
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        
        self.collision_model = pin.buildGeomFromUrdf(
            self.model,
            str(urdf_path),
            pin.GeometryType.COLLISION,
            package_dirs=[str(pkg_dir)],
        )

        self.visual_model = pin.buildGeomFromUrdf(
            self.model,
            str(urdf_path),
            pin.GeometryType.VISUAL,
            package_dirs=[str(pkg_dir)],
        )
        self.visual_data = pin.GeometryData(self.visual_model)
        
        for i, joint in enumerate(self.model.joints):
            print(f"  {i}: {self.model.names[i]}")
        for i, geom in enumerate(self.collision_model.geometryObjects):
            print(f"  {i}: {geom.name}")

        print(self.data.oMf[self.model.getFrameId("LINK_ELBOW_END_L")])

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
        self._add_collision_pairs_by_name(collision_pairs)

        self.collision_data = pin.GeometryData(self.collision_model)

        q = np.zeros(self.model.nq)
        q[self.model.joints[self.model.getJointId("J14_SHOULDER_ROLL_L")].idx_q] = 0.5
        q[self.model.joints[self.model.getJointId("J15_SHOULDER_YAW_L")].idx_q] = -1.2
        q[self.model.joints[self.model.getJointId("J16_ELBOW_PITCH_L")].idx_q] = -1

        print(q)

        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

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
        
        # Compute collisions per pair (safer per-pair API)
        for k in range(len(self.collision_model.collisionPairs)):
            try:
                pin.computeCollision(self.collision_model, self.collision_data, k)
                result = self.collision_data.collisionResults[k]
                if result.isCollision():
                    first_name, second_name = self._pair_geometry_names(k)
                    print(f"Collision detected for pair {k}: {first_name} <-> {second_name}")
            except Exception as e:
                first_name, second_name = self._pair_geometry_names(k)
                print(
                    "Error occurred while computing collision for pair "
                    f"{k} ({first_name} <-> {second_name}): {e}"
                )

        viz = MeshcatVisualizer(self.model, self.collision_model, self.visual_model) 
        # 2. Initialize viewer (creates window/connection)
        viz.initViewer()        
        # 3. Load geometry into viewer
        viz.loadViewerModel()        
        # 4. Display at configuration q
        while True:
            viz.display(q)

if __name__ == "__main__":
    collision_test = CollisionTest()