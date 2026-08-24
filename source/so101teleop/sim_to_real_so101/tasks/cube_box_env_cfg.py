# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from sim_to_real_so101.assets.so101 import SO101_CFG
from sim_to_real_so101.mdp import (
    cube_in_box,
    cube_in_box_termination,
    object_root_pose,
    reset_cube_position,
    reset_joints_by_offset,
)

from .so101_env_cfg import ActionsCfg, LerobotSo101BaseSceneCfg, ObservationsCfg, SO101TeleopEnvCfg

TABLE_TOP_Z = 0.04
CUBE_Z = 0.025
BOX_POS = (0.24, 0.16, 0.0)
BOX_SUCCESS_OFFSET = (0.0, 0.0, 0.055)
BOX_HALF_EXTENTS = (0.07, 0.07, 0.07)


@configclass
class SO101CubeBoxSceneCfg(LerobotSo101BaseSceneCfg):
    """SO101 pick-place scene with a table, one cube, and one open box."""

    robot: ArticulationCfg = SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = (-0.12, 0.0, TABLE_TOP_Z)

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.15, 0.0, 0.0), rot=(0.0, 0.0, 0.707, 0.707)),
    )

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.22, -0.08, CUBE_Z), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.035),
        ),
    )

    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        init_state=RigidObjectCfg.InitialStateCfg(pos=BOX_POS, rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Objects/Box/box.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )


@configclass
class CubeBoxObservationsCfg(ObservationsCfg):
    @configclass
    class ObjectCfg(ObsGroup):
        cube_pose = ObsTerm(func=object_root_pose, params={"asset_cfg": SceneEntityCfg("cube")})
        cube_in_box = ObsTerm(
            func=cube_in_box,
            params={
                "cube_cfg": SceneEntityCfg("cube"),
                "box_cfg": SceneEntityCfg("box"),
                "center_offset": BOX_SUCCESS_OFFSET,
                "half_extents": BOX_HALF_EXTENTS,
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    object: ObjectCfg = ObjectCfg()


@configclass
class CubeBoxEventCfg:
    reset_robot_position = EventTerm(
        func=reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"],
            ),
            "position_range": (0, 0),
            "velocity_range": (0, 0),
        },
    )
    reset_cube_position = EventTerm(
        func=reset_cube_position,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube"),
            "x_range": (0.14, 0.27),
            "y_range": (-0.14, 0.02),
            "z": CUBE_Z,
        },
    )


@configclass
class CubeBoxTerminationsCfg:
    success = DoneTerm(
        func=cube_in_box_termination,
        time_out=False,
        params={
            "cube_cfg": SceneEntityCfg("cube"),
            "box_cfg": SceneEntityCfg("box"),
            "center_offset": BOX_SUCCESS_OFFSET,
            "half_extents": BOX_HALF_EXTENTS,
            "warmup_steps": 30,
        },
    )


@configclass
class SO101CubeBoxTeleopEnvCfg(SO101TeleopEnvCfg):
    scene: SO101CubeBoxSceneCfg = SO101CubeBoxSceneCfg()
    observations: CubeBoxObservationsCfg = CubeBoxObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: CubeBoxEventCfg = CubeBoxEventCfg()
    terminations: CubeBoxTerminationsCfg = CubeBoxTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 60
        self.viewer.eye = (-0.12, -0.55, 0.42)
        self.viewer.lookat = (0.16, 0.05, 0.08)
