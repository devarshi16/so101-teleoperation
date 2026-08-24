# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SO-101 surface-cloth teleoperation tasks with one fixed world RGB camera."""

from __future__ import annotations

import torch
from pxr import PhysxSchema, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim.spawners.from_files import spawn_from_usd
from isaaclab.sim.spawners.meshes import spawn_mesh_rectangle
from isaaclab.utils.configclass import configclass
from isaaclab_physx.sim.schemas import PhysxDeformableBodyPropertiesCfg
from isaaclab_physx.sim.spawners.materials import PhysxSurfaceDeformableBodyMaterialCfg

from sim_to_real_so101.assets.so101 import S0101_NO_CAMERA_CFG
from sim_to_real_so101.mdp import (
    ClothGraspJointPositionActionCfg,
    image,
    joint_pos,
    reset_joints_by_offset,
    reset_nodal_state_uniform,
)
from sim_to_real_so101.utils.rotations import euler_angles_to_quat

from .so101_env_cfg import ActionsCfg, LerobotSo101BaseSceneCfg, SO101TeleopEnvCfg


TASK_SPECS = {
    "corner_lift": {
        "gym_id": "SO101-Cloth-Corner-Lift-v0",
        "instruction": "Grasp the orange near-left corner and lift it at least 6 cm above the table.",
        "timeout_s": 10.0,
        "lift_height_m": 0.06,
    },
    "edge_drag": {
        "gym_id": "SO101-Cloth-Edge-Drag-v0",
        "instruction": "Grasp the marked near edge and drag it onto the green target without a high lift.",
        "timeout_s": 12.0,
        "target_xy": (0.27, -0.14),
    },
    "corner_fold": {
        "gym_id": "SO101-Cloth-Corner-Fold-v0",
        "instruction": "Place the orange corner on the opposite green target to make a diagonal fold.",
        "timeout_s": 15.0,
        "target_xy": (0.28, 0.08),
    },
    "obstacle_drape_pull": {
        "gym_id": "SO101-Cloth-Obstacle-Drape-Pull-v0",
        "instruction": "Pull the marked edge across the blue cylinder to the green target side.",
        "timeout_s": 18.0,
        "target_xy": (0.28, -0.14),
    },
}


def spawn_so101_with_tight_hand_contacts(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation=None,
    orientation=None,
    **kwargs,
):
    """Spawn SO-101 and author contact margins on its instanced hand colliders."""
    prim = spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    stage = prim.GetStage()
    hand_material_path = f"{prim_path}/hand_physics_material"
    hand_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=1.5,
        dynamic_friction=1.2,
        restitution=0.0,
        friction_combine_mode="max",
    )
    hand_material.func(hand_material_path, hand_material)
    hand_usd_material = UsdShade.Material(stage.GetPrimAtPath(hand_material_path))
    for link_name in ("wrist", "gripper", "jaw"):
        collision_prim = stage.GetPrimAtPath(f"{prim_path}/{link_name}/collisions")
        if collision_prim.IsValid():
            collision_api = PhysxSchema.PhysxCollisionAPI.Apply(collision_prim)
            collision_api.GetRestOffsetAttr().Set(0.0)
            collision_api.GetContactOffsetAttr().Set(0.001)
            binding_api = UsdShade.MaterialBindingAPI.Apply(collision_prim)
            binding_api.Bind(
                hand_usd_material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                materialPurpose="physics",
            )
    return prim


def spawn_cloth_with_tight_contacts(
    prim_path: str,
    cfg: sim_utils.MeshRectangleCfg,
    translation=None,
    orientation=None,
    **kwargs,
):
    """Spawn surface cloth and set offsets on the generated simulation mesh."""
    prim = spawn_mesh_rectangle(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    sim_mesh = prim.GetStage().GetPrimAtPath(f"{prim_path}/sim_mesh")
    if not sim_mesh.IsValid():
        raise RuntimeError(f"Surface-cloth simulation mesh was not generated at {prim_path}/sim_mesh")
    collision_api = PhysxSchema.PhysxCollisionAPI.Apply(sim_mesh)
    collision_api.GetRestOffsetAttr().Set(0.0)
    collision_api.GetContactOffsetAttr().Set(0.002)
    attachment = prim.GetStage().DefinePrim(f"{prim_path}/grasp_attachment", "OmniPhysicsVtxXformAttachment")
    attachment.GetAttribute("omniphysics:attachmentEnabled").Set(False)
    attachment.GetRelationship("omniphysics:src0").SetTargets([sim_mesh.GetPath()])
    robot_gripper_path = f"{prim_path.rsplit('/Cloth', 1)[0]}/Robot/gripper"
    attachment.GetRelationship("omniphysics:src1").SetTargets([robot_gripper_path])
    attachment.GetAttribute("omniphysics:vtxIndicesSrc0").Set([])
    attachment.GetAttribute("omniphysics:localPositionsSrc1").Set([])
    return prim


def set_fixed_policy_camera(
    env,
    env_ids: torch.Tensor,
    sensor_cfg: SceneEntityCfg,
    eye_m: tuple[float, float, float],
    target_m: tuple[float, float, float],
) -> None:
    """Set the external policy camera from an eye/target pair in each environment."""
    camera = env.scene[sensor_cfg.name]
    indices = (
        torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        if env_ids is None
        else env_ids.to(device=env.device, dtype=torch.long)
    )
    origins = env.scene.env_origins[indices]
    eye = torch.tensor(eye_m, device=env.device, dtype=torch.float32).unsqueeze(0)
    target = torch.tensor(target_m, device=env.device, dtype=torch.float32).unsqueeze(0)
    camera.set_world_poses_from_view(origins + eye, origins + target, env_ids=indices)


def cloth_state(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("cloth")) -> torch.Tensor:
    """Auxiliary cloth state for QA; ACT does not consume this observation group."""
    positions = env.scene[asset_cfg.name].data.nodal_pos_w.torch - env.scene.env_origins[:, None, :]
    centroid = positions.mean(dim=1)
    minimum = positions.amin(dim=1)
    maximum = positions.amax(dim=1)
    extent = maximum - minimum
    return torch.cat((centroid, minimum[:, 2:3], maximum[:, 2:3], extent[:, :2]), dim=-1)


@configclass
class SO101ClothSceneCfg(LerobotSo101BaseSceneCfg):
    """Shared robot, tabletop, surface cloth, markers, light, and fixed camera."""

    robot: ArticulationCfg = S0101_NO_CAMERA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = (-0.12, 0.0, 0.04)
    robot.spawn.func = spawn_so101_with_tight_hand_contacts
    robot.spawn.collision_props = None
    robot.actuators["gripper"].stiffness = 7.0
    robot.actuators["gripper"].damping = 0.5

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.28, 0.0, 0.02)),
        spawn=sim_utils.CuboidCfg(
            size=(0.76, 0.72, 0.04),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.52, 0.46)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.65),
        ),
    )

    source_marker = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SourceMarker",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.16, -0.14, 0.046)),
        spawn=sim_utils.CuboidCfg(
            size=(0.032, 0.032, 0.012),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.42, 0.02), emissive_color=(0.3, 0.08, 0.0)
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.85, dynamic_friction=0.7),
        ),
    )

    cloth = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cloth",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.30, 0.0, 0.056)),
        spawn=sim_utils.MeshRectangleCfg(
            func=spawn_cloth_with_tight_contacts,
            size=(0.28, 0.28),
            resolution=(30, 30),
            deformable_props=PhysxDeformableBodyPropertiesCfg(
                mass=0.025,
                solver_position_iteration_count=16,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.78, 0.16, 0.08)),
            physics_material=PhysxSurfaceDeformableBodyMaterialCfg(
                density=63.0,
                static_friction=0.65,
                dynamic_friction=0.55,
                youngs_modulus=100_000.0,
                poissons_ratio=0.4,
                elasticity_damping=0.005,
                surface_thickness=0.002,
                bend_damping=0.0,
                surface_stretch_stiffness=0.0,
                surface_shear_stiffness=0.0,
                surface_bend_stiffness=0.0,
            ),
        ),
    )

    target_marker = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TargetMarker",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.28, 0.08, 0.043)),
        spawn=sim_utils.SphereCfg(
            radius=0.014,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.08, 0.82, 0.26), emissive_color=(0.0, 0.25, 0.04)),
        ),
    )

    camera_fixed_rgb = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/PolicyCamera",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 20.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(convention="opengl"),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/ClothTaskDomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=1800.0, color=(0.92, 0.95, 1.0)),
    )


@configclass
class SO101ClothObstacleSceneCfg(SO101ClothSceneCfg):
    cylinder = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Cylinder",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.32, 0.0, 0.085),
            rot=euler_angles_to_quat((90.0, 0.0, 0.0), degrees=True),
        ),
        spawn=sim_utils.CylinderCfg(
            radius=0.045,
            height=0.34,
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.32, 0.78)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.65, dynamic_friction=0.55),
        ),
    )


@configclass
class ClothObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos_obs = ObsTerm(func=joint_pos)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class VisualCfg(ObsGroup):
        rgb_fixed_rgb = ObsTerm(
            func=image,
            params={"sensor_cfg": SceneEntityCfg("camera_fixed_rgb"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class DiagnosticsCfg(ObsGroup):
        cloth_state = ObsTerm(func=cloth_state)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    visual: VisualCfg = VisualCfg()
    diagnostics: DiagnosticsCfg = DiagnosticsCfg()


@configclass
class ClothActionsCfg(ActionsCfg):
    joint_positions = ClothGraspJointPositionActionCfg(
        asset_name="robot",
        joint_names=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"],
        scale=1.0,
        use_default_offset=False,
    )


@configclass
class ClothEventsCfg:
    reset_robot_position = EventTerm(
        func=reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
            ),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )
    reset_cloth = EventTerm(
        func=reset_nodal_state_uniform,
        mode="reset",
        params={
            "position_range": {"x": (-0.015, 0.015), "y": (-0.015, 0.015), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cloth"),
        },
    )
    position_policy_camera = EventTerm(
        func=set_fixed_policy_camera,
        mode="startup",
        params={
            "sensor_cfg": SceneEntityCfg("camera_fixed_rgb"),
            "eye_m": (0.95, -0.95, 0.85),
            "target_m": (0.24, 0.0, 0.10),
        },
    )


@configclass
class SO101ClothEnvCfg(SO101TeleopEnvCfg):
    scene: SO101ClothSceneCfg = SO101ClothSceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=True)
    observations: ClothObservationsCfg = ClothObservationsCfg()
    actions: ClothActionsCfg = ClothActionsCfg()
    events: ClothEventsCfg = ClothEventsCfg()
    task_id: str = "corner_lift"
    instruction: str = TASK_SPECS["corner_lift"]["instruction"]

    def __post_init__(self) -> None:
        super().__post_init__()
        self.decimation = 4
        self.episode_length_s = 30.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (0.95, -0.95, 0.85)
        self.viewer.lookat = (0.24, 0.0, 0.10)
        self.sim.render.enable_translucency = False


@configclass
class SO101ClothCornerLiftEnvCfg(SO101ClothEnvCfg):
    task_id: str = "corner_lift"
    instruction: str = TASK_SPECS["corner_lift"]["instruction"]


@configclass
class SO101ClothEdgeDragEnvCfg(SO101ClothEnvCfg):
    task_id: str = "edge_drag"
    instruction: str = TASK_SPECS["edge_drag"]["instruction"]

    def __post_init__(self) -> None:
        super().__post_init__()
        target_xy = TASK_SPECS[self.task_id]["target_xy"]
        self.scene.target_marker.init_state.pos = (*target_xy, 0.043)


@configclass
class SO101ClothCornerFoldEnvCfg(SO101ClothEnvCfg):
    task_id: str = "corner_fold"
    instruction: str = TASK_SPECS["corner_fold"]["instruction"]

    def __post_init__(self) -> None:
        super().__post_init__()
        target_xy = TASK_SPECS[self.task_id]["target_xy"]
        self.scene.target_marker.init_state.pos = (*target_xy, 0.043)


@configclass
class SO101ClothObstacleDrapePullEnvCfg(SO101ClothEnvCfg):
    scene: SO101ClothObstacleSceneCfg = SO101ClothObstacleSceneCfg(
        num_envs=1, env_spacing=2.0, replicate_physics=True
    )
    task_id: str = "obstacle_drape_pull"
    instruction: str = TASK_SPECS["obstacle_drape_pull"]["instruction"]

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.cloth.init_state.pos = (0.20, 0.0, 0.16)
        self.scene.source_marker.init_state.pos = (0.06, -0.14, 0.046)
        self.scene.cylinder.init_state.pos = (0.19, 0.0, 0.085)
        target_xy = TASK_SPECS[self.task_id]["target_xy"]
        self.scene.target_marker.init_state.pos = (*target_xy, 0.043)
