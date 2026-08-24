# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import torch
import warp as wp
import glob
import os
import yaml

from pxr import Gf, Sdf, UsdGeom

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.sim import get_current_stage
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation, RigidObject




# Robot color palette based on WowRobo and Seeed Studio offerings
ROBOT_COLORS = {
    "orange": (0.876, 0.317, 0.132),
    "teal": (0.0, 0.8, 0.502),
    "white": (0.95, 0.95, 0.95),
    "black": (0.08, 0.08, 0.08),
}



def _to_torch(x, env):
    """Convert IsaacLab 3.0 / Warp-backed arrays to torch tensors."""
    if isinstance(x, torch.Tensor):
        return x.to(device=env.device, dtype=torch.float32)
    try:
        return wp.to_torch(x).to(device=env.device, dtype=torch.float32)
    except Exception:
        return torch.as_tensor(x, device=env.device, dtype=torch.float32)


def _env_ids_long(env_ids, device):
    """Use torch.long indices for indexing torch tensors, while keeping original env_ids for IsaacLab write calls."""
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=device, dtype=torch.long)


def _matching_prim_paths(pattern: str):
    return sim_utils.find_matching_prims(pattern.replace("{ENV_REGEX_NS}", "/World/envs/env_.*"))


def _set_local_orientations(prim_path_pattern: str, orientations: torch.Tensor):
    for index, prim in enumerate(_matching_prim_paths(prim_path_pattern)):
        if not prim.IsValid():
            continue
        quat_xyzw = orientations[min(index, orientations.shape[0] - 1)]
        sim_utils.standardize_xform_ops(
            prim,
            orientation=(
                float(quat_xyzw[0].item()),
                float(quat_xyzw[1].item()),
                float(quat_xyzw[2].item()),
                float(quat_xyzw[3].item()),
            ),
        )


def _get_local_poses(prim_path_pattern: str, device: str):
    positions = []
    orientations = []
    for prim in _matching_prim_paths(prim_path_pattern):
        if not prim.IsValid() or not prim.IsA(UsdGeom.Xformable):
            continue
        transform = Gf.Transform(UsdGeom.Xformable(prim).GetLocalTransformation())
        translation = transform.GetTranslation()
        quat = transform.GetRotation().GetQuat()
        imag = quat.GetImaginary()
        positions.append((translation[0], translation[1], translation[2]))
        orientations.append((imag[0], imag[1], imag[2], quat.GetReal()))
    return (
        torch.tensor(positions, device=device, dtype=torch.float32),
        torch.tensor(orientations, device=device, dtype=torch.float32),
    )


def randomize_robot_color(env, 
    env_ids: torch.Tensor | None,
    color_names: list[str] = list(ROBOT_COLORS.keys()),
):
    """Randomly set robot color from predefined palette on each reset."""
    # color_names = list(ROBOT_COLORS.keys())
    idx = torch.randint(0, len(color_names), (1,), device="cpu").item()
    selected_color = ROBOT_COLORS[color_names[idx]]
    
    with Sdf.ChangeBlock():
        robot = env.scene["robot"]
        material_prim_path = robot.cfg.prim_path + "/Looks/material_a_3d_printed/Shader"
        material_prim = sim_utils.find_matching_prims(material_prim_path)[0]
        material_prim.GetAttribute("inputs:diffuse_color_constant").Set(selected_color)

def randomize_mat_rotation(
    env,
    env_ids: torch.Tensor | None,
    yaw_range: tuple[float, float] = (-0.3, 0.3),
    asset_cfg: SceneEntityCfg = None,
):
    """Randomize mat yaw rotation on reset.
    
    Args:
        yaw_range: Range of yaw rotation in radians (default ±0.3 rad ≈ ±17°).
    """
    asset = env.scene[asset_cfg.name]
    # Isaac Sim 6 / IsaacLab 3 compatibility:
    # Some USD scene entities are registered but resolve to no concrete prim paths.
    # For smoke tests, skip non-critical visual randomizers instead of crashing.
    if len(asset.prim_paths) == 0:
        return
    asset_prim_path = asset.prim_paths[0]
    
    # Sample random yaw for each environment
    yaw = math_utils.sample_uniform(
        yaw_range[0], yaw_range[1], (len(env_ids),), device=env.unwrapped.device
    )
    # Keep roll and pitch at 0, add 90° base yaw (mat's default orientation)
    roll = torch.zeros_like(yaw)
    pitch = torch.zeros_like(yaw)
    base_yaw = torch.full_like(yaw, math.pi / 2)  # 90° in radians
    
    orientations = math_utils.quat_from_euler_xyz(roll, pitch, base_yaw + yaw)
    
    _set_local_orientations(asset_prim_path, orientations)


def randomize_camera_focal_length(
    env,
    env_ids: torch.Tensor | None,
    focal_length_range: tuple[float, float] = (12.0, 15.0),
    asset_cfg: SceneEntityCfg = None,
):
    """Randomize camera focal length on reset (affects field of view).
    
    Args:
        focal_length_range: Range of focal lengths in mm (default 12-15mm).
            Lower = wider FOV, higher = narrower FOV.
    """
    stage = get_current_stage()
    camera = env.scene[asset_cfg.name]
    camera_prim_path = camera.cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_.*")
    
    focal_length = math_utils.sample_uniform(
        focal_length_range[0], focal_length_range[1], (1,), device="cpu"
    ).item()
    
    camera_prims = sim_utils.find_matching_prims(camera_prim_path)
    
    with Sdf.ChangeBlock():
        for prim in camera_prims:
            if prim.IsValid():
                focal_attr = prim.GetAttribute("focalLength")
                if focal_attr.IsValid():
                    focal_attr.Set(focal_length)


# Cache for storing default poses read from USD
_default_poses_cache = {}


def randomize_camera_pose(
    env,
    env_ids: torch.Tensor | None,
    prim_path_pattern: str,
    pos_range: dict[str, tuple[float, float]] = None,
    rot_range: dict[str, tuple[float, float]] = None,
):
    """Randomize camera mount position and orientation relative to its USD default.
    
    Args:
        prim_path_pattern: Prim path pattern for the camera mount xform.
        pos_range: Dict with x, y, z keys and (min, max) tuple values in meters.
        rot_range: Dict with roll, pitch, yaw keys and (min, max) tuple values in radians.
    """
    if pos_range is None:
        pos_range = {}
    if rot_range is None:
        rot_range = {}
    
    stage = get_current_stage()
    prim_path = prim_path_pattern.replace("{ENV_REGEX_NS}", "/World/envs/env_.*")
    prims = sim_utils.find_matching_prims(prim_path)
    
    if not prims:
        return
    
    # Read and cache default pose from first prim on first call
    if prim_path_pattern not in _default_poses_cache:
        prim = prims[0]
        translate_attr = prim.GetAttribute("xformOp:translate")
        orient_attr = prim.GetAttribute("xformOp:orient")
        
        default_pos = (0.0, 0.0, 0.0)
        default_quat = (1.0, 0.0, 0.0, 0.0)  # wxyz identity
        
        if translate_attr.IsValid():
            val = translate_attr.Get()
            if val is not None:
                default_pos = (val[0], val[1], val[2])
        
        if orient_attr.IsValid():
            val = orient_attr.Get()
            if val is not None:
                default_quat = (val.GetReal(), val.GetImaginary()[0], val.GetImaginary()[1], val.GetImaginary()[2])
        
        _default_poses_cache[prim_path_pattern] = {"pos": default_pos, "quat": default_quat}
    
    base_pos = _default_poses_cache[prim_path_pattern]["pos"]
    base_quat = _default_poses_cache[prim_path_pattern]["quat"]
    
    # Sample random offsets
    x = base_pos[0] + math_utils.sample_uniform(*pos_range.get("x", (0, 0)), (1,), device="cpu").item()
    y = base_pos[1] + math_utils.sample_uniform(*pos_range.get("y", (0, 0)), (1,), device="cpu").item()
    z = base_pos[2] + math_utils.sample_uniform(*pos_range.get("z", (0, 0)), (1,), device="cpu").item()
    
    # Sample rotation offsets and combine with base quaternion
    roll = math_utils.sample_uniform(*rot_range.get("roll", (0, 0)), (1,), device="cpu").item()
    pitch = math_utils.sample_uniform(*rot_range.get("pitch", (0, 0)), (1,), device="cpu").item()
    yaw = math_utils.sample_uniform(*rot_range.get("yaw", (0, 0)), (1,), device="cpu").item()
    
    delta_quat = math_utils.quat_from_euler_xyz(
        torch.tensor([roll]), torch.tensor([pitch]), torch.tensor([yaw])
    )[0]
    
    # Combine base quaternion with delta
    base_quat_tensor = torch.tensor([base_quat])
    final_quat = math_utils.quat_mul(base_quat_tensor, delta_quat.unsqueeze(0))[0]
    
    with Sdf.ChangeBlock():
        for prim in prims:
            if prim.IsValid():
                # Set translation
                translate_attr = prim.GetAttribute("xformOp:translate")
                if translate_attr.IsValid():
                    translate_attr.Set(Gf.Vec3d(x, y, z))
                # Set orientation
                orient_attr = prim.GetAttribute("xformOp:orient")
                if orient_attr.IsValid():
                    orient_attr.Set(Gf.Quatd(final_quat[0].item(), final_quat[1].item(), final_quat[2].item(), final_quat[3].item()))


def randomize_light_exposure(
    env,
    env_ids: torch.Tensor | None,
    exposure_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = None,
):

    stage = get_current_stage()
    asset = env.scene[asset_cfg.name]
    # Isaac Sim 6 / IsaacLab 3 compatibility:
    # Some USD scene entities are registered but resolve to no concrete prim paths.
    # For smoke tests, skip non-critical visual randomizers instead of crashing.
    if len(asset.prim_paths) == 0:
        return
    asset_prim_path = asset.prim_paths[0]

    exposure = math_utils.sample_uniform(*exposure_range, (1,), device="cpu").item()

    with Sdf.ChangeBlock():
        prim = stage.GetPrimAtPath(asset_prim_path)
        if prim.IsValid():
            prim.GetAttribute("inputs:exposure").Set(exposure)


def randomize_sky_light(
    env,
    env_ids: torch.Tensor | None,
    exposure_range: tuple[float, float],
    temperature_range: tuple[float, float],
    textures_root: str,
    asset_cfg: SceneEntityCfg = None,
):

    stage = get_current_stage()
    asset = env.scene[asset_cfg.name]
    # Isaac Sim 6 / IsaacLab 3 compatibility:
    # Some USD scene entities are registered but resolve to no concrete prim paths.
    # For smoke tests, skip non-critical visual randomizers instead of crashing.
    if len(asset.prim_paths) == 0:
        return
    asset_prim_path = asset.prim_paths[0]

    exposure = math_utils.sample_uniform(*exposure_range, (1,), device="cpu").item()
    temperature = math_utils.sample_uniform(*temperature_range, (1,), device="cpu").item()

    # list all .exr files in the textures_root
    textures = glob.glob(os.path.join(textures_root, "*.exr"))
    # get a random texture (using torch)

    if not textures:
        print("[WARNING] No textures found in the textures_root")
        return

    texture = torch.randint(0, len(textures), (1,), device="cpu").item()
    texture = textures[texture]

    yaw_mapping_path = os.path.join(textures_root, "yaw_mapping.yaml")
    with open(yaw_mapping_path, 'r') as f:
        yaw_mapping = yaml.safe_load(f)

    yaw_mapping = yaw_mapping.get(os.path.basename(texture), None)

    range_list = [
        (0.0, 0.0),
        (0.0, 0.0),
        (
            yaw_mapping[0] if yaw_mapping else 0.0,
            yaw_mapping[1] if yaw_mapping else 0.0,
        ),
    ]
    ranges = torch.tensor(range_list, device=env.unwrapped.device)
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=env.unwrapped.device
    )
    orientations = math_utils.quat_from_euler_xyz(
        rand_samples[:, 0], rand_samples[:, 1], rand_samples[:, 2]
    )

    _set_local_orientations(asset_prim_path, orientations)

    with Sdf.ChangeBlock():
        prim = stage.GetPrimAtPath(asset_prim_path)
        if prim.IsValid():
            prim.GetAttribute("inputs:exposure").Set(exposure)
            prim.GetAttribute("inputs:colorTemperature").Set(temperature)
            texture = Sdf.AssetPath(texture)
            prim.GetAttribute("inputs:texture:file").Set(texture)


def random_asset_pose(
    env,
    env_ids,
    asset,
    pose_range,
    pos_offset,
):
    # IsaacLab 3.0 compatibility:
    # asset.data.default_root_state may be Warp-backed, so convert before indexing.
    env_ids_index = _env_ids_long(env_ids, asset.device)

    root_states_all = _to_torch(asset.data.default_root_state, env)
    root_states = root_states_all[env_ids_index].clone()

    pos_offset_list = [pos_offset.get(key, 0.0) for key in ["x", "y", "z"]]
    pos_offset = torch.tensor(pos_offset_list, device=asset.device, dtype=torch.float32)

    range_list = [
        pose_range.get(key, (0.0, 0.0))
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=asset.device, dtype=torch.float32)

    rand_samples = math_utils.sample_uniform(
        ranges[:, 0],
        ranges[:, 1],
        (len(env_ids_index), 6),
        device=asset.device,
    )

    env_origins = env.scene.env_origins[env_ids_index].to(device=asset.device, dtype=torch.float32)

    positions = (
        root_states[:, 0:3]
        + env_origins
        + rand_samples[:, 0:3]
        + pos_offset
    )

    orientations_delta = math_utils.quat_from_euler_xyz(
        rand_samples[:, 3],
        rand_samples[:, 4],
        rand_samples[:, 5],
    )
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    asset.write_root_pose_to_sim(
        torch.cat([positions, orientations], dim=-1),
        env_ids=env_ids,
    )

    return positions, orientations


def reset_cube_position(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    x_range: tuple[float, float] = (0.16, 0.28),
    y_range: tuple[float, float] = (-0.12, 0.04),
    z: float = 0.07,
):
    """Randomize the cube start pose on the table."""
    asset: RigidObject = env.scene[asset_cfg.name]
    env_ids_index = _env_ids_long(env_ids, asset.device)
    positions = _to_torch(asset.data.default_root_state, env)[env_ids_index, 0:3].clone()
    positions[:, 0] = math_utils.sample_uniform(x_range[0], x_range[1], (len(env_ids_index),), device=asset.device)
    positions[:, 1] = math_utils.sample_uniform(y_range[0], y_range[1], (len(env_ids_index),), device=asset.device)
    positions[:, 2] = z
    positions += env.scene.env_origins[env_ids_index].to(device=asset.device, dtype=torch.float32)
    orientations = torch.zeros((len(env_ids_index), 4), device=asset.device, dtype=torch.float32)
    orientations[:, 3] = 1.0
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros((len(env_ids_index), 6), device=asset.device), env_ids=env_ids)

def reset_vials_rack(
        env,
        env_ids: torch.Tensor,
        vials: list[str],
        rack: str,
        rack_pose_range: dict[str, tuple[float, float]],
        pose_range: dict[str, tuple[float, float]],
        fixed_vial_z: float,
        rack_placement_prob: float = 0.33,
):

    vial_objects: list[RigidObject | Articulation] = [
        env.scene[asset_name] for asset_name in vials
    ]

    rack = env.scene[rack]
    rack_prim_path = rack.prim_paths[0] if len(rack.prim_paths) > 0 else rack.cfg.prim_path
    slot_positions_local, slot_orientations_local = _get_local_poses(
        f"{rack_prim_path}/Body1/Mesh/top_.*", rack.device
    )
    total_slots = slot_positions_local.shape[0]

    # randomize rack pose
    new_rack_positions, new_rack_orientations = random_asset_pose(env, env_ids, rack, rack_pose_range, {})
    # Clear velocities immediately after positioning
    zero_velocity = torch.zeros((len(env_ids), 6), device=rack.device)
    rack.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)

    placed_on_rack_indices = []
    if total_slots > 0 and torch.rand(1, device=env.unwrapped.device).item() < rack_placement_prob:
        # Pick a random vial to place on the rack
        vial_idx = torch.randint(0, len(vial_objects), (1,), device=env.unwrapped.device).item()
        placed_on_rack_indices.append(vial_idx)
    
    # Place selected vials on rack
    for vial_idx in placed_on_rack_indices:
        vial = vial_objects[vial_idx]
        
        # Select a random slot for this vial
        slot_idx = torch.randint(0, total_slots, (1,), device=env.unwrapped.device).item()
        
        # Thank you Sonnet lol
        # IMPORTANT: Cannot use get_world_poses() here because write_root_pose_to_sim() doesn't
        # update USD until sim.step(). Instead, manually compute slot positions from local transforms.
        # Transform slot positions from rack local frame to world frame using the NEW rack pose
        # For each environment, we need to transform the slot position
        slot_position_local = slot_positions_local[slot_idx].unsqueeze(0).repeat(len(env_ids), 1)
        slot_orientation_local = slot_orientations_local[slot_idx].unsqueeze(0).repeat(len(env_ids), 1)
        
        # Combine transforms: world_pose = rack_pose ⊕ slot_local_pose
        slot_position, slot_orientation = math_utils.combine_frame_transforms(
            new_rack_positions, new_rack_orientations, slot_position_local, slot_orientation_local
        )
        # slot_position and slot_orientation are already per-env tensors (shape: len(env_ids), 3/4)
        slot_pose = torch.cat([slot_position, slot_orientation], dim=-1)
        vial.write_root_pose_to_sim(slot_pose, env_ids=env_ids)

        zero_velocity = torch.zeros((len(env_ids), 6), device=vial.device)
        vial.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)

    pose_range_z_fixed = {**pose_range, "z": (0.0, 0.0)}
    for i, v in enumerate(vial_objects):
        if i not in placed_on_rack_indices:
            env_ids_index = _env_ids_long(env_ids, v.device)
            default_root_state = _to_torch(v.data.default_root_state, env)
            default_z = default_root_state[env_ids_index[0], 2].item()
            pos_offset = {"z": fixed_vial_z - default_z}
            _, _ = random_asset_pose(env, env_ids, v, pose_range_z_fixed, pos_offset)
            zero_velocity = torch.zeros((len(env_ids), 6), device=v.device)
            v.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)
