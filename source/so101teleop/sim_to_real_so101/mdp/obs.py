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
import torch
import warp as wp


import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer


def ee_frame_state(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Return the state of the end effector frame in the robot coordinate system.

    Compatibility patch:
    - IsaacLab 3.0 / Isaac Sim 6 may expose FrameTransformer pose data as Warp arrays.
    - Single-target FrameTransformer tensors may be (N, dim) instead of (N, 1, dim).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    def _to_torch(x):
        if isinstance(x, torch.Tensor):
            return x.to(device=env.device, dtype=torch.float32)
        try:
            return wp.to_torch(x).to(device=env.device, dtype=torch.float32)
        except Exception:
            return torch.as_tensor(x, device=env.device, dtype=torch.float32)

    robot_root_pos = _to_torch(robot.data.root_pos_w)
    robot_root_quat = _to_torch(robot.data.root_quat_w)

    ee_frame_pos = _to_torch(ee_frame.data.target_pos_w)
    ee_frame_quat = _to_torch(ee_frame.data.target_quat_w)

    # IsaacLab 2.x style: (num_envs, num_targets, dim)
    # IsaacLab 3.0 single-target style: (num_envs, dim)
    if ee_frame_pos.ndim == 3:
        ee_frame_pos = ee_frame_pos[:, 0, :]
    if ee_frame_quat.ndim == 3:
        ee_frame_quat = ee_frame_quat[:, 0, :]

    # Defensive handling for num_envs=1 cases that may come as flat vectors.
    if ee_frame_pos.ndim == 1:
        ee_frame_pos = ee_frame_pos.reshape(1, -1)
    if ee_frame_quat.ndim == 1:
        ee_frame_quat = ee_frame_quat.reshape(1, -1)
    if robot_root_pos.ndim == 1:
        robot_root_pos = robot_root_pos.reshape(1, -1)
    if robot_root_quat.ndim == 1:
        robot_root_quat = robot_root_quat.reshape(1, -1)

    ee_frame_pos_robot, ee_frame_quat_robot = math_utils.subtract_frame_transforms(
        robot_root_pos,
        robot_root_quat,
        ee_frame_pos,
        ee_frame_quat,
    )

    return torch.cat([ee_frame_pos_robot, ee_frame_quat_robot], dim=1)

def image_raw(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
) -> torch.Tensor:

    sensor = env.scene[sensor_cfg.name]
    images = sensor.data.output[data_type]

    return images.clone()