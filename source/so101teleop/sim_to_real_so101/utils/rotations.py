# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rotation helpers for Isaac Lab 3 / Isaac Sim 6 compatibility."""

from __future__ import annotations

import numpy as np
import torch

from isaaclab.utils.math import quat_from_euler_xyz


def euler_angles_to_quat(euler_angles: np.ndarray | list[float] | tuple[float, float, float], degrees: bool = False):
    """Convert XYZ Euler angles to an ``(x, y, z, w)`` quaternion tuple."""
    angles = torch.as_tensor(euler_angles, dtype=torch.float32)
    if degrees:
        angles = torch.deg2rad(angles)
    quat = quat_from_euler_xyz(angles[0:1], angles[1:2], angles[2:3])[0]
    return tuple(float(value) for value in quat.tolist())
