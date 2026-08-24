# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Joint control with an explicit proximity-based surface-cloth grasp latch."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import omni.usd
from pxr import Gf

from isaaclab.assets import DeformableObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils.configclass import configclass


def _quat_apply(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by normalized XYZW quaternions without TorchScript/NVRTC."""
    xyz = quat[..., :3]
    twice_cross = 2.0 * torch.cross(xyz, vector, dim=-1)
    return vector + quat[..., 3:4] * twice_cross + torch.cross(xyz, twice_cross, dim=-1)


def _quat_apply_inverse(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Inverse-rotate vectors by normalized XYZW quaternions."""
    inverse = torch.cat((-quat[..., :3], quat[..., 3:4]), dim=-1)
    return _quat_apply(inverse, vector)


class ClothGraspJointPositionAction(JointPositionAction):
    """Drive SO-101 joints and latch a small cloth patch between closing fingertips.

    Surface deformables cannot use PhysX nodal kinematic targets. This term therefore
    enables a native vertex-to-rigid PhysX attachment for the captured patch. Capture
    requires both a closing jaw command and geometric proximity to the physical pinch
    center. Opening the jaw disables the attachment immediately.
    """

    cfg: ClothGraspJointPositionActionCfg

    def __init__(self, cfg: ClothGraspJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._cloth: DeformableObject = env.scene[cfg.cloth_asset_name]
        body_ids, _ = self._asset.find_bodies(cfg.gripper_body_name)
        if len(body_ids) != 1:
            raise ValueError(f"Expected one gripper body named {cfg.gripper_body_name!r}, found {body_ids}")
        self._gripper_body_id = body_ids[0]
        try:
            self._jaw_action_id = self._joint_names.index(cfg.jaw_joint_name)
        except ValueError as exc:
            raise ValueError(f"Jaw joint {cfg.jaw_joint_name!r} is not part of the cloth action") from exc

        self._pinch_offset = torch.tensor(cfg.pinch_offset, device=self.device, dtype=torch.float32)
        self._latched = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._node_ids = torch.full(
            (self.num_envs, cfg.num_pinned_nodes), -1, device=self.device, dtype=torch.long
        )
        self._node_offsets = torch.zeros(
            (self.num_envs, cfg.num_pinned_nodes, 3), device=self.device, dtype=torch.float32
        )
        stage = omni.usd.get_context().get_stage()
        self._attachments = [
            stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Cloth/grasp_attachment")
            for env_id in range(self.num_envs)
        ]
        missing = [index for index, prim in enumerate(self._attachments) if not prim.IsValid()]
        if missing:
            raise RuntimeError(f"Missing cloth grasp attachments for environment(s): {missing}")

    def _resolved_env_ids(self, env_ids: Sequence[int] | None) -> list[int]:
        if env_ids is None:
            return list(range(self.num_envs))
        if isinstance(env_ids, slice):
            return list(range(self.num_envs))[env_ids]
        if isinstance(env_ids, torch.Tensor):
            return [int(value) for value in env_ids.detach().cpu().tolist()]
        return [int(value) for value in env_ids]

    def _set_attachment_enabled(self, env_id: int, enabled: bool) -> None:
        self._attachments[env_id].GetAttribute("omniphysics:attachmentEnabled").Set(enabled)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        for env_id in self._resolved_env_ids(env_ids):
            self._set_attachment_enabled(env_id, False)
        self._latched[env_ids] = False
        self._node_ids[env_ids] = -1
        self._node_offsets[env_ids] = 0.0

    def _gripper_pose(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        body_pos = self._asset.data.body_pos_w.torch[:, self._gripper_body_id]
        body_quat = self._asset.data.body_quat_w.torch[:, self._gripper_body_id]
        offset = self._pinch_offset.expand(self.num_envs, -1)
        pinch_pos = body_pos + _quat_apply(body_quat, offset)
        return body_pos, pinch_pos, body_quat

    def _capture_nearby_nodes(
        self, body_pos: torch.Tensor, pinch_pos: torch.Tensor, body_quat: torch.Tensor
    ) -> None:
        jaw_target = self.processed_actions[:, self._jaw_action_id]
        candidates = (~self._latched) & (jaw_target >= self.cfg.capture_jaw_position)
        if not candidates.any():
            return

        positions = self._cloth.data.nodal_pos_w.torch
        distances = torch.linalg.vector_norm(positions - pinch_pos[:, None, :], dim=-1)
        nearest_distances, nearest_ids = torch.topk(
            distances, k=self.cfg.num_pinned_nodes, dim=1, largest=False
        )
        captured = candidates & (nearest_distances[:, 0] <= self.cfg.capture_radius)
        if not captured.any():
            return

        env_ids = torch.where(captured)[0]
        selected_ids = nearest_ids[env_ids]
        selected = positions[env_ids[:, None], selected_ids]
        quats = body_quat[env_ids, None, :].expand(-1, self.cfg.num_pinned_nodes, -1)
        offsets = _quat_apply_inverse(quats, selected - body_pos[env_ids, None, :])
        # The local X axis is the SO-101 jaw closing axis. Centering captured
        # nodes on that plane prevents a one-sided sheet from being squeezed out.
        offsets[..., 0] = 0.0

        self._latched[env_ids] = True
        self._node_ids[env_ids] = selected_ids
        self._node_offsets[env_ids] = offsets
        for row, env_id_tensor in enumerate(env_ids):
            env_id = int(env_id_tensor.item())
            indices = [int(value) for value in selected_ids[row].detach().cpu().tolist()]
            local_positions = [
                Gf.Vec3f(*[float(component) for component in point])
                for point in offsets[row].detach().cpu().tolist()
            ]
            attachment = self._attachments[env_id]
            attachment.GetAttribute("omniphysics:vtxIndicesSrc0").Set(indices)
            attachment.GetAttribute("omniphysics:localPositionsSrc1").Set(local_positions)
            self._set_attachment_enabled(env_id, True)
        print(f"[CLOTH GRASP]: latched {self.cfg.num_pinned_nodes} nodes in env(s) {env_ids.tolist()}")

    def _release_open_grippers(self) -> None:
        jaw_target = self.processed_actions[:, self._jaw_action_id]
        released = self._latched & (jaw_target <= self.cfg.release_jaw_position)
        if released.any():
            env_ids = torch.where(released)[0]
            for env_id in [int(value) for value in env_ids.detach().cpu().tolist()]:
                self._set_attachment_enabled(env_id, False)
            self._latched[env_ids] = False
            self._node_ids[env_ids] = -1
            self._node_offsets[env_ids] = 0.0
            print(f"[CLOTH GRASP]: released nodes in env(s) {env_ids.tolist()}")

    def apply_actions(self) -> None:
        super().apply_actions()
        self._release_open_grippers()
        body_pos, pinch_pos, body_quat = self._gripper_pose()
        self._capture_nearby_nodes(body_pos, pinch_pos, body_quat)


@configclass
class ClothGraspJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for the explicit SO-101 surface-cloth grasp latch."""

    class_type: type = ClothGraspJointPositionAction
    cloth_asset_name: str = "cloth"
    gripper_body_name: str = "gripper"
    jaw_joint_name: str = "Jaw"
    pinch_offset: tuple[float, float, float] = (0.0, 0.0, -0.1018)
    capture_jaw_position: float = 0.0
    release_jaw_position: float = -0.08
    capture_radius: float = 0.014
    num_pinned_nodes: int = 4
