# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect a fixed number of SO-101 cloth demonstrations with an evdev gamepad."""

from __future__ import annotations

import argparse
import os
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


TASK_IDS = {
    "corner_lift": "SO101-Cloth-Corner-Lift-v0",
    "edge_drag": "SO101-Cloth-Edge-Drag-v0",
    "corner_fold": "SO101-Cloth-Corner-Fold-v0",
    "obstacle_drape_pull": "SO101-Cloth-Obstacle-Drape-Pull-v0",
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task_id", choices=sorted(TASK_IDS), required=True)
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--dataset_root", type=Path, default=None)
parser.add_argument("--repo_id", type=str, default=None)
parser.add_argument("--controller_device", type=str, default=None)
parser.add_argument("--joint_rate", type=float, default=0.75, help="Stick joint rate in rad/s.")
parser.add_argument("--gripper_rate", type=float, default=1.2, help="Gripper joint rate in rad/s.")
parser.add_argument("--seed", type=int, default=101)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--smoke_steps", type=int, default=0, help=argparse.SUPPRESS)
parser.add_argument("--smoke_grasp_latch", action="store_true", help=argparse.SUPPRESS)
parser.add_argument("--smoke_task_success", action="store_true", help=argparse.SUPPRESS)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

if args_cli.episodes < 1:
    parser.error("--episodes must be >= 1")

WORKSHOP_ROOT = Path(__file__).resolve().parents[4]
if args_cli.dataset_root is None:
    args_cli.dataset_root = WORKSHOP_ROOT / "datasets" / "cloth" / args_cli.task_id / "human_10"
if args_cli.repo_id is None:
    args_cli.repo_id = f"{args_cli.task_id}/human_10"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import sim_to_real_so101.tasks  # noqa: F401
from sim_to_real_so101.tasks.cloth_env_cfg import TASK_SPECS
from sim_to_real_so101.utils.gamepad import EvdevController
from sim_to_real_so101.utils.keyboard import KeyboardControl
from sim_to_real_so101.utils.lerobot_recorder import LeRobotRecorder


JOINT_LIMITS = torch.tensor(
    [
        [-1.920, 1.920],
        [-1.745, 1.745],
        [-1.745, 1.571],
        [-1.658, 1.658],
        [-2.793, 2.793],
        [-0.175, 1.745],
    ],
    dtype=torch.float32,
)

RESET_SETTLE_STEPS = 30
SUCCESS_CONFIRM_STEPS = 3
SOURCE_CORNER_XY = (0.16, -0.14)
TABLE_TOP_Z = 0.04


def reset_and_settle(env):
    """Reset, then hold the robot still for one simulated second while the cloth settles."""
    obs, _ = env.reset()
    targets = obs["policy"]["joint_pos_obs"].clone().to(env.unwrapped.device)
    with torch.inference_mode():
        for _ in range(RESET_SETTLE_STEPS):
            obs, _, _, _, _ = env.step(targets)
    return obs, targets


def _cloth_positions_local(env) -> torch.Tensor:
    """Return cloth vertices relative to each environment origin."""
    positions = env.unwrapped.scene["cloth"].data.nodal_pos_w.torch
    return positions - env.unwrapped.scene.env_origins[:, None, :]


def _source_region_ids(positions_local: torch.Tensor) -> torch.Tensor:
    """Resolve the marked x-min/y-min corner patch once at episode start."""
    source_xy = torch.tensor(SOURCE_CORNER_XY, device=positions_local.device)
    distances = torch.linalg.vector_norm(positions_local[0, :, :2] - source_xy, dim=-1)
    return torch.topk(distances, k=9, largest=False).indices


def _initial_cloth_area(positions_local: torch.Tensor) -> float:
    extent = positions_local[0, :, :2].amax(dim=0) - positions_local[0, :, :2].amin(dim=0)
    return float((extent[0] * extent[1]).item())


def _task_succeeded(
    task_id: str,
    positions_local: torch.Tensor,
    source_ids: torch.Tensor,
    initial_area: float,
) -> bool:
    """Evaluate the same geometric task outcomes used to accept Mimic rollouts."""
    source = positions_local[0, source_ids]
    centroid = positions_local[0].mean(dim=0)
    extent = positions_local[0].amax(dim=0) - positions_local[0].amin(dim=0)
    if task_id == "corner_lift":
        # Four captured vertices must form a clearly unsupported lifted patch.
        lifted_patch_z = source[:, 2].topk(k=4, largest=True).values.mean()
        threshold = TABLE_TOP_Z + float(TASK_SPECS[task_id]["lift_height_m"])
        return float(lifted_patch_z.item()) >= threshold
    if task_id == "edge_drag":
        target = torch.tensor(TASK_SPECS[task_id]["target_xy"], device=positions_local.device)
        close = torch.linalg.vector_norm(source[:, :2].mean(dim=0) - target).item() <= 0.07
        return close and float(source[:, 2].mean().item()) <= 0.12
    if task_id == "corner_fold":
        target = torch.tensor(TASK_SPECS[task_id]["target_xy"], device=positions_local.device)
        close = torch.linalg.vector_norm(source[:, :2].mean(dim=0) - target).item() <= 0.075
        final_area = float((extent[0] * extent[1]).item())
        return close and final_area <= 0.80 * initial_area
    target_x = float(TASK_SPECS[task_id]["target_xy"][0])
    coverage = (positions_local[0, :, 0] >= target_x - 0.03).float().mean().item()
    return coverage >= 0.60 and float(centroid[2].item()) >= 0.075


def main() -> int:
    gym_id = TASK_IDS[args_cli.task_id]
    spec = TASK_SPECS[args_cli.task_id]
    dataset_root = args_cli.dataset_root.resolve()
    dataset_root.parent.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(
        gym_id,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    env = gym.make(gym_id, cfg=env_cfg)
    if args_cli.smoke_task_success:
        reset_and_settle(env)
        positions = _cloth_positions_local(env)
        source_ids = _source_region_ids(positions)
        initial_area = _initial_cloth_area(positions)
        baseline = _task_succeeded("corner_lift", positions, source_ids, initial_area)

        marked_lift = positions.clone()
        marked_lift[0, source_ids[:4], 2] = 0.11
        marked_success = _task_succeeded("corner_lift", marked_lift, source_ids, initial_area)

        opposite_xy = torch.tensor((0.44, 0.14), device=positions.device)
        opposite_distances = torch.linalg.vector_norm(positions[0, :, :2] - opposite_xy, dim=-1)
        opposite_ids = torch.topk(opposite_distances, k=4, largest=False).indices
        opposite_lift = positions.clone()
        opposite_lift[0, opposite_ids, 2] = 0.11
        opposite_success = _task_succeeded("corner_lift", opposite_lift, source_ids, initial_area)
        print(
            f"[SMOKE SUCCESS]: baseline={baseline} marked_corner={marked_success} "
            f"opposite_corner={opposite_success} source_ids={source_ids.tolist()}"
        )
        env.close()
        simulation_app.close()
        return 0 if not baseline and marked_success and not opposite_success else 2
    if args_cli.smoke_grasp_latch:
        obs, _ = env.reset()
        action_term = env.unwrapped.action_manager.get_term("joint_positions")
        cloth = env.unwrapped.scene["cloth"]
        _, pinch_pos, _ = action_term._gripper_pose()
        positions = cloth.data.nodal_pos_w.torch.clone()
        velocities = cloth.data.nodal_vel_w.torch.clone()
        positions_local = positions - env.unwrapped.scene.env_origins[:, None, :]
        source_ids = _source_region_ids(positions_local)
        initial_area = _initial_cloth_area(positions_local)
        test_offsets = torch.tensor(
            ((0.0, -0.003, -0.003), (0.0, -0.003, 0.003), (0.0, 0.003, -0.003), (0.0, 0.003, 0.003)),
            device=env.unwrapped.device,
        )
        positions[0, source_ids[:4]] = pinch_pos[0] + test_offsets
        velocities[0, source_ids[:4]] = 0.0
        cloth.write_nodal_pos_to_sim_index(positions)
        cloth.write_nodal_velocity_to_sim_index(velocities)
        actions = obs["policy"]["joint_pos_obs"].clone().to(env.unwrapped.device)
        actions[:, 5] = 0.3
        for _ in range(5):
            obs, _, _, _, _ = env.step(actions)
        latched_after_close = bool(action_term._latched[0].item())
        actions[:, 0] += 0.12
        for _ in range(15):
            obs, _, _, _, _ = env.step(actions)
        _, moved_pinch_pos, _ = action_term._gripper_pose()
        held_ids = action_term._node_ids[0]
        held_positions = cloth.data.nodal_pos_w.torch[0, held_ids]
        max_follow_distance = float(
            torch.linalg.vector_norm(held_positions - moved_pinch_pos[0], dim=-1).max().item()
        )
        followed_moving_hand = max_follow_distance <= 0.02
        lift_success = _task_succeeded(
            "corner_lift",
            _cloth_positions_local(env),
            source_ids,
            initial_area,
        )
        actions[:, 5] = -0.15
        obs, _, _, _, _ = env.step(actions)
        released_after_open = not bool(action_term._latched[0].item())
        print(
            f"[SMOKE GRASP]: latched_after_close={latched_after_close} "
            f"followed_moving_hand={followed_moving_hand} max_distance={max_follow_distance:.4f}m "
            f"lift_success={lift_success} "
            f"released_after_open={released_after_open}"
        )
        env.close()
        simulation_app.close()
        return 0 if latched_after_close and followed_moving_hand and lift_success and released_after_open else 2
    if args_cli.smoke_steps > 0:
        obs, _ = env.reset()
        actions = obs["policy"]["joint_pos_obs"].clone().to(env.unwrapped.device)
        initial_cloth_state = obs["diagnostics"]["cloth_state"][0].detach().cpu().tolist()
        for _ in range(args_cli.smoke_steps):
            obs, _, _, _, _ = env.step(actions)
        rgb = obs["visual"]["rgb_fixed_rgb"]
        cloth_state = obs["diagnostics"]["cloth_state"][0].detach().cpu().tolist()
        import omni.usd
        from pxr import Usd, UsdGeom

        stage = omni.usd.get_context().get_stage()
        collision_offsets = {}
        cloth_geometry = {}
        cloth_schemas = {}
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if path.startswith(
                (
                    "/World/envs/env_0/Table",
                    "/World/envs/env_0/Cloth",
                    "/World/envs/env_0/SourceMarker",
                    "/World/envs/env_0/Robot/wrist/collisions",
                    "/World/envs/env_0/Robot/gripper/collisions",
                    "/World/envs/env_0/Robot/jaw/collisions",
                )
            ):
                if path.startswith("/World/envs/env_0/Cloth"):
                    cloth_schemas[path] = prim.GetAppliedSchemas()
                contact = prim.GetAttribute("physxCollision:contactOffset")
                rest = prim.GetAttribute("physxCollision:restOffset")
                if contact.IsValid() or rest.IsValid():
                    collision_offsets[path] = {
                        "contact": contact.Get() if contact.IsValid() else None,
                        "rest": rest.Get() if rest.IsValid() else None,
                    }
                if path == "/World/envs/env_0/Cloth":
                    cloth_geometry["translate"] = prim.GetAttribute("xformOp:translate").Get()
                if path == "/World/envs/env_0/Cloth/geometry/mesh":
                    points = prim.GetAttribute("points").Get()
                    cloth_geometry["mesh_z_bounds"] = (
                        min(float(point[2]) for point in points),
                        max(float(point[2]) for point in points),
                    )
                if path == "/World/envs/env_0/Cloth/geometry/material":
                    cloth_geometry["surface_thickness"] = prim.GetAttribute(
                        "omniphysics:surfaceThickness"
                    ).Get()
        print(
            f"[SMOKE]: {gym_id} steps={args_cli.smoke_steps} "
            f"rgb={tuple(rgb.shape)} mean={float(rgb.float().mean().item()):.2f} "
            f"std={float(rgb.float().std().item()):.2f} "
            f"initial_cloth_state={initial_cloth_state} final_cloth_state={cloth_state}"
        )
        print(f"[SMOKE]: collision_offsets={collision_offsets}")
        print(f"[SMOKE]: cloth_schemas={cloth_schemas}")
        print(
            f"[SMOKE]: cfg_cloth_pos={env.unwrapped.cfg.scene.cloth.init_state.pos} "
            f"cloth_geometry={cloth_geometry}"
        )
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        table_range = bbox_cache.ComputeWorldBound(stage.GetPrimAtPath("/World/envs/env_0/Table")).ComputeAlignedRange()
        print(f"[SMOKE]: table_world_bounds={table_range.GetMin()}..{table_range.GetMax()}")
        marker_range = bbox_cache.ComputeWorldBound(
            stage.GetPrimAtPath("/World/envs/env_0/SourceMarker")
        ).ComputeAlignedRange()
        print(f"[SMOKE]: source_marker_world_bounds={marker_range.GetMin()}..{marker_range.GetMax()}")
        target_range = bbox_cache.ComputeWorldBound(
            stage.GetPrimAtPath("/World/envs/env_0/TargetMarker")
        ).ComputeAlignedRange()
        print(f"[SMOKE]: target_marker_world_bounds={target_range.GetMin()}..{target_range.GetMax()}")
        cylinder_prim = stage.GetPrimAtPath("/World/envs/env_0/Cylinder")
        if cylinder_prim.IsValid():
            cylinder_range = bbox_cache.ComputeWorldBound(cylinder_prim).ComputeAlignedRange()
            print(f"[SMOKE]: cylinder_world_bounds={cylinder_range.GetMin()}..{cylinder_range.GetMax()}")
        hand_material = stage.GetPrimAtPath("/World/envs/env_0/Robot/hand_physics_material")
        hand_bindings = {}
        for link_name in ("wrist", "gripper", "jaw"):
            collision_prim = stage.GetPrimAtPath(f"/World/envs/env_0/Robot/{link_name}/collisions")
            hand_bindings[link_name] = [
                str(target)
                for target in collision_prim.GetRelationship("material:binding:physics").GetTargets()
            ]
        print(
            "[SMOKE]: hand_material="
            f"static={hand_material.GetAttribute('physics:staticFriction').Get()} "
            f"dynamic={hand_material.GetAttribute('physics:dynamicFriction').Get()} "
            f"bindings={hand_bindings}"
        )
        env.close()
        simulation_app.close()
        return 0
    keyboard = KeyboardControl(reset_recording_policy="cancel")
    controller = EvdevController(args_cli.controller_device)
    recorder = LeRobotRecorder(
        task_name=args_cli.task_id,
        repo_id=args_cli.repo_id,
        dataset_root=str(dataset_root),
        fps=30,
        device=env.unwrapped.device,
        cameras={"fixed_rgb": {"height": 480, "width": 640}},
        save_mp4=False,
        depth=False,
        instance_id_seg=False,
        max_episode_seconds=float(spec["timeout_s"]) + 1.0,
    )
    recorder.init_dataset()

    print("\n[COLLECTION]")
    print(f"  Task ID:       {args_cli.task_id}")
    print(f"  Gym ID:        {gym_id}")
    print(f"  Instruction:   {spec['instruction']}")
    print(f"  Dataset:       {dataset_root}")
    print(f"  Target count:  {args_cli.episodes}")
    print(f"  Existing:      {recorder.total_episodes}")
    if args_cli.task_id == "corner_lift":
        print("  Pickup corner: above the orange riser at (x=0.16, y=-0.14); robot-near/camera-facing")
        print("  Auto-success:  marked corner patch at least 6 cm above the table for 3 frames")
    print("  Auto-success is evaluated after S starts recording.")
    print("  D-pad left/right: Joint 3 (elbow) · D-pad up/down: Joint 4 (wrist pitch)")
    print("  S: start/stop and SAVE · C: cancel current take · R: cancel and reset")

    obs, targets = reset_and_settle(env)
    initial_positions = _cloth_positions_local(env)
    source_ids = _source_region_ids(initial_positions)
    initial_area = _initial_cloth_area(initial_positions)
    success_streak = 0
    limits = JOINT_LIMITS.to(env.unwrapped.device)
    last_time = time.monotonic()
    recording_started_at = None
    last_progress_time = 0.0

    try:
        while simulation_app.is_running() and recorder.total_episodes < args_cli.episodes:
            loop_start = time.monotonic()
            with torch.inference_mode():
                dt = min(max(loop_start - last_time, 1.0 / 240.0), 1.0 / 15.0)
                last_time = loop_start
                command = controller.poll().to(env.unwrapped.device)
                targets[:, :5] += command[:5].unsqueeze(0) * args_cli.joint_rate * dt
                targets[:, 5] += command[5] * args_cli.gripper_rate * dt
                targets[:] = torch.max(torch.min(targets, limits[:, 1]), limits[:, 0])
                obs, _, _, _, _ = env.step(targets)

                if keyboard.reset_world:
                    keyboard.reset_world = False
                    obs, targets = reset_and_settle(env)
                    initial_positions = _cloth_positions_local(env)
                    source_ids = _source_region_ids(initial_positions)
                    initial_area = _initial_cloth_area(initial_positions)
                    success_streak = 0
                    recording_started_at = None
                    continue

                if keyboard.recording:
                    if recording_started_at is None:
                        recording_started_at = loop_start
                    rgb = obs["visual"]["rgb_fixed_rgb"][0]
                    if rgb.shape[-1] > 3:
                        rgb = rgb[..., :3]
                    recorder.push_frame_to_buffer(
                        targets[0],
                        obs["policy"]["joint_pos_obs"][0],
                        {"fixed_rgb": rgb},
                        {},
                        {},
                    )
                    positions_local = _cloth_positions_local(env)
                    if _task_succeeded(
                        args_cli.task_id,
                        positions_local,
                        source_ids,
                        initial_area,
                    ):
                        success_streak += 1
                    else:
                        success_streak = 0

                    if args_cli.task_id == "corner_lift" and loop_start - last_progress_time >= 0.5:
                        source_z = positions_local[0, source_ids, 2].topk(k=4).values.mean().item()
                        action_term = env.unwrapped.action_manager.get_term("joint_positions")
                        print(
                            f"[LIFT PROGRESS]: height_above_table={max(0.0, source_z - TABLE_TOP_Z):.3f}m "
                            f"target={TASK_SPECS['corner_lift']['lift_height_m']:.3f}m "
                            f"latched={bool(action_term._latched[0].item())} "
                            f"confirm={success_streak}/{SUCCESS_CONFIRM_STEPS}"
                        )
                        last_progress_time = loop_start

                    if success_streak >= SUCCESS_CONFIRM_STEPS:
                        print(f"[SUCCESS]: {args_cli.task_id} criterion confirmed; saving take and resetting.")
                        keyboard.stop_recording()
                        obs, targets = reset_and_settle(env)
                        initial_positions = _cloth_positions_local(env)
                        source_ids = _source_region_ids(initial_positions)
                        initial_area = _initial_cloth_area(initial_positions)
                        success_streak = 0
                        recording_started_at = None
                        continue
                    if loop_start - recording_started_at >= float(spec["timeout_s"]):
                        print(f"[INFO]: {spec['timeout_s']:.0f}s timeout without success; cancelling take and resetting.")
                        keyboard.cancel_recording()
                        obs, targets = reset_and_settle(env)
                        initial_positions = _cloth_positions_local(env)
                        source_ids = _source_region_ids(initial_positions)
                        initial_area = _initial_cloth_area(initial_positions)
                        success_streak = 0
                        recording_started_at = None
                        continue
                else:
                    recording_started_at = None
                    success_streak = 0

            remaining = 1.0 / 30.0 - (time.monotonic() - loop_start)
            if remaining > 0.0:
                time.sleep(remaining)

        if recorder.total_episodes >= args_cli.episodes:
            print(f"[INFO]: Target reached: {recorder.total_episodes}/{args_cli.episodes} episodes.")
    except KeyboardInterrupt:
        print("[INFO]: Collection interrupted; completed episodes remain resumable.")
    except BaseException as exc:
        print(f"[ERROR]: Collection failed with {type(exc).__name__}: {exc!r}")
        traceback.print_exc()
        raise
    finally:
        if keyboard.recording:
            keyboard.cancel_recording()
        controller.close()
        keyboard.cleanup()
        recorder.close()
        env.close()
        simulation_app.close()

    print(f"[DONE]: {recorder.total_episodes} episodes at {dataset_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
