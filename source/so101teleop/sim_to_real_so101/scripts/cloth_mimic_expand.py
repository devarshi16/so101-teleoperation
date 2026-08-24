# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Expand 10 SO-101 cloth demos into 200 simulated Mimic-replay episodes.

This generator is deliberately deformable-aware: it replays source joint-space
skills in freshly randomized cloth scenes, adds smooth bounded action noise,
renders new fixed-camera RGB, and keeps only episodes that pass task-specific
cloth geometry gates. It does not pretend that cloth has the single rigid object
frame assumed by stock object-centric MimicGen.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from isaaclab.app import AppLauncher


TASK_IDS = {
    "corner_lift": "SO101-Cloth-Corner-Lift-v0",
    "edge_drag": "SO101-Cloth-Edge-Drag-v0",
    "corner_fold": "SO101-Cloth-Corner-Fold-v0",
    "obstacle_drape_pull": "SO101-Cloth-Obstacle-Drape-Pull-v0",
}

RESET_SETTLE_STEPS = 30

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task_id", choices=sorted(TASK_IDS), required=True)
parser.add_argument("--source_root", type=Path, default=None)
parser.add_argument("--output_root", type=Path, default=None)
parser.add_argument("--source_repo_id", type=str, default=None)
parser.add_argument("--output_repo_id", type=str, default=None)
parser.add_argument("--source_episodes", type=int, default=10)
parser.add_argument("--target_episodes", type=int, default=200)
parser.add_argument("--action_noise_rad", type=float, default=0.008)
parser.add_argument("--max_failures", type=int, default=400)
parser.add_argument("--seed", type=int, default=2026)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

WORKSHOP_ROOT = Path(__file__).resolve().parents[4]
if args_cli.source_root is None:
    args_cli.source_root = WORKSHOP_ROOT / "datasets" / "cloth" / args_cli.task_id / "human_10"
if args_cli.output_root is None:
    args_cli.output_root = WORKSHOP_ROOT / "datasets" / "cloth" / args_cli.task_id / "mimic_200"
if args_cli.source_repo_id is None:
    args_cli.source_repo_id = f"{args_cli.task_id}/human_10"
if args_cli.output_repo_id is None:
    args_cli.output_repo_id = f"{args_cli.task_id}/mimic_200"
if args_cli.target_episodes < 1 or args_cli.source_episodes < 1:
    parser.error("episode counts must be >= 1")
if args_cli.action_noise_rad < 0.0:
    parser.error("--action_noise_rad must be >= 0")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import sim_to_real_so101.tasks  # noqa: F401
from sim_to_real_so101.tasks.cloth_env_cfg import TASK_SPECS


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


def load_source_actions(dataset: LeRobotDataset, expected_episodes: int) -> dict[int, torch.Tensor]:
    """Load action sequences without decoding source videos."""
    if dataset.num_episodes != expected_episodes:
        raise ValueError(f"Expected exactly {expected_episodes} source episodes, found {dataset.num_episodes}")
    dataset._ensure_hf_dataset_loaded()
    grouped: dict[int, list[torch.Tensor]] = {}
    for frame in dataset.hf_dataset:
        episode_index = int(frame["episode_index"].item())
        grouped.setdefault(episode_index, []).append(torch.as_tensor(frame["action"], dtype=torch.float32))
    if len(grouped) != expected_episodes:
        raise ValueError(f"Source frame table contains {len(grouped)} episodes, expected {expected_episodes}")
    return {key: torch.stack(values) for key, values in grouped.items()}


def make_target_dataset(root: Path, repo_id: str) -> LeRobotDataset:
    if root.exists():
        return LeRobotDataset(repo_id, root=root)
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": [
                "shoulder_pan.pos",
                "shoulder_lift.pos",
                "elbow_flex.pos",
                "wrist_flex.pos",
                "wrist_roll.pos",
                "gripper.pos",
            ],
        },
        "observation.images.fixed_rgb": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": [
                "shoulder_pan.pos",
                "shoulder_lift.pos",
                "elbow_flex.pos",
                "wrist_flex.pos",
                "wrist_roll.pos",
                "gripper.pos",
            ],
        },
    }
    return LeRobotDataset.create(repo_id, fps=30, features=features, root=root, robot_type="so101_follower")


def source_region_ids(positions: torch.Tensor) -> torch.Tensor:
    local_xy = positions[0, :, :2]
    source_xy = torch.tensor((0.16, -0.14), device=positions.device)
    return torch.topk(torch.linalg.vector_norm(local_xy - source_xy, dim=-1), k=9, largest=False).indices


def task_succeeded(
    task_id: str,
    positions_local: torch.Tensor,
    source_ids: torch.Tensor,
    initial_area: float,
) -> bool:
    source = positions_local[0, source_ids]
    centroid = positions_local[0].mean(dim=0)
    extent = positions_local[0].amax(dim=0) - positions_local[0].amin(dim=0)
    if task_id == "corner_lift":
        lifted_patch_z = source[:, 2].topk(k=4, largest=True).values.mean()
        return float(lifted_patch_z.item()) >= 0.04 + float(TASK_SPECS[task_id]["lift_height_m"])
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


def smooth_noisy_actions(actions: torch.Tensor, noise_rad: float, generator: torch.Generator) -> torch.Tensor:
    if noise_rad == 0.0:
        return actions.clone()
    noise = torch.zeros_like(actions)
    state = torch.zeros(6)
    for index in range(actions.shape[0]):
        innovation = torch.randn(6, generator=generator) * noise_rad
        state = 0.92 * state + 0.08 * innovation
        noise[index] = state
    noise[:, 5] *= 0.25
    return actions + noise


def main() -> int:
    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    generator = torch.Generator(device="cpu").manual_seed(args_cli.seed)
    gym_id = TASK_IDS[args_cli.task_id]
    source_root = args_cli.source_root.resolve()
    output_root = args_cli.output_root.resolve()
    if source_root == output_root:
        raise ValueError("source and output roots must differ")

    source = LeRobotDataset(args_cli.source_repo_id, root=source_root)
    trajectories = load_source_actions(source, args_cli.source_episodes)
    source_keys = sorted(trajectories)
    target = make_target_dataset(output_root, args_cli.output_repo_id)

    env_cfg = parse_env_cfg(gym_id, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)
    env_cfg.seed = args_cli.seed
    env = gym.make(gym_id, cfg=env_cfg)
    limits = JOINT_LIMITS.to(env.unwrapped.device)
    accepted = int(target.meta.total_episodes)
    failures = 0
    trial = accepted

    print(f"[MIMIC]: {args_cli.task_id} · source={len(source_keys)} · accepted={accepted}/{args_cli.target_episodes}")
    try:
        while simulation_app.is_running() and accepted < args_cli.target_episodes:
            obs, _ = env.reset(seed=args_cli.seed + trial)
            hold_action = obs["policy"]["joint_pos_obs"].clone().to(env.unwrapped.device)
            with torch.inference_mode():
                for _ in range(RESET_SETTLE_STEPS):
                    obs, _, _, _, _ = env.step(hold_action)
            cloth = env.unwrapped.scene["cloth"]
            initial_positions = cloth.data.nodal_pos_w.torch - env.unwrapped.scene.env_origins[:, None, :]
            ids = source_region_ids(initial_positions)
            initial_extent = initial_positions[0].amax(dim=0) - initial_positions[0].amin(dim=0)
            initial_area = float((initial_extent[0] * initial_extent[1]).item())

            source_id = source_keys[trial % len(source_keys)]
            actions = smooth_noisy_actions(trajectories[source_id], args_cli.action_noise_rad, generator)
            for action_cpu in actions:
                action = action_cpu.to(env.unwrapped.device).clamp(limits[:, 0], limits[:, 1]).unsqueeze(0)
                with torch.inference_mode():
                    obs, _, _, _, _ = env.step(action)
                    rgb = obs["visual"]["rgb_fixed_rgb"][0]
                    if rgb.shape[-1] > 3:
                        rgb = rgb[..., :3]
                    target.add_frame(
                        {
                            "action": action[0].detach().cpu().numpy(),
                            "observation.state": obs["policy"]["joint_pos_obs"][0].detach().cpu().numpy(),
                            "observation.images.fixed_rgb": rgb.detach().cpu().numpy(),
                            "task": args_cli.task_id,
                        }
                    )

            final_positions = cloth.data.nodal_pos_w.torch - env.unwrapped.scene.env_origins[:, None, :]
            if task_succeeded(args_cli.task_id, final_positions, ids, initial_area):
                target.save_episode()
                target.finalize()
                target = LeRobotDataset(args_cli.output_repo_id, root=output_root)
                accepted += 1
                print(f"[MIMIC]: accepted {accepted}/{args_cli.target_episodes} from source {source_id}")
            else:
                target.clear_episode_buffer()
                failures += 1
                print(f"[MIMIC]: rejected trial {trial}; failures={failures}/{args_cli.max_failures}")
                if failures >= args_cli.max_failures:
                    raise RuntimeError("Mimic replay reached max_failures before target count")
            trial += 1
    finally:
        env.close()
        simulation_app.close()

    print(f"[DONE]: {accepted} validated episodes at {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
