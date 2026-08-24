# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay one Isaac Lab HDF5 episode and render it to an MP4 file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset_file", type=Path, required=True)
parser.add_argument("--task", type=str, default=None, help="Override the task stored in HDF5 metadata.")
parser.add_argument("--episode", type=int, default=0)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--start_frame", type=int, default=0)
parser.add_argument("--end_frame", type=int, default=None)
parser.add_argument("--frame_stride", type=int, default=1)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=540)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.video = True

if args_cli.episode < 0:
    parser.error("--episode must be >= 0")
if args_cli.start_frame < 0:
    parser.error("--start_frame must be >= 0")
if args_cli.frame_stride < 1:
    parser.error("--frame_stride must be >= 1")
if args_cli.fps < 1 or args_cli.width < 1 or args_cli.height < 1:
    parser.error("--fps, --width, and --height must be >= 1")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import h5py
import imageio.v2 as imageio
import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401
from isaaclab.sensors import TiledCameraCfg
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.datasets import HDF5DatasetFileHandler

import sim_to_real_so101.tasks  # noqa: F401


def _task_from_metadata(dataset_file: Path) -> str | None:
    with h5py.File(dataset_file, "r") as stream:
        raw = stream["data"].attrs.get("env_args", "{}")
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        metadata = json.loads(raw)
    return metadata.get("task") or metadata.get("env_name")


def _restore_recorded_initial_state(env, episode) -> None:
    """Restore the custom state fields written by ``controller_agent``."""
    states = episode.data.get("states", {})
    robot_state = states.get("robot", {})
    cube_state = states.get("cube", {})
    if "joint_pos" in robot_state and "joint_vel" in robot_state:
        env.scene["robot"].write_joint_state_to_sim(
            robot_state["joint_pos"][0:1],
            robot_state["joint_vel"][0:1],
        )
    if "root_pose" in cube_state:
        env.scene["cube"].write_root_pose_to_sim(cube_state["root_pose"][0:1])
    env.scene.write_data_to_sim()
    env.sim.forward()


def main() -> int:
    dataset_file = args_cli.dataset_file.expanduser().resolve()
    if not dataset_file.is_file():
        raise FileNotFoundError(dataset_file)

    handler = HDF5DatasetFileHandler()
    handler.open(str(dataset_file))
    episode_names = list(handler.get_episode_names())
    if args_cli.episode >= len(episode_names):
        raise IndexError(
            f"Episode {args_cli.episode} does not exist; dataset contains {len(episode_names)} episode(s)."
        )

    task = args_cli.task or _task_from_metadata(dataset_file)
    if not task:
        raise ValueError("No task was provided and no task/env_name exists in the HDF5 metadata.")

    env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=1)
    recorded_seed = handler.load_episode(episode_names[args_cli.episode], args_cli.device).seed
    env_cfg.seed = int(recorded_seed or env_cfg.seed)
    env_cfg.recorders = {}
    env_cfg.terminations = {}
    env_cfg.scene.replay_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/ReplayCamera",
        update_period=0.0,
        height=args_cli.height,
        width=args_cli.width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 20.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(convention="opengl"),
    )
    camera_eye = tuple(env_cfg.viewer.eye)
    camera_target = tuple(env_cfg.viewer.lookat)

    env = gym.make(task, cfg=env_cfg).unwrapped
    episode = handler.load_episode(episode_names[args_cli.episode], env.device)
    actions = episode.data["actions"]
    end_frame = len(actions) if args_cli.end_frame is None else min(args_cli.end_frame, len(actions))
    if args_cli.start_frame >= end_frame:
        raise ValueError(f"Empty replay range: start={args_cli.start_frame}, end={end_frame}.")

    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    env.reset(seed=int(episode.seed))
    _restore_recorded_initial_state(env, episode)
    camera = env.scene["replay_camera"]
    origins = env.scene.env_origins
    eye = origins + torch.tensor([camera_eye], device=env.device)
    target = origins + torch.tensor([camera_target], device=env.device)
    camera.set_world_poses_from_view(eye, target)

    captured = 0
    writer = imageio.get_writer(
        args_cli.output,
        fps=args_cli.fps,
        codec="libx264",
        quality=8,
        macro_block_size=2,
    )
    try:
        with torch.inference_mode():
            for frame_index, action in enumerate(actions):
                env.step(action.unsqueeze(0))
                if frame_index < args_cli.start_frame:
                    continue
                if frame_index >= end_frame:
                    break
                if (frame_index - args_cli.start_frame) % args_cli.frame_stride:
                    continue
                frame = camera.data.output["rgb"][0]
                frame = frame[..., :3].detach().cpu().numpy()
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                if frame.any():
                    writer.append_data(frame)
                    captured += 1
    finally:
        writer.close()
        handler.close()
        env.close()

    if captured == 0:
        raise RuntimeError("The renderer returned no non-empty frames.")
    print(f"[DONE] Rendered {captured} frames from {task} to {args_cli.output.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
