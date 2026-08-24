# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controller teleop for the SO101 cube-to-box task."""

import argparse
import os
import time
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SO101 gamepad/controller teleop with HDF5 success recording.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Lerobot-So101-Controller-Cube-Box")
parser.add_argument("--dataset_file", type=str, default="datasets/so101_cube_box_controller.hdf5")
parser.add_argument("--controller_device", type=str, default=None, help="Optional evdev path, e.g. /dev/input/by-id/...-event-joystick")
parser.add_argument("--seed", type=int, default=101)
parser.add_argument("--joint_rate", type=float, default=0.75, help="Radians/sec for stick-driven joints.")
parser.add_argument("--gripper_rate", type=float, default=1.2, help="Radians/sec for trigger/button-driven gripper.")
parser.add_argument("--debug_controller", action="store_true", help="Print controller loop diagnostics once per second.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

import sim_to_real_so101.tasks  # noqa: F401


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


class EvdevController:
    """Small Linux gamepad reader that works even when Kit has no controller remapping."""

    def __init__(self, device_path: str | None = None, deadzone: float = 0.12):
        self.deadzone = deadzone
        self.device_path = device_path
        self.device = None
        self.grabbed = False
        self.axes = {}
        self.buttons = set()
        self.abs_info = {}
        self.last_scan_time = 0.0
        try:
            from evdev import InputDevice, ecodes, list_devices

            self.InputDevice = InputDevice
            self.ecodes = ecodes
            self.list_devices = list_devices
            self._connect(print_devices=True)
            if self.device is None:
                print("[WARNING]: No evdev gamepad found. Actions will stay at reset targets.")
        except Exception as exc:
            print(f"[WARNING]: Could not initialize evdev controller: {exc}")

    def _candidate_paths(self, print_devices: bool = False) -> list[str]:
        available_paths = self.list_devices()
        if print_devices and available_paths:
            print("[INFO]: Available evdev input devices:")
            for path in available_paths:
                try:
                    dev = self.InputDevice(path)
                    print(f"[INFO]:   {path}: {dev.name}")
                except OSError as exc:
                    print(f"[WARNING]:   {path}: unavailable ({exc})")

        if self.device_path:
            if os.path.exists(self.device_path):
                return [self.device_path]
            if print_devices:
                print(f"[WARNING]: Requested controller device does not exist: {self.device_path}")
                print("[WARNING]: Falling back to automatic evdev controller detection.")
        return available_paths

    def _connect(self, print_devices: bool = False):
        self.last_scan_time = time.monotonic()
        for path in self._candidate_paths(print_devices=print_devices):
            try:
                dev = self.InputDevice(path)
            except OSError as exc:
                if print_devices:
                    print(f"[WARNING]: Could not open input device {path}: {exc}")
                continue
            caps = dev.capabilities()
            if self.ecodes.EV_ABS in caps and self.ecodes.EV_KEY in caps:
                try:
                    dev.grab()
                    self.grabbed = True
                except OSError as exc:
                    print(f"[WARNING]: Could not exclusively grab controller {dev.path}: {exc}")
                os.set_blocking(dev.fd, False)
                self.device = dev
                self.abs_info = {code: info for code, info in dev.capabilities(absinfo=True).get(self.ecodes.EV_ABS, [])}
                self.axes = {code: info.value for code, info in self.abs_info.items()}
                print(f"[INFO]: Controller device: {dev.path} ({dev.name})")
                return

    def close(self):
        if self.device is not None and self.grabbed:
            try:
                self.device.ungrab()
            except Exception:
                pass
        self.device = None
        self.grabbed = False

    def _axis_value(self, code: int) -> float:
        raw = self.axes.get(code, 0)
        info = self.abs_info.get(code)
        if info is None:
            return 0.0
        center = (info.max + info.min) * 0.5
        radius = max((info.max - info.min) * 0.5, 1.0)
        value = (raw - center) / radius
        if abs(value) < self.deadzone:
            return 0.0
        return max(-1.0, min(1.0, value))

    def poll(self) -> torch.Tensor:
        if self.device is None:
            if time.monotonic() - self.last_scan_time > 1.0:
                self._connect()
            return torch.zeros(6)
        try:
            events = list(self.device.read())
        except BlockingIOError:
            events = []
        except OSError as exc:
            print(f"[WARNING]: Lost controller device {self.device.path}: {exc}")
            self.close()
            return torch.zeros(6)
        for event in events:
            if event.type == self.ecodes.EV_ABS:
                self.axes[event.code] = event.value
            elif event.type == self.ecodes.EV_KEY:
                if event.value:
                    self.buttons.add(event.code)
                else:
                    self.buttons.discard(event.code)

        e = self.ecodes
        command = torch.zeros(6, dtype=torch.float32)
        command[0] = self._axis_value(e.ABS_X)
        command[1] = -self._axis_value(e.ABS_Y)
        command[2] = -self._axis_value(e.ABS_RY)
        command[3] = -self._axis_value(e.ABS_HAT0Y)
        command[4] = self._axis_value(e.ABS_RX)
        close_grip = float(e.BTN_SOUTH in self.buttons or e.BTN_TR in self.buttons)
        open_grip = float(e.BTN_EAST in self.buttons or e.BTN_TL in self.buttons)
        command[5] = close_grip - open_grip
        return command


def _episode_add_obs(episode: EpisodeData, obs: dict):
    for group_name, group_obs in obs.items():
        if isinstance(group_obs, dict):
            for key, value in group_obs.items():
                episode.add(f"obs/{group_name}/{key}", value[0].detach().cpu())


def _episode_add_state(episode: EpisodeData, env, obs: dict):
    robot = env.unwrapped.scene["robot"]
    cube = env.unwrapped.scene["cube"]
    episode.add("states/robot/joint_pos", robot.data.joint_pos[0].detach().cpu())
    episode.add("states/robot/joint_vel", robot.data.joint_vel[0].detach().cpu())
    episode.add("states/cube/root_pose", obs["object"]["cube_pose"][0].detach().cpu())


def _open_dataset(path: str, task: str) -> HDF5DatasetFileHandler:
    path = os.path.abspath(path)
    handler = HDF5DatasetFileHandler()
    if os.path.exists(path):
        handler.open(path, "r+")
    else:
        handler.create(path, env_name=task)
    handler.add_env_args({"task": task, "teleop": "evdev_controller", "format": "isaaclab_episode_data"})
    return handler


def _wait_for_app_running(timeout_s: float = 5.0) -> bool:
    start_time = time.monotonic()
    while not simulation_app.is_running() and not simulation_app.is_exiting():
        if time.monotonic() - start_time > timeout_s:
            print("[WARNING]: Simulation app did not report running before timeout.")
            return False
        simulation_app.update()
        time.sleep(0.05)
    return simulation_app.is_running() and not simulation_app.is_exiting()


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    os.makedirs(os.path.dirname(args_cli.dataset_file) or ".", exist_ok=True)
    controller = EvdevController(args_cli.controller_device)
    dataset = _open_dataset(args_cli.dataset_file, args_cli.task)

    obs, _ = env.reset()
    targets = obs["policy"]["joint_pos_obs"].clone().to(env.unwrapped.device)
    episode = EpisodeData()
    episode.seed = args_cli.seed
    episode.env_id = 0
    last_time = time.monotonic()
    last_debug_time = last_time
    step_count = 0

    try:
        if simulation_app.is_exiting():
            print("[WARNING]: Simulation app is already exiting before controller loop starts.")
        print("[INFO]: Controller loop started. Close the Kit window or press Ctrl-C to stop.")
        while True:
            with torch.inference_mode():
                now = time.monotonic()
                dt = min(max(now - last_time, 1.0 / 240.0), 1.0 / 15.0)
                last_time = now
                command = controller.poll().to(env.unwrapped.device)
                targets[:, :5] += command[:5].unsqueeze(0) * args_cli.joint_rate * dt
                targets[:, 5] += command[5] * args_cli.gripper_rate * dt
                limits = JOINT_LIMITS.to(env.unwrapped.device)
                targets[:] = torch.max(torch.min(targets, limits[:, 1]), limits[:, 0])

                obs, _, terminated, truncated, _ = env.step(targets)
                step_count += 1
                episode.add("actions", targets[0].detach().cpu())
                _episode_add_obs(episode, obs)
                _episode_add_state(episode, env, obs)

                success = bool(obs["object"]["cube_in_box"][0, 0].item()) or bool(terminated[0].item())
                if args_cli.debug_controller and now - last_debug_time >= 1.0:
                    print(
                        "[DEBUG]: "
                        f"steps={step_count} "
                        f"app_running={simulation_app.is_running()} "
                        f"app_exiting={simulation_app.is_exiting()} "
                        f"command={command.tolist()} "
                        f"target={targets[0].detach().cpu().tolist()} "
                        f"success={success} "
                        f"terminated={terminated.detach().cpu().tolist()} "
                        f"truncated={truncated.detach().cpu().tolist()} "
                        f"cube_in_box={obs['object']['cube_in_box'][0].detach().cpu().tolist()}"
                    )
                    last_debug_time = now
                if success:
                    print(f"[INFO]: Success termination detected at controller step {step_count}.")
                    episode.success = True
                    episode.pre_export()
                    dataset.write_episode(episode)
                    dataset.flush()
                    print(f"[INFO]: Success episode saved to {os.path.abspath(args_cli.dataset_file)}")
                    obs, _ = env.reset()
                    targets = obs["policy"]["joint_pos_obs"].clone().to(env.unwrapped.device)
                    episode = EpisodeData()
                    episode.seed = args_cli.seed
                    episode.env_id = 0
                elif bool(truncated[0].item()):
                    print("[INFO]: Episode timed out without success; not saved.")
                    obs, _ = env.reset()
                    targets = obs["policy"]["joint_pos_obs"].clone().to(env.unwrapped.device)
                    episode = EpisodeData()
                    episode.seed = args_cli.seed
                    episode.env_id = 0
    except KeyboardInterrupt:
        print("[INFO]: Controller loop stopped by user.")
    except BaseException as exc:
        print(f"[ERROR]: Controller loop exited due to {type(exc).__name__}: {exc!r}")
        traceback.print_exc()
        raise
    finally:
        print(f"[INFO]: Controller loop cleanup after {step_count} steps.")
        controller.close()
        dataset.close()
        env.close()
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
