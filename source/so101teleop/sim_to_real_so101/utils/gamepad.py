# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Non-blocking Linux evdev gamepad input for SO-101 joint teleoperation."""

from __future__ import annotations

import os
import time

import torch


class EvdevController:
    """Read two sticks, D-pad, and shoulder/face buttons as six joint-rate commands."""

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
                print("[WARNING]: No evdev gamepad found. Actions remain at reset targets.")
        except Exception as exc:
            print(f"[WARNING]: Could not initialize evdev controller: {exc}")

    def _candidate_paths(self, print_devices: bool = False) -> list[str]:
        paths = self.list_devices()
        if print_devices and paths:
            print("[INFO]: Available evdev input devices:")
            for path in paths:
                try:
                    print(f"[INFO]:   {path}: {self.InputDevice(path).name}")
                except OSError as exc:
                    print(f"[WARNING]:   {path}: unavailable ({exc})")
        if self.device_path:
            if os.path.exists(self.device_path):
                return [self.device_path]
            print(f"[WARNING]: Requested controller device does not exist: {self.device_path}")
        return paths

    def _connect(self, print_devices: bool = False) -> None:
        self.last_scan_time = time.monotonic()
        for path in self._candidate_paths(print_devices=print_devices):
            try:
                device = self.InputDevice(path)
            except OSError:
                continue
            capabilities = device.capabilities()
            if self.ecodes.EV_ABS not in capabilities or self.ecodes.EV_KEY not in capabilities:
                continue
            try:
                device.grab()
                self.grabbed = True
            except OSError as exc:
                print(f"[WARNING]: Could not exclusively grab {device.path}: {exc}")
            os.set_blocking(device.fd, False)
            self.device = device
            self.abs_info = {
                code: info for code, info in device.capabilities(absinfo=True).get(self.ecodes.EV_ABS, [])
            }
            self.axes = {code: info.value for code, info in self.abs_info.items()}
            print(f"[INFO]: Controller device: {device.path} ({device.name})")
            return

    def close(self) -> None:
        if self.device is not None and self.grabbed:
            try:
                self.device.ungrab()
            except Exception:
                pass
        self.device = None
        self.grabbed = False

    def _axis_value(self, code: int) -> float:
        info = self.abs_info.get(code)
        if info is None:
            return 0.0
        center = (info.max + info.min) * 0.5
        radius = max((info.max - info.min) * 0.5, 1.0)
        value = (self.axes.get(code, info.value) - center) / radius
        if abs(value) < self.deadzone:
            return 0.0
        return max(-1.0, min(1.0, value))

    def poll(self) -> torch.Tensor:
        if self.device is None:
            if time.monotonic() - self.last_scan_time > 1.0:
                self._connect()
            return torch.zeros(6, dtype=torch.float32)
        try:
            events = list(self.device.read())
        except BlockingIOError:
            events = []
        except OSError as exc:
            print(f"[WARNING]: Lost controller device {self.device.path}: {exc}")
            self.close()
            return torch.zeros(6, dtype=torch.float32)
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
        # Joint 3 (elbow) remains on right-stick Y, with D-pad left/right as
        # a precise full-rate alternative for reaching across the table.
        dpad_elbow = self._axis_value(e.ABS_HAT0X)
        command[2] = dpad_elbow if dpad_elbow != 0.0 else -self._axis_value(e.ABS_RY)
        command[3] = -self._axis_value(e.ABS_HAT0Y)
        command[4] = self._axis_value(e.ABS_RX)
        command[5] = float(e.BTN_SOUTH in self.buttons or e.BTN_TR in self.buttons) - float(
            e.BTN_EAST in self.buttons or e.BTN_TL in self.buttons
        )
        return command
