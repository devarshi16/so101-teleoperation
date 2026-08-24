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
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm

from sim_to_real_so101 import assets
from sim_to_real_so101.utils.rotations import euler_angles_to_quat
from sim_to_real_so101.mdp import (
    randomize_light_exposure,
    randomize_sky_light,
    randomize_mat_rotation,
    randomize_camera_focal_length,
    randomize_camera_pose,
    image,
    image_raw,
)

from .so101_env_cfg import (
    SO101TeleopEnvCfg,
    LerobotSo101BaseSceneCfg,
    EventCfg,
    ObservationsCfg,
)

assets_path = str(assets.require_assets())

camera_object = TiledCameraCfg(
    prim_path="",
    update_period=0.0,
    height=480,
    width=640,
    data_types=["rgb", "depth", "instance_id_segmentation_fast"],
    colorize_instance_segmentation=True,
    spawn=sim_utils.PinholeCameraCfg(
        projection_type="pinhole",
        f_stop=100,  # x10 of real
        focal_length=13.5,  # 10th of real
        focus_distance=0.05,  # 5cm in front of the camera
    ),
    offset=TiledCameraCfg.OffsetCfg(
        pos=(0.0, 0.0, 0.0),
        rot=euler_angles_to_quat(np.array([0, 0, 0]), degrees=True),
        convention="opengl",
    ),
)


@configclass
class SO101TaskSceneCfg(LerobotSo101BaseSceneCfg):
    lightstudio = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/LightStudio",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{assets_path}/usd/lightbox-simple.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.1, 0, 0.0257)),
    )
    lightbox_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/LightStudio/LightBox/RectLight"
    )

    # sky_light = AssetBaseCfg(
    #     prim_path="/World/sky_light",
    #     spawn=sim_utils.DomeLightCfg(
    #         intensity=1000.0,
    #         texture_file=f"{assets_path}/hdri/moon_lab_1k.exr",
    #         visible_in_primary_ray=False,
    #         enable_color_temperature=True,
    #         color_temperature=6500.0
    #     )
    # )

    mat = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Mat",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{assets_path}/usd/mat.usda",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.22, 0, 0.032),
            rot=euler_angles_to_quat(np.array([0, 0, 90]), degrees=True),
        ),
    )

    # Camera
    camera_ego = camera_object.replace()
    camera_ego.prim_path = "{ENV_REGEX_NS}/Robot/gripper/gripper_cam"
    camera_ego.offset.pos = (-0.005, 0.06, -0.062)
    camera_ego.offset.rot = euler_angles_to_quat(np.array([-45, 0, 0]), degrees=True)

    # Disabled for Isaac Sim 6 / IsaacLab 3 smoke test:
    # the expected external D455 camera prim is not present in lightbox-simple.usd.
    camera_external_D455 = None


@configclass
class TaskEventCfg(EventCfg):
    """Configuration for events."""

    # Disabled for Isaac Sim 6 / IsaacLab 3 smoke test: lightbox_light prim_paths is empty.
    reset_lightbox_light_exposure = None

    reset_mat_rotation = EventTerm(
        func=randomize_mat_rotation,
        mode="reset",
        params={
            "yaw_range": (-0.1, 0.1),
            "asset_cfg": SceneEntityCfg("mat"),
        },
    )

    reset_camera_ego_fov = EventTerm(
        func=randomize_camera_focal_length,
        mode="reset",
        params={
            "focal_length_range": (12.0, 15.0),  # ~±10% around 13.5mm
            "asset_cfg": SceneEntityCfg("camera_ego"),
        },
    )

    # Disabled because camera_external_D455 is disabled.
    reset_camera_external_pose = None


@configclass
class TaskObservationsCfg(ObservationsCfg):

    @configclass
    class VisualCfg(ObsGroup):

        rgb_ego = ObsTerm(
            func=image,
            params={
                "sensor_cfg": SceneEntityCfg("camera_ego"),
                "data_type": "rgb",
                "normalize": False,
            },
        )

        depth_ego = ObsTerm(
            func=image,
            params={
                "sensor_cfg": SceneEntityCfg("camera_ego"),
                "data_type": "depth",
                # "normalize": False,
            },
        )

        instance_id_seg_ego = ObsTerm(
            func=image_raw,
            params={
                "sensor_cfg": SceneEntityCfg("camera_ego"),
                "data_type": "instance_id_segmentation_fast",
            },
        )

        # Disabled because camera_external_D455 is not present in Isaac Sim 6 / IsaacLab 3.
        rgb_external_D455 = None

        # Disabled because camera_external_D455 is not present in Isaac Sim 6 / IsaacLab 3.
        depth_external_D455 = None

        # Disabled because camera_external_D455 is not present in Isaac Sim 6 / IsaacLab 3.
        instance_id_seg_external_D455 = None

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    visual: VisualCfg = VisualCfg()


@configclass
class SO101TaskEnvCfg(SO101TeleopEnvCfg):
    """Configuration for the task environment."""

    scene: SO101TaskSceneCfg = SO101TaskSceneCfg()
    events: TaskEventCfg = TaskEventCfg()
    observations: TaskObservationsCfg = TaskObservationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        super().__post_init__()

        self.sim.render.enable_translucency = True
        carb_settings = {
            "rtx.reflections.enabled": True,
            "rtx.translucency.reflectAtAllBounce": True,
            "rtx.translucency.sampleRoughness": True,
            "rtx.translucency.reflectionThroughputThreshold": 0.05,
            "rtx.translucency.maxRefractionBounces": 5,
            "rtx.raytracing.fractionalCutoutOpacity": True,
        }
        self.sim.render.carb_settings = carb_settings
