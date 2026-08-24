# SO-101 Teleoperation for Isaac Lab

SO-101 teleoperation, demonstration collection, and deformable-cloth tasks for NVIDIA Isaac Lab. The project supports:

- a physical SO-101 leader arm driving a simulated SO-101;
- Linux `evdev` gamepad teleoperation;
- resumable LeRobot dataset collection;
- a cube-to-box controller task;
- four cloth-manipulation tasks, Mimic-style replay expansion, and per-task ACT training.

The Python code is stored in this repository. The large robot USD files, scene assets, textures, and HDRIs are **not duplicated**: `third_party/Sim-to-Real-SO-101-Workshop` is a Git submodule pinned to NVIDIA's original [Sim-to-Real SO-101 Workshop](https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop).

## Tested configuration

This checkout was validated with the following local stack:

- Ubuntu Linux with an NVIDIA CUDA GPU;
- Python 3.12;
- Isaac Sim 6.0;
- Isaac Lab `release/3.0.0-beta2` at commit `e722b5b245`;
- LeRobot 0.4.3;
- a USB SO-101 leader arm or a Linux gamepad exposed through `/dev/input`.

Other versions may work, but Isaac Sim and Isaac Lab APIs change frequently. Start with the versions above when reproducing the setup.

## Installation

### 1. Install Isaac Sim and Isaac Lab

Install Isaac Sim 6.0, create/activate a Python 3.12 environment for it, then install the tested Isaac Lab revision:

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout e722b5b245
./isaaclab.sh --install
```

Follow the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) if `isaaclab.sh` cannot find Isaac Sim. All commands below must run in the Python environment in which `import isaaclab` and `import isaaclab_tasks` succeed.

Install the tested LeRobot release in the same environment:

```bash
python -m pip install "lerobot==0.4.3"
```

### 2. Clone this repository with its asset submodule

```bash
git clone --recurse-submodules https://github.com/devarshi16/so101-teleoperation.git
cd so101-teleoperation
```

If the repository was cloned without `--recurse-submodules`, initialize the asset repository now:

```bash
git submodule update --init --recursive
```

The expected robot asset will then be at:

```text
third_party/Sim-to-Real-SO-101-Workshop/source/sim_to_real_so101/assets/usd/SO-ARM101-USD.usd
```

### 3. Install the teleoperation package

From this repository's root, in the Isaac Lab environment:

```bash
python -m pip install -e source/so101teleop
```

Verify the package and submodule without launching a full task:

```bash
python -c "from sim_to_real_so101.assets import require_assets; print(require_assets())"
list_envs
```

`list_envs` launches Isaac Sim briefly and prints every registered task. If the assets live outside this checkout (for example, in a container mount), set `SO101_ASSETS_ROOT` to the upstream `assets` directory.

## Device setup

### Physical SO-101 leader arm

Find the serial device and give your user serial-port access:

```bash
ls -l /dev/ttyACM* /dev/ttyUSB* 2>/dev/null
sudo usermod -aG dialout "$USER"
```

Log out and back in after changing group membership. The examples below use `/dev/ttyACM0` and the calibration ID `leader_arm_1`; replace them with the port and ID used when your LeRobot leader arm was calibrated.

### Linux gamepad

List stable gamepad device paths:

```bash
ls -l /dev/input/by-id/*-event-joystick 2>/dev/null
```

Pass one of those paths through `--controller_device`. If no path is supplied, the code scans available `evdev` devices. Your user must have permission to read the selected `/dev/input/event*` node.

## Start a teleoperation task

### Physical leader arm: create a new LeRobot dataset

The following values define the simulation task, dataset identity, task instruction, and local output directory. A new dataset is created when `OUTPUT_DIR` does not exist; rerunning the same command resumes a valid existing dataset.

```bash
TASK_ID="Lerobot-So101-Teleop-Vials-To-Rack"
DATASET_ID="devarshi16/so101-vials-to-rack-demo"
TASK_INSTRUCTION="Pick up a vial and place it in the yellow rack"
OUTPUT_DIR="$(pwd)/outputs/vials_to_rack/run_001"

lerobot_agent \
  --task "$TASK_ID" \
  --port /dev/ttyACM0 \
  --robot_id leader_arm_1 \
  --repo_id "$DATASET_ID" \
  --repo_root "$OUTPUT_DIR" \
  --task_name "$TASK_INSTRUCTION" \
  --device cuda:0 \
  --viz kit
```

Recording controls in the Isaac Sim window:

- `S`: start recording; press again to save the episode.
- `C`: cancel the current episode without saving it.
- `R`: save the current episode if recording, then reset the scene.
- `Ctrl+C` in the terminal: stop the program cleanly.

To teleoperate without recording, omit `--repo_id`, `--repo_root`, and `--task_name`:

```bash
lerobot_agent \
  --task Lerobot-So101-Teleop-Base \
  --port /dev/ttyACM0 \
  --robot_id leader_arm_1 \
  --device cuda:0 \
  --viz kit
```

### Gamepad: cube-to-box task

`controller_agent` creates the HDF5 file and its parent directories when needed. Only successful episodes are appended.

```bash
controller_agent \
  --task Lerobot-So101-Controller-Cube-Box \
  --dataset_file "$(pwd)/outputs/cube_box/run_001/episodes.hdf5" \
  --controller_device /dev/input/by-id/YOUR-GAMEPAD-event-joystick \
  --device cuda:0 \
  --viz kit
```

Use `--debug_controller` to print the input and joint targets once per second.

### Gamepad: cloth demonstration collection

Choose a short task ID and an output directory. The corresponding Isaac Gym environment ID is selected automatically.

```bash
cloth_controller_collect \
  --task_id corner_lift \
  --episodes 10 \
  --dataset_root "$(pwd)/outputs/cloth/corner_lift/human_10" \
  --repo_id "devarshi16/so101-cloth-corner-lift-human-10" \
  --controller_device /dev/input/by-id/YOUR-GAMEPAD-event-joystick \
  --device cuda:0 \
  --viz kit
```

Cloth recording controls:

- `S`: start a take; press again to save manually.
- `C`: cancel the current take.
- `R`: cancel the current take and reset the robot and cloth.
- Left stick X/Y: shoulder pan/lift.
- Right stick Y/X: elbow/wrist roll.
- D-pad left/right: precise elbow control.
- D-pad up/down: wrist pitch.
- A or right bumper: close gripper.
- B or left bumper: open gripper.

A confirmed task success saves and resets automatically. Failed takes time out and are discarded. Collection is resumable: rerun the same command to continue until `--episodes` is reached.

## Task IDs

| Purpose | Command/task ID |
|---|---|
| Leader-arm debug | `Lerobot-So101-Teleop-Base` |
| Lightbox/camera debug | `Lerobot-So101-Teleop-Task` |
| Vials to rack | `Lerobot-So101-Teleop-Vials-To-Rack` |
| Vials to rack + domain randomization | `Lerobot-So101-Teleop-Vials-To-Rack-DR` |
| Vials evaluation | `Lerobot-So101-Teleop-Vials-To-Rack-Eval` |
| Vials evaluation + domain randomization | `Lerobot-So101-Teleop-Vials-To-Rack-DR-Eval` |
| Gamepad cube to box | `Lerobot-So101-Controller-Cube-Box` |
| Cloth corner lift | `corner_lift` -> `SO101-Cloth-Corner-Lift-v0` |
| Cloth edge drag | `edge_drag` -> `SO101-Cloth-Edge-Drag-v0` |
| Cloth corner fold | `corner_fold` -> `SO101-Cloth-Corner-Fold-v0` |
| Cloth obstacle drape/pull | `obstacle_drape_pull` -> `SO101-Cloth-Obstacle-Drape-Pull-v0` |

## Expand cloth data and train ACT

Expand 10 recorded demonstrations into 200 accepted simulation replays:

```bash
cloth_mimic_expand \
  --task_id corner_lift \
  --source_root "$(pwd)/outputs/cloth/corner_lift/human_10" \
  --output_root "$(pwd)/outputs/cloth/corner_lift/mimic_200" \
  --source_repo_id "devarshi16/so101-cloth-corner-lift-human-10" \
  --output_repo_id "devarshi16/so101-cloth-corner-lift-mimic-200" \
  --source_episodes 10 \
  --target_episodes 200 \
  --device cuda:0
```

Train one ACT model:

```bash
train_cloth_act \
  --task_id corner_lift \
  --dataset_base "$(pwd)/outputs/cloth" \
  --output_base "$(pwd)/outputs/cloth_act" \
  --episodes 200 \
  --steps 100000
```

Use `--task_id all` to train all four cloth tasks sequentially. See [docs/cloth_pipeline.md](docs/cloth_pipeline.md) for the compact end-to-end workflow.

## Repository layout

```text
source/so101teleop/                 editable Python package
  sim_to_real_so101/tasks/          Gym/Isaac Lab task configurations
  sim_to_real_so101/scripts/        teleop, collection, expansion, training
  sim_to_real_so101/utils/          recorder, keyboard, gamepad interfaces
third_party/Sim-to-Real-SO-101-Workshop/  pinned upstream asset submodule
docs/cloth_pipeline.md              cloth workflow reference
```

Generated datasets, videos, model checkpoints, logs, caches, and HDF5 files are ignored by Git.

## Updating the upstream assets

The submodule is intentionally pinned for reproducibility. To update it deliberately:

```bash
git -C third_party/Sim-to-Real-SO-101-Workshop fetch origin
git -C third_party/Sim-to-Real-SO-101-Workshop checkout origin/main
git add third_party/Sim-to-Real-SO-101-Workshop
git commit -m "Update upstream SO-101 workshop assets"
```

Review the upstream asset paths before committing an update, because renamed USD or HDRI files can break task configuration.

## License and attribution

This project contains code derived from NVIDIA's Sim-to-Real SO-101 Workshop and Isaac Lab. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES). Large upstream assets remain in their original repository and are referenced through the Git submodule.
