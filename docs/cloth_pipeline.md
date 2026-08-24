# SO-101 cloth pipeline

Run these commands from the repository root after activating the Isaac Lab environment and installing `source/so101teleop` in editable mode.

## Task map

| Short task ID | Isaac Gym ID | Default human demos | Default generated demos |
|---|---|---:|---:|
| `corner_lift` | `SO101-Cloth-Corner-Lift-v0` | 10 | 200 |
| `edge_drag` | `SO101-Cloth-Edge-Drag-v0` | 10 | 200 |
| `corner_fold` | `SO101-Cloth-Corner-Fold-v0` | 10 | 200 |
| `obstacle_drape_pull` | `SO101-Cloth-Obstacle-Drape-Pull-v0` | 10 | 200 |

## 1. Collect human demonstrations

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

Rerun the command with the same dataset root to resume collection.

## 2. Expand demonstrations

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

The replay pass adds bounded joint-space perturbations in randomized cloth resets and keeps only rollouts that pass the task-specific cloth-geometry success check.

## 3. Train ACT

```bash
train_cloth_act \
  --task_id corner_lift \
  --dataset_base "$(pwd)/outputs/cloth" \
  --output_base "$(pwd)/outputs/cloth_act" \
  --episodes 200 \
  --steps 100000
```

Replace `corner_lift` in all three stages with another short task ID. `train_cloth_act --task_id all` trains all four policies sequentially.
