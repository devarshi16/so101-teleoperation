# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train one local-GPU LeRobot ACT model per SO-101 cloth task."""

from __future__ import annotations

import argparse
from pathlib import Path

from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.scripts.lerobot_train import train as lerobot_train


TASK_IDS = ("corner_lift", "edge_drag", "corner_fold", "obstacle_drape_pull")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task_id", choices=(*TASK_IDS, "all"), default="all")
parser.add_argument("--dataset_base", type=Path, default=None)
parser.add_argument("--output_base", type=Path, default=None)
parser.add_argument("--episodes", type=int, default=200)
parser.add_argument("--steps", type=int, default=100_000)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--seed", type=int, default=2026)
parser.add_argument("--dry_run", action="store_true")
args = parser.parse_args()

WORKSHOP_ROOT = Path(__file__).resolve().parents[4]
dataset_base = (args.dataset_base or WORKSHOP_ROOT / "datasets" / "cloth").resolve()
output_base = (args.output_base or WORKSHOP_ROOT / "outputs" / "cloth_act").resolve()


def train(task_id: str) -> None:
    dataset_root = dataset_base / task_id / "mimic_200"
    repo_id = f"{task_id}/mimic_200"
    dataset = LeRobotDataset(repo_id, root=dataset_root)
    if dataset.num_episodes != args.episodes:
        raise ValueError(f"{task_id}: expected {args.episodes} episodes, found {dataset.num_episodes}")
    output_dir = output_base / task_id
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=repo_id, root=str(dataset_root)),
        policy=ACTConfig(device="cuda", push_to_hub=False),
        output_dir=output_dir,
        job_name=f"so101_cloth_{task_id}_act",
        seed=args.seed,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        steps=args.steps,
        eval_freq=0,
        log_freq=200,
        save_checkpoint=True,
        save_freq=10_000,
        wandb=WandBConfig(enable=False),
    )
    print(f"[TRAIN] task={task_id} dataset={dataset_root} output={output_dir} steps={args.steps}")
    if not args.dry_run:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        lerobot_train(cfg)


def main() -> int:
    selected = TASK_IDS if args.task_id == "all" else (args.task_id,)
    for task_id in selected:
        train(task_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
