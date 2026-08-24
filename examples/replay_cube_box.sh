#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
output_file="${1:-$repo_root/outputs/cube_box/example_replay.mp4}"

render_hdf5_replay \
  --dataset_file "$repo_root/examples/data/so101_cube_box_controller.hdf5" \
  --episode 0 \
  --output "$output_file" \
  --start_frame 443 \
  --frame_stride 20 \
  --fps 20 \
  --width 960 \
  --height 540 \
  --device cuda:0 \
  --viz none
