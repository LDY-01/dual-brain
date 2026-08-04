#!/usr/bin/env bash
# Ten independent ACT-v2 evaluations. A home return follows only a clean rollout exit.
set -euo pipefail

ROOT="/Users/doyounglim/Desktop/SO101"
PORT="/dev/tty.usbmodem5B140307781"
DATASET_ROOT="$ROOT/lerobot_data/rollout_act_lens_cap_v2_home_mac"
DATASET_REPO_ID="DY-01/rollout_act_lens_cap_v2_home_mac"
MODEL="$ROOT/models/act_lens_cap_v2"

if [[ -e "$DATASET_ROOT" ]]; then
  echo "Refusing to reuse an existing evaluation folder: $DATASET_ROOT"
  echo "Rename or remove it deliberately before starting a new 10-episode evaluation."
  exit 1
fi

echo "Moving to the fixed task-start home pose before episode 1. Watch the arm."
(cd "$ROOT" && uv run python scripts/return_home.py --execute --duration 12)

for episode in $(seq 1 10); do
  echo
  echo "========== Episode $episode / 10 =========="
  read -r -p "Place the box and cap at their fixed start positions, then press Enter. "

  rollout_args=(
    lerobot-rollout
    --strategy.type=episodic
    --policy.path="$MODEL"
    --device=cpu
    --robot.type=so101_follower
    --robot.port="$PORT"
    --robot.id=so101_follower
    --robot.max_relative_target=5
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}"
    --task="Place the lens cap into the open box."
    --dataset.repo_id="$DATASET_REPO_ID"
    --dataset.root="$DATASET_ROOT"
    --dataset.single_task="Place the lens cap into the open box."
    --dataset.fps=30
    --dataset.num_episodes=1
    --dataset.episode_time_s=30
    --dataset.reset_time_s=0
    --dataset.video=true
    --dataset.push_to_hub=false
    --dataset.streaming_encoding=false
    --dataset.encoder_threads=1
    --strategy.reset_to_initial_position=false
    --display_data=false
  )

  if (( episode > 1 )); then
    rollout_args+=(--resume=true)
  fi

  # With set -e, a crashed/interrupted rollout stops here: no automatic home motion.
  (cd "$ROOT" && uv run "${rollout_args[@]}")
  (cd "$ROOT" && uv run python scripts/return_home.py --execute --duration 8)
done

echo "Completed 10 episodes. Review videos in: $DATASET_ROOT"
