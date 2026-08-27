#!/usr/bin/env bash
# Keep the system disk safe during the long CALVIN pure-DiT paper run.
#
# The trainer writes a complete checkpoint every 5k optimizer steps.  This
# watcher moves only *completed intermediate* CALVIN pure-DiT checkpoints to
# the large data disk.  It deliberately leaves the final checkpoint in
# outputs/, because queue_calvin_pure_dit_paper.sh discovers it there before
# launching the official evaluation.
set -euo pipefail

REPO_DIR=${REPO_DIR:-/home/xiaguanxiao/code/Skill-Conditioned-Representation-Depth-for-Vision-Language-Robot-Manipulation}
OUTPUT_DIR="$REPO_DIR/outputs"
ARCHIVE_DIR=${ARCHIVE_DIR:-/data/xiaguanxiao/archives/calvin_pure_dit_intermediates}
FINAL_STEP=${FINAL_STEP:-400005}
POLL_SECONDS=${POLL_SECONDS:-120}

mkdir -p "$ARCHIVE_DIR"

is_complete_checkpoint() {
  local checkpoint=$1
  compgen -G "$checkpoint/training_state--*_checkpoint.pt" >/dev/null
}

archive_once() {
  local checkpoint base step target
  shopt -s nullglob
  for checkpoint in "$OUTPUT_DIR"/*CALVIN-ABC-pureDIT-*-skilltoken-*--*_chkpt; do
    base=$(basename "$checkpoint")
    if [[ ! "$base" =~ --([0-9]+)_chkpt$ ]]; then
      continue
    fi
    step=${BASH_REMATCH[1]}
    # Saving finishes by writing training_state last.  Never touch the final
    # model: the queue expects that directory to remain in outputs/.
    if (( step >= FINAL_STEP )) || ! is_complete_checkpoint "$checkpoint"; then
      continue
    fi
    target="$ARCHIVE_DIR/$base"
    if [[ -e "$target" ]]; then
      echo "[$(date '+%F %T')] archive target already exists; leaving source: $checkpoint" >&2
      continue
    fi
    echo "[$(date '+%F %T')] archiving completed intermediate checkpoint: $base"
    mv -- "$checkpoint" "$ARCHIVE_DIR/"
  done
}

while true; do
  archive_once
  sleep "$POLL_SECONDS"
done
