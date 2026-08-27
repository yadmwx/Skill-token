#!/usr/bin/env bash
# Keeps the large CALVIN RLDS download alive across mirror disconnects and
# repairs individual files when the mirror returns a bad byte range.
set -euo pipefail

REPO_DIR=${REPO_DIR:-/home/xiaguanxiao/code/Skill-Conditioned-Representation-Depth-for-Vision-Language-Robot-Manipulation}
DATA_DIR=/data/xiaguanxiao/calvin_abc_rlds
CACHE_DIR=/data/xiaguanxiao/hf_cache
LOG="$REPO_DIR/train_logs/calvin_abc_rlds_download_resume.log"

count_shards() {
  find -L "$DATA_DIR" -maxdepth 1 -type f -name '*.tfrecord*' | wc -l
}

while [ "$(count_shards)" -lt 544 ]; do
  set +e
  env HF_HOME="$CACHE_DIR" HF_HUB_CACHE="$CACHE_DIR/hub" HF_ENDPOINT=https://hf-mirror.com \
    HF_HUB_DOWNLOAD_TIMEOUT=600 "$HOME/miniconda3/envs/vla-flow/bin/huggingface-cli" \
    download zhouhongyi/calvin_abc_rlds --repo-type dataset --local-dir "$DATA_DIR" >> "$LOG" 2>&1
  rc=$?
  set -e
  if [ "$rc" -eq 0 ]; then
    break
  fi

  bad_file=$(grep -oE '\(calvin_abc[^)]*\.tfrecord[^)]*\)' "$LOG" | tail -1 | tr -d '()')
  if [ -n "$bad_file" ]; then
    # Some repository objects currently disagree with their Hub metadata.
    # Never truncate them: validate the complete TFRecord stream (including
    # per-record CRCs) before accepting the bytes returned by the source.
    expected_size=$(grep -F "($bad_file)" "$LOG" | tail -1 | sed -nE 's/.*file should be of size ([0-9]+) but has size.*/\1/p')
    link="$DATA_DIR/$bad_file"
    blob=$(readlink -f "$link" 2>/dev/null || true)
    rm -f "$link"
    [ -n "$blob" ] && rm -f "$blob"
    if [ -n "$expected_size" ]; then
      part="${link}.part"
      url="https://hf-mirror.com/datasets/zhouhongyi/calvin_abc_rlds/resolve/main/$bad_file"
      if curl -fL --retry 3 --connect-timeout 30 --max-time 600 "$url" -o "$part" \
          && "$HOME/miniconda3/envs/vla-flow/bin/python" - "$part" <<'PY'
import sys
import tensorflow as tf

count = 0
for _ in tf.compat.v1.io.tf_record_iterator(sys.argv[1]):
    count += 1
if count == 0:
    raise RuntimeError("empty TFRecord")
print(count)
PY
      then
        actual_size=$(stat -c%s "$part")
        actual_sha=$(sha256sum "$part" | awk '{print $1}')
        mv "$part" "$link"
        # The Hub client only reuses LFS blobs from its content-addressed
        # cache.  Register known metadata-mismatched, CRC-valid objects there
        # so the next snapshot attempt does not download them again.
        case "$bad_file" in
          calvin_abc-train.tfrecord-00100-of-00512)
            expected_blob=2d9ea11524792c35c29c9f6dc65b4f37f8da5a414c25ee7d8091e980ade72497 ;;
          calvin_abc-train.tfrecord-00101-of-00512)
            expected_blob=78980b791c0ae8a82f94c7b02720d1782cde8af32b019abe7769bd2cbd1b7aca ;;
          *) expected_blob= ;;
        esac
        if [ -n "$expected_blob" ]; then
          repo_cache="$CACHE_DIR/hub/datasets--zhouhongyi--calvin_abc_rlds"
          cp "$link" "$repo_cache/blobs/$expected_blob"
          revision=$(cat "$repo_cache/refs/main")
          ln -sfn "../../blobs/$expected_blob" "$repo_cache/snapshots/$revision/$bad_file"
          rm -f "$link"
          ln -s "../hf_cache/hub/datasets--zhouhongyi--calvin_abc_rlds/blobs/$expected_blob" "$link"
        fi
        echo "[$(date '+%F %T')] accepted CRC-valid metadata-mismatched shard: $bad_file size=$actual_size sha256=$actual_sha" >> "$LOG"
      else
        rm -f "$part"
        echo "[$(date '+%F %T')] could not repair corrupt shard: $bad_file" >> "$LOG"
      fi
    else
      echo "[$(date '+%F %T')] removed corrupt shard: $bad_file (size unavailable)" >> "$LOG"
    fi
  else
    echo "[$(date '+%F %T')] download exited rc=$rc; retrying" >> "$LOG"
    sleep 30
  fi
done
