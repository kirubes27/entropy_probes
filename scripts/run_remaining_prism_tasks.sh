#!/usr/bin/env bash
set -euo pipefail

# Finish the remaining post-PRISM tasks after the main GPU sequence:
# 1. Resume/certify SimpleMC working-artifact backup.
# 2. Run optional SimpleMC logit lens so the negative/weak dataset has matching artifacts.
# 3. Run stated-confidence transfer for TriviaMC and PopMC.
# 4. Copy only the relevant new outputs locally.

REMOTE_HOST="${REMOTE_HOST:-root@ssh8.vast.ai}"
REMOTE_PORT="${REMOTE_PORT:-31963}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
LOCAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LOCAL_REPO"

REMOTE_REPO="${REMOTE_REPO:-/workspace/entropy_probes}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/entropy_venv/bin/python}"
MODEL="${MODEL:-meta-llama/Llama-3.3-70B-Instruct}"
MODEL_DIR="${MODEL_DIR:-Llama-3.3-70B-Instruct_4bit}"
REMOTE_MODEL_OUT="$REMOTE_REPO/outputs/$MODEL_DIR"

BASE_RUN_DIR="${BASE_RUN_DIR:-$LOCAL_REPO/downloads/prism_seq_20260503_085433}"
LOCAL_SIMPLE_RECOVERED="$BASE_RUN_DIR/simplemc_filtered_recovered"
LOCAL_REMAINING="$BASE_RUN_DIR/remaining_after_backup"
LOCAL_REMAINING_RESULTS="$LOCAL_REMAINING/results"
LOCAL_REMAINING_WORKING="$LOCAL_REMAINING/working"
LOCAL_REMAINING_LOGS="$LOCAL_REMAINING/runlogs"
STATUS_TSV="$LOCAL_REMAINING/run_status.tsv"
LOG_FILE="$LOCAL_REMAINING/remaining_queue.log"

MAIN_DATASETS="${MAIN_DATASETS:-TriviaMC_difficulty_filtered PopMC_0_difficulty_filtered}"
SIMPLE_DATASET="${SIMPLE_DATASET:-SimpleMC_difficulty_filtered}"
SEED="${SEED:-42}"
TRAIN_SPLIT="${TRAIN_SPLIT:-0.8}"
METRICS_PY="${METRICS_PY:-['logit_gap','top_logit','entropy']}"
LOGIT_LENS_LAYERS_PY="${LOGIT_LENS_LAYERS_PY:-[31,32,33,40,41,42,43,75,76,77,78]}"

RUN_SIMPLEMC_BACKUP="${RUN_SIMPLEMC_BACKUP:-1}"
RUN_SIMPLEMC_LOGIT_LENS="${RUN_SIMPLEMC_LOGIT_LENS:-1}"
RUN_STATED_CONFIDENCE="${RUN_STATED_CONFIDENCE:-1}"

mkdir -p "$LOCAL_SIMPLE_RECOVERED/results" "$LOCAL_SIMPLE_RECOVERED/runlogs" "$LOCAL_SIMPLE_RECOVERED/working"
mkdir -p "$LOCAL_REMAINING_RESULTS" "$LOCAL_REMAINING_WORKING" "$LOCAL_REMAINING_LOGS"
printf 'timestamp\tstep\tstatus\tnote\n' > "$STATUS_TSV"
: > "$LOG_FILE"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

status() {
  printf '%s\t%s\t%s\t%s\n' "$(date -Iseconds)" "$1" "$2" "${3:-}" >> "$STATUS_TSV"
}

ssh_base() {
  ssh -i "$SSH_KEY" -p "$REMOTE_PORT" -o StrictHostKeyChecking=no \
    -o ServerAliveInterval=30 -o ServerAliveCountMax=120 "$REMOTE_HOST" "$@"
}

rsync_from_remote() {
  rsync -av --partial --progress \
    -e "ssh -i $SSH_KEY -p $REMOTE_PORT -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ServerAliveCountMax=120" \
    "$@"
}

run_remote_step() {
  local step="$1"
  local cmd="$2"
  local remote_log="/workspace/runlogs/${step}.log"
  local cmd_b64

  cmd_b64="$(printf '%s' "$cmd" | base64 | tr -d '\n')"

  status "$step" "RUNNING" "$remote_log"
  log "START $step"
  log "REMOTE_LOG $remote_log"

  ssh -T -i "$SSH_KEY" -p "$REMOTE_PORT" -o StrictHostKeyChecking=no \
    -o ServerAliveInterval=30 -o ServerAliveCountMax=120 "$REMOTE_HOST" \
    bash -s -- "$REMOTE_REPO" "$remote_log" "$cmd_b64" <<'EOS'
set -euo pipefail
repo="$1"
remote_log="$2"
cmd_b64="$3"
cmd="$(printf '%s' "$cmd_b64" | base64 -d)"
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/hub
export LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" /workspace/runlogs
cd "$repo"
echo "=== Running at $(date -Iseconds) ===" > "$remote_log"
echo "$cmd" >> "$remote_log"
set +e
eval "$cmd" >> "$remote_log" 2>&1
code=$?
tail -n 120 "$remote_log" || true
exit "$code"
EOS

  rsync_from_remote "$REMOTE_HOST:$remote_log" "$LOCAL_REMAINING_LOGS/"
  status "$step" "COMPLETED" "$LOCAL_REMAINING_LOGS/$(basename "$remote_log")"
  log "DONE $step"
}

copy_matching_outputs() {
  local label="$1"
  local prefix="$2"
  local copy_working="${3:-1}"

  log "COPY outputs for $label: $prefix"
  rsync_from_remote \
    --include="${prefix}*" --exclude='*' \
    "$REMOTE_HOST:$REMOTE_MODEL_OUT/results/" "$LOCAL_REMAINING_RESULTS/" || true
  if [ "$copy_working" -eq 1 ]; then
    rsync_from_remote \
      --include="${prefix}*" --exclude='*' \
      "$REMOTE_HOST:$REMOTE_MODEL_OUT/working/" "$LOCAL_REMAINING_WORKING/" || true
  else
    log "SKIP working copy for $label"
  fi
}

copy_simplemc_outputs() {
  log "COPY SimpleMC recovered results/runlogs"
  rsync_from_remote \
    --include="${SIMPLE_DATASET}*" --include='logitlens_direction_controls_prism_next.*' --exclude='*' \
    "$REMOTE_HOST:$REMOTE_MODEL_OUT/results/" "$LOCAL_SIMPLE_RECOVERED/results/" || true
  rsync_from_remote \
    --include='*SimpleMC*' --include='11_logitlens_direction_controls_prism_next.log' --exclude='*' \
    "$REMOTE_HOST:/workspace/runlogs/" "$LOCAL_SIMPLE_RECOVERED/runlogs/" || true
}

backup_simplemc_working() {
  local step="00_backup_${SIMPLE_DATASET}_working"
  local attempt=1
  local max_attempts="${BACKUP_MAX_ATTEMPTS:-20}"
  status "$step" "RUNNING" "$LOCAL_SIMPLE_RECOVERED/working"
  log "START $step"

  while true; do
    log "BACKUP_ATTEMPT $attempt/$max_attempts"
    if rsync_from_remote \
      --include="${SIMPLE_DATASET}*" --exclude='*' \
      "$REMOTE_HOST:$REMOTE_MODEL_OUT/working/" "$LOCAL_SIMPLE_RECOVERED/working/"; then
      break
    fi

    if [ "$attempt" -ge "$max_attempts" ]; then
      status "$step" "FAILED" "rsync failed after $max_attempts attempts"
      log "FAILED $step after $max_attempts attempts"
      exit 1
    fi

    attempt=$((attempt + 1))
    log "BACKUP_RETRY in 15s"
    sleep 15
  done

  copy_simplemc_outputs
  status "$step" "COMPLETED" "$LOCAL_SIMPLE_RECOVERED/working"
  log "DONE $step"
}

simplemc_logit_lens_cmd() {
  cat <<PY
$PYTHON_BIN -u -c "import sys, analyze_directions as m; \
m.MODEL='$MODEL'; m.ADAPTER=None; m.LOAD_IN_4BIT=True; m.LOAD_IN_8BIT=False; \
m.DATASET_FILTER='$SIMPLE_DATASET'; m.TOP_K_TOKENS=20; m.LAYERS_TO_ANALYZE=$LOGIT_LENS_LAYERS_PY; \
sys.argv=['analyze_directions.py']; \
print('SIMPLEMC_LOGIT_LENS_CONFIG', {'dataset_filter':m.DATASET_FILTER,'layers':m.LAYERS_TO_ANALYZE,'top_k':m.TOP_K_TOKENS}); \
m.main()"
PY
}

stated_confidence_cmd() {
  local dataset="$1"
  cat <<PY
$PYTHON_BIN -u -c "import test_meta_transfer as m; \
m.MODEL='$MODEL'; m.DATASET='$dataset'; m.META_TASK='confidence'; m.METRICS=$METRICS_PY; \
m.PROBE_POSITIONS=['final']; m.SEED=$SEED; m.TRAIN_SPLIT=$TRAIN_SPLIT; \
m.LOAD_IN_4BIT=True; m.LOAD_IN_8BIT=False; \
m.DELEGATE_CONFDIR_TARGET='logit_margin'; \
m.FIND_CONFIDENCE_DIRECTIONS=True; m.FIND_MC_UNCERTAINTY_DIRECTIONS=True; \
m.FIND_META_OUTPUT_UNCERTAINTY_DIRECTIONS=True; m.FIND_META_MCQ_DIRECTIONS=False; \
m.SKIP_TRANSFER_TRAINING=False; m.BATCH_SIZE=8; \
print('STATED_CONFIDENCE_CONFIG', {'dataset':m.DATASET,'task':m.META_TASK,'positions':m.PROBE_POSITIONS,'metrics':m.METRICS}); \
m.main()"
PY
}

log "REMOTE=$REMOTE_HOST:$REMOTE_PORT"
log "LOCAL_REMAINING=$LOCAL_REMAINING"
log "RUN_SIMPLEMC_BACKUP=$RUN_SIMPLEMC_BACKUP RUN_SIMPLEMC_LOGIT_LENS=$RUN_SIMPLEMC_LOGIT_LENS RUN_STATED_CONFIDENCE=$RUN_STATED_CONFIDENCE"

log "Preflight remote GPU and repo"
ssh_base "cd '$REMOTE_REPO' && echo repo=\$(pwd) && nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader && test -x '$PYTHON_BIN'"

if [ "$RUN_SIMPLEMC_BACKUP" -eq 1 ]; then
  backup_simplemc_working
fi

if [ "$RUN_SIMPLEMC_LOGIT_LENS" -eq 1 ]; then
  run_remote_step "01_logit_lens_${SIMPLE_DATASET}" "$(simplemc_logit_lens_cmd)"
  copy_matching_outputs "SimpleMC logit lens" "$SIMPLE_DATASET" 0
  copy_simplemc_outputs
fi

if [ "$RUN_STATED_CONFIDENCE" -eq 1 ]; then
  for dataset in $MAIN_DATASETS; do
    run_remote_step "02_stated_confidence_${dataset}" "$(stated_confidence_cmd "$dataset")"
    copy_matching_outputs "stated confidence $dataset" "${dataset}_meta_confidence"
  done
fi

log "ALL_REMAINING_REMOTE_TASKS_DONE"
status "ALL" "COMPLETED" "$LOCAL_REMAINING"
