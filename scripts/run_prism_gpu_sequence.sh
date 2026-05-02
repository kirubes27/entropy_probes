#!/usr/bin/env bash
set -euo pipefail

# Representation-first GPU sequence for the ICML/PRISM push.
#
# This local orchestrator runs commands on a remote GPU box via SSH, snapshots
# outputs after every step, and avoids editing the hard-coded experiment scripts
# by overriding module globals before calling main().
#
# Required:
#   REMOTE_HOST=root@host REMOTE_PORT=12345 bash scripts/run_prism_gpu_sequence.sh --preflight-only
#   REMOTE_HOST=root@host REMOTE_PORT=12345 bash scripts/run_prism_gpu_sequence.sh
#
# Common optional env vars:
#   REMOTE_REPO=/workspace/entropy_probes
#   PYTHON_BIN=/workspace/entropy_venv/bin/python
#   BRANCH=codex/clean-rerun
#   FRESH_OUTPUTS=1
#   RUN_SIMPLEMC=1
#   RUN_STATED_CONFIDENCE=1
#   RUN_OTHER_CONFIDENCE=0
#   RUN_OPTIONS_STEERING=0

REMOTE_HOST="${REMOTE_HOST:-}"
REMOTE_PORT="${REMOTE_PORT:-}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
LOCAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LOCAL_REPO"

REMOTE_REPO="${REMOTE_REPO:-/workspace/entropy_probes}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/entropy_venv/bin/python}"
BRANCH="${BRANCH:-codex/clean-rerun}"
GIT_REMOTE="${GIT_REMOTE:-auto}"

MODEL="${MODEL:-meta-llama/Llama-3.3-70B-Instruct}"
MODEL_DIR="${MODEL_DIR:-Llama-3.3-70B-Instruct_4bit}"
MAIN_DATASETS="${MAIN_DATASETS:-TriviaMC_difficulty_filtered PopMC_0_difficulty_filtered}"
SIMPLE_DATASET="${SIMPLE_DATASET:-SimpleMC}"

NUM_QUESTIONS="${NUM_QUESTIONS:-500}"
SEED="${SEED:-42}"
TRAIN_SPLIT="${TRAIN_SPLIT:-0.8}"
METRICS_PY="${METRICS_PY:-['logit_gap','top_logit','entropy']}"
POSITIONS_PY="${POSITIONS_PY:-['question_mark','question_newline','options_newline','final']}"
LOGIT_LENS_LAYERS_PY="${LOGIT_LENS_LAYERS_PY:-[31,32,33,40,41,42,43,75,76,77,78]}"
LOGIT_LENS_METRICS="${LOGIT_LENS_METRICS:-all}"

FRESH_OUTPUTS="${FRESH_OUTPUTS:-1}"
RUN_SIMPLEMC="${RUN_SIMPLEMC:-1}"
RUN_STATED_CONFIDENCE="${RUN_STATED_CONFIDENCE:-1}"
RUN_OTHER_CONFIDENCE="${RUN_OTHER_CONFIDENCE:-0}"
RUN_OPTIONS_STEERING="${RUN_OPTIONS_STEERING:-0}"

MIN_LOCAL_FREE_GB="${MIN_LOCAL_FREE_GB:-120}"
MIN_REMOTE_FREE_GB="${MIN_REMOTE_FREE_GB:-160}"

REMOTE_MODEL_OUT="$REMOTE_REPO/outputs/$MODEL_DIR"
RUN_ID="prism_seq_$(date +%Y%m%d_%H%M%S)"
LOCAL_BASE="$LOCAL_REPO/downloads/$RUN_ID"
LOCAL_MIRROR="$LOCAL_BASE/mirror"
LOG_FILE="$LOCAL_BASE/orchestrator.log"
STATUS_TSV="$LOCAL_BASE/run_status.tsv"
USE_RSYNC=0
PREFLIGHT_ONLY=0
CURRENT_STEP=""

if [ "${1:-}" = "--preflight-only" ]; then
  PREFLIGHT_ONLY=1
fi

if [ -z "$REMOTE_HOST" ] || [ -z "$REMOTE_PORT" ]; then
  echo "ERROR: Set REMOTE_HOST and REMOTE_PORT." >&2
  echo "Example: REMOTE_HOST=root@ssh.vast.ai REMOTE_PORT=12345 bash scripts/run_prism_gpu_sequence.sh --preflight-only" >&2
  exit 1
fi

mkdir -p "$LOCAL_BASE/logs" "$LOCAL_BASE/snapshots" "$LOCAL_MIRROR/results" "$LOCAL_MIRROR/working"
printf 'timestamp\tstep\tstatus\tnote\n' > "$STATUS_TSV"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

record_status() {
  local step="$1"
  local status="$2"
  local note="${3:-}"
  printf '%s\t%s\t%s\t%s\n' "$(date -Iseconds)" "$step" "$status" "$note" >> "$STATUS_TSV"
}

trap 'code=$?; if [ "$code" -ne 0 ] && [ -n "${CURRENT_STEP:-}" ]; then record_status "$CURRENT_STEP" "FAILED" "exit_code=$code"; fi' EXIT

ssh_cmd() {
  ssh -i "$SSH_KEY" -p "$REMOTE_PORT" -o StrictHostKeyChecking=no "$REMOTE_HOST" "$@"
}

run_step() {
  local step="$1"
  local cmd="$2"
  local cmd_b64
  local remote_log="/workspace/runlogs/${step}_${RUN_ID}.log"
  local local_step_dir="$LOCAL_BASE/snapshots/${step}"

  cmd_b64="$(printf '%s' "$cmd" | base64 | tr -d '\n')"

  CURRENT_STEP="$step"
  record_status "$step" "RUNNING" ""
  log "START $step"
  log "CMD $cmd"

  ssh -i "$SSH_KEY" -p "$REMOTE_PORT" -o StrictHostKeyChecking=no "$REMOTE_HOST" \
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
echo "=== Running at $(date -Iseconds) ==="
echo "$cmd"
eval "$cmd" 2>&1 | tee "$remote_log"
EOS

  log "DONE $step"
  mkdir -p "$local_step_dir"

  scp -i "$SSH_KEY" -P "$REMOTE_PORT" -o StrictHostKeyChecking=no \
    "$REMOTE_HOST:$remote_log" "$LOCAL_BASE/logs/"

  if [ "$USE_RSYNC" -eq 1 ]; then
    rsync -az --delete -e "ssh -i $SSH_KEY -p $REMOTE_PORT -o StrictHostKeyChecking=no" \
      "$REMOTE_HOST:$REMOTE_MODEL_OUT/results/" "$LOCAL_MIRROR/results/"
    rsync -az --delete -e "ssh -i $SSH_KEY -p $REMOTE_PORT -o StrictHostKeyChecking=no" \
      "$REMOTE_HOST:$REMOTE_MODEL_OUT/working/" "$LOCAL_MIRROR/working/"
    cp -al "$LOCAL_MIRROR/results" "$local_step_dir/"
    cp -al "$LOCAL_MIRROR/working" "$local_step_dir/"
  else
    scp -i "$SSH_KEY" -P "$REMOTE_PORT" -o StrictHostKeyChecking=no -r \
      "$REMOTE_HOST:$REMOTE_MODEL_OUT/results" "$local_step_dir/" || true
    scp -i "$SSH_KEY" -P "$REMOTE_PORT" -o StrictHostKeyChecking=no -r \
      "$REMOTE_HOST:$REMOTE_MODEL_OUT/working" "$local_step_dir/" || true
  fi

  log "BACKUP_COMPLETE $step -> $local_step_dir"
  record_status "$step" "COMPLETED" "$local_step_dir"
  CURRENT_STEP=""
}

py_stage0_cmd() {
  local dataset="$1"
  cat <<PY
$PYTHON_BIN -u -c "import identify_mc_correlate as m; \
m.MODEL='$MODEL'; m.DATASET='$dataset'; m.METRICS=$METRICS_PY; \
m.NUM_QUESTIONS=$NUM_QUESTIONS; m.SEED=$SEED; m.TRAIN_SPLIT=$TRAIN_SPLIT; \
m.DIRECTION_N_JOBS=1; \
m.LOAD_IN_4BIT=True; m.LOAD_IN_8BIT=False; m.FIND_ANSWER_DIRECTIONS=True; \
print('STAGE0_CONFIG', {'dataset':m.DATASET,'metrics':m.METRICS,'num_questions':m.NUM_QUESTIONS,'seed':m.SEED,'direction_n_jobs':m.DIRECTION_N_JOBS}); \
m.main()"
PY
}

py_transfer_cmd() {
  local dataset="$1"
  local task="$2"
  local positions="$3"
  local find_meta_mcq="$4"
  cat <<PY
$PYTHON_BIN -u -c "import test_meta_transfer as m; \
m.MODEL='$MODEL'; m.DATASET='$dataset'; m.META_TASK='$task'; m.METRICS=$METRICS_PY; \
m.PROBE_POSITIONS=$positions; \
m.SEED=$SEED; m.TRAIN_SPLIT=$TRAIN_SPLIT; m.LOAD_IN_4BIT=True; m.LOAD_IN_8BIT=False; \
m.DELEGATE_CONFDIR_TARGET='logit_margin'; \
m.FIND_CONFIDENCE_DIRECTIONS=True; m.FIND_MC_UNCERTAINTY_DIRECTIONS=True; \
m.FIND_META_OUTPUT_UNCERTAINTY_DIRECTIONS=True; m.FIND_META_MCQ_DIRECTIONS=$find_meta_mcq; \
m.SKIP_TRANSFER_TRAINING=False; m.BATCH_SIZE=4 if m.META_TASK == 'delegate' else 8; \
print('TRANSFER_CONFIG', {'dataset':m.DATASET,'task':m.META_TASK,'positions':m.PROBE_POSITIONS,'confdir':m.FIND_CONFIDENCE_DIRECTIONS,'mcuncert':m.FIND_MC_UNCERTAINTY_DIRECTIONS,'meta_output_uncert':m.FIND_META_OUTPUT_UNCERTAINTY_DIRECTIONS,'metamcq':m.FIND_META_MCQ_DIRECTIONS}); \
m.main()"
PY
}

py_cosine_cmd() {
  cat <<PY
$PYTHON_BIN -u -c "import compare_direction_similarity as m; \
m.MODEL='$MODEL'; m.ADAPTER=None; m.LOAD_IN_4BIT=True; m.LOAD_IN_8BIT=False; \
m.METRICS=['logit_gap','top_logit','entropy']; m.METHODS=['probe','mean_diff']; \
m.N_PERMUTATIONS=100; m.SEED=$SEED; \
print('COSINE_CONFIG', {'model':m.MODEL,'metrics':m.METRICS,'methods':m.METHODS,'n_perm':m.N_PERMUTATIONS}); \
m.main()"
PY
}

py_logit_lens_cmd() {
  local dataset="$1"
  local metric="$2"
  local argv_expr
  if [ "$metric" = "all" ]; then
    argv_expr="['analyze_directions.py']"
  else
    argv_expr="['analyze_directions.py','--metric','$metric']"
  fi
  cat <<PY
$PYTHON_BIN -u -c "import sys, analyze_directions as m; \
m.MODEL='$MODEL'; m.ADAPTER=None; m.LOAD_IN_4BIT=True; m.LOAD_IN_8BIT=False; \
m.DATASET_FILTER='$dataset'; m.TOP_K_TOKENS=20; m.LAYERS_TO_ANALYZE=$LOGIT_LENS_LAYERS_PY; \
sys.argv=$argv_expr; \
print('LOGIT_LENS_CONFIG', {'dataset_filter':m.DATASET_FILTER,'metric':'$metric','layers':m.LAYERS_TO_ANALYZE,'top_k':m.TOP_K_TOKENS}); \
m.main()"
PY
}

py_options_steering_cmd() {
  local dataset="$1"
  cat <<PY
$PYTHON_BIN -u -c "import run_steering_causality as m; \
m.MODEL='$MODEL'; m.DATASET='$dataset'; m.META_TASK='delegate'; \
m.DIRECTION_TYPE='uncertainty'; m.METRIC='logit_gap'; m.CONFIDENCE_SIGNAL='logit_margin'; \
m.SEED=$SEED; m.TRAIN_SPLIT=$TRAIN_SPLIT; m.USE_TRANSFER_SPLIT=True; \
m.LOAD_IN_4BIT=True; m.LOAD_IN_8BIT=False; m.METHODS=['probe','mean_diff']; \
m.PROBE_POSITIONS=['options_newline']; m.LAYERS=None; m.NUM_CONTROLS_NONFINAL=10; \
print('OPTIONS_STEERING_CONFIG', {'dataset':m.DATASET,'position':m.PROBE_POSITIONS,'methods':m.METHODS,'layers':'transfer-threshold','signal':m.CONFIDENCE_SIGNAL}); \
m.main()"
PY
}

log "RUN_ID=$RUN_ID"
log "LOCAL_BASE=$LOCAL_BASE"
log "REMOTE=$REMOTE_HOST:$REMOTE_PORT"
log "REMOTE_REPO=$REMOTE_REPO"
log "MODEL=$MODEL MODEL_DIR=$MODEL_DIR"

local_free_kb="$(df -Pk "$LOCAL_BASE" | awk 'NR==2 {print $4}')"
local_free_gb="$(( local_free_kb / 1024 / 1024 ))"
log "Preflight: local free space = ${local_free_gb}GB"
if [ "$local_free_gb" -lt "$MIN_LOCAL_FREE_GB" ]; then
  log "ERROR: local free space ${local_free_gb}GB < required ${MIN_LOCAL_FREE_GB}GB"
  exit 1
fi

if command -v rsync >/dev/null 2>&1 && ssh_cmd "command -v rsync >/dev/null 2>&1"; then
  USE_RSYNC=1
  log "Backup mode: rsync + hardlink snapshots"
else
  USE_RSYNC=0
  log "Backup mode: scp snapshots"
fi

log "Preflight: syncing remote branch"
if [ "$GIT_REMOTE" = "auto" ]; then
  SELECTED_REMOTE="$(ssh_cmd "cd $REMOTE_REPO && (git ls-remote --exit-code --heads fork $BRANCH >/dev/null 2>&1 && echo fork) || (git ls-remote --exit-code --heads origin $BRANCH >/dev/null 2>&1 && echo origin) || true")"
else
  SELECTED_REMOTE="$GIT_REMOTE"
fi
if [ -z "${SELECTED_REMOTE:-}" ]; then
  log "ERROR: Could not find branch '$BRANCH' on remote 'fork' or 'origin'."
  exit 1
fi
ssh_cmd "cd $REMOTE_REPO && git fetch $SELECTED_REMOTE && git checkout $BRANCH && git pull --ff-only $SELECTED_REMOTE $BRANCH"

log "Preflight: validating Python, GPU, model dir, and HF token"
ssh_cmd "cd $REMOTE_REPO && HF_HOME=/workspace/.cache/huggingface TRANSFORMERS_CACHE=/workspace/.cache/huggingface/hub $PYTHON_BIN - <<'PY'
import torch
from huggingface_hub import get_token
from core.model_utils import get_model_dir_name
model_dir = get_model_dir_name('$MODEL', None, True, False)
print('CUDA_AVAILABLE', torch.cuda.is_available())
print('CUDA_DEVICE_COUNT', torch.cuda.device_count())
if torch.cuda.is_available():
    print('GPU_NAME', torch.cuda.get_device_name(0))
print('MODEL_DIR', model_dir)
print('HF_TOKEN_SET', bool(get_token()))
assert torch.cuda.is_available(), 'CUDA not available'
assert model_dir == '$MODEL_DIR', (model_dir, '$MODEL_DIR')
assert get_token(), 'HF token missing'
PY"

remote_free_kb="$(ssh_cmd "df -Pk /workspace | tail -n 1 | tr -s ' ' | cut -d ' ' -f 4" | tail -n 1)"
remote_free_gb="$(( remote_free_kb / 1024 / 1024 ))"
log "Preflight: remote free space = ${remote_free_gb}GB"
if [ "$remote_free_gb" -lt "$MIN_REMOTE_FREE_GB" ]; then
  log "ERROR: remote free space ${remote_free_gb}GB < required ${MIN_REMOTE_FREE_GB}GB"
  exit 1
fi

if [ "$PREFLIGHT_ONLY" -eq 1 ]; then
  log "PREFLIGHT_ONLY=1: preflight passed."
  exit 0
fi

if [ "$FRESH_OUTPUTS" -eq 1 ]; then
  log "Archiving existing remote outputs for a clean run"
  ssh_cmd "if [ -d '$REMOTE_MODEL_OUT' ]; then mv '$REMOTE_MODEL_OUT' '${REMOTE_MODEL_OUT}_pre_${RUN_ID}'; fi; mkdir -p '$REMOTE_MODEL_OUT/results' '$REMOTE_MODEL_OUT/working'"
else
  log "FRESH_OUTPUTS=0: preserving existing remote outputs/caches"
  ssh_cmd "mkdir -p '$REMOTE_MODEL_OUT/results' '$REMOTE_MODEL_OUT/working'"
fi

step_i=1
for dataset in $MAIN_DATASETS; do
  run_step "$(printf '%02d_stage0_%s' "$step_i" "$dataset")" "$(py_stage0_cmd "$dataset")"
  step_i=$((step_i + 1))
  run_step "$(printf '%02d_delegate_transfer_%s' "$step_i" "$dataset")" "$(py_transfer_cmd "$dataset" "delegate" "$POSITIONS_PY" "True")"
  step_i=$((step_i + 1))
done

run_step "$(printf '%02d_cross_dataset_cosine' "$step_i")" "$(py_cosine_cmd)"
step_i=$((step_i + 1))

for dataset in $MAIN_DATASETS; do
  for metric in $LOGIT_LENS_METRICS; do
    run_step "$(printf '%02d_logit_lens_%s_%s' "$step_i" "$dataset" "$metric")" "$(py_logit_lens_cmd "$dataset" "$metric")"
    step_i=$((step_i + 1))
  done
done

if [ "$RUN_SIMPLEMC" -eq 1 ]; then
  run_step "$(printf '%02d_stage0_%s' "$step_i" "$SIMPLE_DATASET")" "$(py_stage0_cmd "$SIMPLE_DATASET")"
  step_i=$((step_i + 1))
  run_step "$(printf '%02d_delegate_transfer_%s' "$step_i" "$SIMPLE_DATASET")" "$(py_transfer_cmd "$SIMPLE_DATASET" "delegate" "$POSITIONS_PY" "True")"
  step_i=$((step_i + 1))
fi

if [ "$RUN_STATED_CONFIDENCE" -eq 1 ]; then
  for dataset in $MAIN_DATASETS; do
    run_step "$(printf '%02d_stated_confidence_%s' "$step_i" "$dataset")" "$(py_transfer_cmd "$dataset" "confidence" "['final']" "False")"
    step_i=$((step_i + 1))
  done
fi

if [ "$RUN_OTHER_CONFIDENCE" -eq 1 ]; then
  for dataset in $MAIN_DATASETS; do
    run_step "$(printf '%02d_other_confidence_%s' "$step_i" "$dataset")" "$(py_transfer_cmd "$dataset" "other_confidence" "['final']" "False")"
    step_i=$((step_i + 1))
  done
fi

if [ "$RUN_OPTIONS_STEERING" -eq 1 ]; then
  for dataset in $MAIN_DATASETS; do
    run_step "$(printf '%02d_options_newline_steering_%s' "$step_i" "$dataset")" "$(py_options_steering_cmd "$dataset")"
    step_i=$((step_i + 1))
  done
fi

log "ALL_DONE"
