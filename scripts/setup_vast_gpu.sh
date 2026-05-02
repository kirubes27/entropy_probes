#!/usr/bin/env bash
set -euo pipefail

# Fresh Vast setup for the entropy-probes PRISM run.
#
# Required:
#   REMOTE_HOST=root@host REMOTE_PORT=12345 bash scripts/setup_vast_gpu.sh
#
# This script does not ask for or print your Hugging Face token. After it
# finishes, run the printed `huggingface-cli login` command in your terminal.

REMOTE_HOST="${REMOTE_HOST:-}"
REMOTE_PORT="${REMOTE_PORT:-}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_REPO="${REMOTE_REPO:-/workspace/entropy_probes}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/entropy_venv/bin/python}"
VENV_DIR="${VENV_DIR:-/workspace/entropy_venv}"
BRANCH="${BRANCH:-codex/clean-rerun}"
GIT_URL="${GIT_URL:-https://github.com/kirubes27/entropy_probes.git}"

if [ -z "$REMOTE_HOST" ] || [ -z "$REMOTE_PORT" ]; then
  echo "ERROR: Set REMOTE_HOST and REMOTE_PORT." >&2
  echo "Example: REMOTE_HOST=root@ssh.vast.ai REMOTE_PORT=12345 bash scripts/setup_vast_gpu.sh" >&2
  exit 1
fi

ssh_cmd() {
  ssh -i "$SSH_KEY" -p "$REMOTE_PORT" -o StrictHostKeyChecking=no "$REMOTE_HOST" "$@"
}

echo "== Vast setup =="
echo "Remote: $REMOTE_HOST:$REMOTE_PORT"
echo "Repo:   $REMOTE_REPO"
echo "Venv:   $VENV_DIR"
echo

ssh_cmd "set -euo pipefail
mkdir -p /workspace /workspace/.cache/huggingface /workspace/.cache/huggingface/hub /workspace/runlogs
if [ ! -d '$REMOTE_REPO/.git' ]; then
  rm -rf '$REMOTE_REPO'
  git clone '$GIT_URL' '$REMOTE_REPO'
fi
cd '$REMOTE_REPO'
git fetch --all --prune
git checkout '$BRANCH'
git pull --ff-only origin '$BRANCH'
python3 -m venv '$VENV_DIR'
'$PYTHON_BIN' -m pip install --upgrade pip setuptools wheel
'$VENV_DIR/bin/pip' install -r requirements.txt
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/hub
'$PYTHON_BIN' - <<'PY'
import torch
import transformers
import bitsandbytes
from core.model_utils import get_model_dir_name
print('CUDA_AVAILABLE', torch.cuda.is_available())
print('GPU_COUNT', torch.cuda.device_count())
if torch.cuda.is_available():
    print('GPU_NAME', torch.cuda.get_device_name(0))
print('TORCH', torch.__version__)
print('TRANSFORMERS', transformers.__version__)
print('MODEL_DIR', get_model_dir_name('meta-llama/Llama-3.3-70B-Instruct', None, True, False))
PY
"

cat <<EOF

Setup complete.

Now enter your Hugging Face token interactively with:

  ssh -p $REMOTE_PORT $REMOTE_HOST
  export HF_HOME=/workspace/.cache/huggingface
  $VENV_DIR/bin/huggingface-cli login

If that CLI is unavailable on a future image, use:

  $VENV_DIR/bin/hf auth login

After login, verify with:

  REMOTE_HOST=$REMOTE_HOST REMOTE_PORT=$REMOTE_PORT bash scripts/run_prism_gpu_sequence.sh --preflight-only

EOF
