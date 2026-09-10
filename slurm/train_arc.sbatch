#!/bin/bash
#SBATCH --job-name=isbi2027
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

CONFIG_PATH="${1:-configs/clinical_only.toml}"
if [ "$#" -gt 0 ]; then
    shift
fi
EXTRA_ARGS=("$@")
DATA_ROOT="${HECKTOR_DATA_ROOT:-}"
VENV_PATH="${ISBI2027_VENV:-${HOME}/.venvs/isbi2027}"
export PYTHONUNBUFFERED=1

mkdir -p logs

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Config: ${CONFIG_PATH}"
echo "Extra training args: ${EXTRA_ARGS[*]:-none}"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

if [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
else
    echo "Missing virtual environment: ${VENV_PATH}"
    echo "See slurm/README.md for ARC environment setup."
    exit 1
fi

MODEL_NAME=$(python -c 'import sys; from src.config import load_config; print(load_config(sys.argv[1])["model"]["name"])' "${CONFIG_PATH}")

python - "${MODEL_NAME}" <<'PY'
import sys
import torch
import pandas
import numpy
import nibabel

print("Python:", sys.executable)
print("torch:", torch.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
if sys.argv[1] != "clinical_only":
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA unavailable: imaging/fusion ARC jobs require a GPU allocation "
            "and a compatible PyTorch build. Install requirements-arc.txt in "
            "a fresh environment; see slurm/README.md."
        )
    torch.ones(1, device="cuda").sum().item()
    print("GPU:", torch.cuda.get_device_name(0))
print("Environment preflight passed.")
PY

if [ "${MODEL_NAME}" = "clinical_only" ]; then
    DEVICE="cpu"
    python -m src.smoke_check --config "${CONFIG_PATH}" "${EXTRA_ARGS[@]}" --device "${DEVICE}"
else
    if [ -z "${DATA_ROOT}" ]; then
        echo "HECKTOR_DATA_ROOT must be set for imaging/fusion experiments."
        exit 1
    fi

    export HECKTOR_DATA_ROOT="${DATA_ROOT}"
    DEVICE="cuda"
    python -m src.smoke_check --config "${CONFIG_PATH}" --require-images "${EXTRA_ARGS[@]}" --device "${DEVICE}"
fi

python -m src.train --config "${CONFIG_PATH}" "${EXTRA_ARGS[@]}" --device "${DEVICE}"
