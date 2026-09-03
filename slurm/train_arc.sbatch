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

mkdir -p logs

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Config: ${CONFIG_PATH}"
echo "Extra training args: ${EXTRA_ARGS[*]:-none}"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

if [ -f "${HOME}/.venvs/isbi2027/bin/activate" ]; then
    source "${HOME}/.venvs/isbi2027/bin/activate"
else
    echo "Missing virtual environment: ${HOME}/.venvs/isbi2027"
    echo "Create it before submitting this job."
    exit 1
fi

python -c "import torch, pandas, numpy, nibabel; print('Python environment OK'); print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"

if grep -q 'name = "clinical_only"' "${CONFIG_PATH}"; then
    python -m src.smoke_check --config "${CONFIG_PATH}" "${EXTRA_ARGS[@]}"
else
    if [ -z "${DATA_ROOT}" ]; then
        echo "HECKTOR_DATA_ROOT must be set for imaging/fusion experiments."
        exit 1
    fi

    export HECKTOR_DATA_ROOT="${DATA_ROOT}"
    python -m src.smoke_check --config "${CONFIG_PATH}" --require-images "${EXTRA_ARGS[@]}"
fi

python -m src.train --config "${CONFIG_PATH}" "${EXTRA_ARGS[@]}"
