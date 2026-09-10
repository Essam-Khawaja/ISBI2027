# ARC baseline jobs

Run these commands from the repository root on ARC, using the Python 3.10
module already available in your session. The baseline pipeline uses
`requirements-arc.txt`; the original `requirements.txt` pins CUDA 13 packages
that are incompatible with the reported ARC driver 560.35.03.

Create a separate environment so the existing installation is preserved:

```bash
python -m venv "$HOME/.venvs/isbi2027-cu126"
source "$HOME/.venvs/isbi2027-cu126/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements-arc.txt
python -m pip check
export ISBI2027_VENV="$HOME/.venvs/isbi2027-cu126"
mkdir -p logs
```

Export `ISBI2027_VENV` again when submitting from a new shell. Without it,
the script uses `$HOME/.venvs/isbi2027` for compatibility with existing setups.
Do not install the original requirements file into this ARC environment.
MONAI and the legacy scripts are outside this minimal baseline environment.

Python 3.10 uses the `tomli` backport; Python 3.11+ uses built-in `tomllib`.
The PyTorch wheel includes its CUDA runtime. Loading a different CUDA module
does not replace the runtime in an incompatible PyTorch wheel.

## Submit

Slurm uses your default account. Create `logs` before submission because Slurm
opens output files before running the script. Start with a short clinical run:

```bash
sbatch --time=00:15:00 slurm/train_arc.sbatch configs/clinical_only.toml --fold 0 --epochs 1 --max-train-batches 2 --max-val-batches 2
```

Then submit the full clinical baseline:

```bash
sbatch slurm/train_arc.sbatch configs/clinical_only.toml --fold 0 --epochs 20
```

For imaging and fusion, set the actual dataset location before submission:

```bash
export HECKTOR_DATA_ROOT="/actual/path/to/HECKTOR 2026 Training Data"
sbatch --gpus-per-node=a100:1 --cpus-per-task=8 --mem=64G --time=24:00:00 slurm/train_arc.sbatch configs/imaging_petct.toml --fold 0 --epochs 20
sbatch --gpus-per-node=a100:1 --cpus-per-task=8 --mem=64G --time=24:00:00 slurm/train_arc.sbatch configs/fusion_petct_tabular.toml --fold 0 --epochs 20
```

The wrapper runs clinical jobs on CPU and imaging/fusion jobs on CUDA. It
checks CUDA availability and performs a small GPU operation before loading
images. Imaging/fusion smoke checks require real images. GPU checks must run
inside a GPU allocation; `cuda False` on a CPU or login node is not evidence
of a broken GPU installation.

```bash
squeue -u "$USER"
tail -f logs/isbi2027_<JOBID>.out
```

Errors are in `logs/isbi2027_<JOBID>.err`; metrics and checkpoints are in `runs/`.

## Compatibility references

- [PyTorch 2.6.0 CUDA 12.6 wheels](https://docs.pytorch.org/get-started/previous-versions/)
- [NVIDIA driver compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
