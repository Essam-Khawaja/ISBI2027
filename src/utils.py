import csv
from datetime import datetime
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from src.config import resolve_project_path


def set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device(device_name):
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        if torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    return torch.device(device_name)


def batch_to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device)

    if isinstance(batch, dict):
        return {
            key: batch_to_device(value, device)
            for key, value in batch.items()
        }

    if isinstance(batch, list):
        return [
            batch_to_device(value, device)
            for value in batch
        ]

    return batch


def make_run_dir(config):
    output_dir = resolve_project_path(config.get("paths", {}).get("output_dir", "runs"))
    experiment_name = config.get("experiment_name", "experiment")
    fold = config.get("split", {}).get("fold", 0)
    seed = config.get("seed", 42)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_dir / experiment_name / f"{timestamp}_fold{fold}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as file:
        json.dump(payload, file, indent=2, sort_keys=True, default=json_default)


def json_default(value):
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        value = float(value)

    if isinstance(value, float):
        if math.isnan(value):
            return None

        return value

    if torch.is_tensor(value):
        return value.detach().cpu().tolist()

    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def append_metrics_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()

    with open(path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))

        if write_header:
            writer.writeheader()

        writer.writerow(row)


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

