from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # Python 3.10 compatibility.


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path_value):
    path = Path(path_value).expanduser()

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_config(config_path):
    path = resolve_project_path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    with open(path, "rb") as config_file:
        config = tomllib.load(config_file)

    config["_config_path"] = str(path)
    config["_config_text"] = path.read_text()

    return config


def get_nested(config, keys, default=None):
    current = config

    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default

        current = current[key]

    return current


def set_nested(config, keys, value):
    current = config

    for key in keys[:-1]:
        current = current.setdefault(key, {})

    current[keys[-1]] = value


def apply_cli_overrides(config, args):
    overrides = {
        ("split", "fold"): args.fold,
        ("training", "epochs"): args.epochs,
        ("training", "batch_size"): args.batch_size,
        ("training", "max_train_batches"): args.max_train_batches,
        ("training", "max_val_batches"): args.max_val_batches,
        ("training", "learning_rate"): args.learning_rate,
        ("paths", "output_dir"): args.output_dir,
        ("device", "name"): args.device,
    }

    for keys, value in overrides.items():
        if value is not None:
            set_nested(config, keys, value)

    return config
