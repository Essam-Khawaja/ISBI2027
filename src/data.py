from dataclasses import asdict, dataclass
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import PROJECT_ROOT, resolve_project_path
from src.imaging import load_image_tensor
from src.splits import check_split_patients_exist, load_fold, split_center_counts
from src.tasks import get_task_specs, normalize_target_columns, target_columns_for_tasks


CLINICAL_COLUMNS = [
    "Age",
    "Gender",
    "Tobacco Consumption",
    "Alcohol Consumption",
    "Performance Status",
    "Treatment",
    "HPV Status",
]

TUMOR_FEATURE_PREFIXES = (
    "GTVp_",
    "GTVn_",
    "Total_tumor_",
)

TUMOR_FEATURE_SUFFIXES = (
    "_volume_ratio",
)


@dataclass
class TabularPreprocessor:
    feature_columns: list
    medians: dict
    means: dict
    stds: dict
    standardize: bool

    def to_dict(self):
        return asdict(self)


class HecktorPatientDataset(Dataset):
    def __init__(
        self,
        dataframe,
        case_ids,
        feature_columns,
        task_specs,
        data_root=None,
        image_config=None,
        load_images=False,
    ):
        self.dataframe = dataframe.copy()
        self.dataframe["PatientID"] = self.dataframe["PatientID"].astype(str)
        self.dataframe = self.dataframe.set_index("PatientID", drop=False)
        self.case_ids = [str(case_id) for case_id in case_ids]
        self.feature_columns = feature_columns
        self.task_specs = task_specs
        self.data_root = data_root
        self.image_config = image_config or {}
        self.load_images = load_images

        missing_ids = [
            case_id for case_id in self.case_ids
            if case_id not in self.dataframe.index
        ]

        if missing_ids:
            raise ValueError(f"Dataset is missing patients from split: {missing_ids[:5]}")

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, index):
        case_id = self.case_ids[index]
        row = self.dataframe.loc[case_id]

        sample = {
            "case_id": case_id,
            "center_id": row["CenterID"] if "CenterID" in row else -1,
            "targets": {},
            "target_masks": {},
        }

        if self.feature_columns:
            sample["tabular"] = torch.tensor(
                row[self.feature_columns].to_numpy(dtype=np.float32),
                dtype=torch.float32,
            )

        if self.load_images:
            sample["image"] = load_image_tensor(
                case_id=case_id,
                data_root=self.data_root,
                image_config=self.image_config,
            )

        for task in self.task_specs:
            value = row[task["column"]]
            is_valid = not pd.isna(value)
            target_value = 0.0 if not is_valid else float(value)
            sample["targets"][task["name"]] = torch.tensor(target_value, dtype=torch.float32)
            sample["target_masks"][task["name"]] = torch.tensor(is_valid, dtype=torch.bool)

        return sample


def read_metadata(metadata_csv):
    dataframe = pd.read_csv(metadata_csv)
    dataframe.columns = dataframe.columns.str.strip()

    if "PatientID" not in dataframe.columns:
        raise ValueError("Metadata CSV must contain a PatientID column.")

    if dataframe["PatientID"].duplicated().any():
        duplicate_ids = dataframe.loc[
            dataframe["PatientID"].duplicated(),
            "PatientID",
        ].astype(str).tolist()
        raise ValueError(f"Metadata has duplicate PatientID values: {duplicate_ids[:5]}")

    dataframe["PatientID"] = dataframe["PatientID"].astype(str)
    dataframe = normalize_target_columns(dataframe)

    return dataframe


def select_feature_columns(dataframe, config):
    feature_config = config.get("features", {})
    feature_sets = feature_config.get("sets", ["clinical"])
    explicit_columns = feature_config.get("columns", [])
    columns = []

    if "clinical" in feature_sets:
        columns.extend(CLINICAL_COLUMNS)

    if "tumor" in feature_sets:
        columns.extend(find_tumor_feature_columns(dataframe))

    columns.extend(explicit_columns)

    deduplicated = []

    for column in columns:
        if column not in deduplicated:
            deduplicated.append(column)

    missing = [column for column in deduplicated if column not in dataframe.columns]

    if missing:
        raise ValueError(f"Requested tabular feature columns are missing: {missing}")

    return deduplicated


def find_tumor_feature_columns(dataframe):
    protected = set(["PatientID", "CenterID", *CLINICAL_COLUMNS])
    protected.update(["Relapse", "RFS", "T-stage", "N-stage"])
    columns = []

    for column in dataframe.columns:
        is_tumor_feature = column.startswith(TUMOR_FEATURE_PREFIXES)
        is_tumor_feature = is_tumor_feature or column.endswith(TUMOR_FEATURE_SUFFIXES)

        if is_tumor_feature and column not in protected:
            columns.append(column)

    return columns


def fit_transform_tabular(dataframe, train_ids, feature_columns, standardize):
    dataframe = dataframe.copy()

    if not feature_columns:
        return dataframe, TabularPreprocessor([], {}, {}, {}, standardize)

    for column in feature_columns:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    train_mask = dataframe["PatientID"].isin(train_ids)
    train_rows = dataframe.loc[train_mask, feature_columns]

    if train_rows.empty:
        raise ValueError("No training rows matched the split IDs.")

    medians = {}
    means = {}
    stds = {}

    for column in feature_columns:
        median = train_rows[column].median()

        if pd.isna(median):
            raise ValueError(f"Training split has no usable values for feature: {column}")

        dataframe[column] = dataframe[column].fillna(median)
        medians[column] = float(median)

    filled_train = dataframe.loc[train_mask, feature_columns]

    for column in feature_columns:
        mean = float(filled_train[column].mean())
        std = float(filled_train[column].std(ddof=0))

        if not np.isfinite(std) or std < 1e-8:
            std = 1.0

        means[column] = mean
        stds[column] = std

        if standardize:
            dataframe[column] = (dataframe[column] - mean) / std

    remaining_missing = int(dataframe[feature_columns].isna().sum().sum())

    if remaining_missing:
        raise ValueError(f"Tabular preprocessing left {remaining_missing} missing values.")

    preprocessor = TabularPreprocessor(
        feature_columns=feature_columns,
        medians=medians,
        means=means,
        stds=stds,
        standardize=standardize,
    )

    return dataframe, preprocessor


def make_datasets(config, allow_missing_images=False):
    paths_config = config.get("paths", {})
    split_config = config.get("split", {})
    model_name = config.get("model", {}).get("name", "clinical_only")
    requires_images = model_name in {"imaging_petct", "fusion_petct_tabular"}

    metadata_csv = resolve_project_path(paths_config.get("metadata_csv"))
    splits_file = resolve_project_path(paths_config.get("splits_file"))
    fold_index = int(split_config.get("fold", 0))

    dataframe = read_metadata(metadata_csv)
    task_specs = get_task_specs(config)
    required_target_columns = target_columns_for_tasks(task_specs)
    missing_targets = [
        column for column in required_target_columns
        if column not in dataframe.columns
    ]

    if missing_targets:
        raise ValueError(f"Metadata CSV is missing target columns: {missing_targets}")

    split = load_fold(splits_file, fold_index)
    check_split_patients_exist(split, dataframe["PatientID"])

    feature_columns = select_feature_columns(dataframe, config)
    standardize = bool(config.get("features", {}).get("standardize", True))
    dataframe, preprocessor = fit_transform_tabular(
        dataframe=dataframe,
        train_ids=split["train"],
        feature_columns=feature_columns,
        standardize=standardize,
    )

    data_root = resolve_data_root(config, required=requires_images and not allow_missing_images)
    images_available = data_root is not None
    load_images = requires_images and images_available

    train_dataset = HecktorPatientDataset(
        dataframe=dataframe,
        case_ids=split["train"],
        feature_columns=feature_columns,
        task_specs=task_specs,
        data_root=data_root,
        image_config=config.get("image", {}),
        load_images=load_images,
    )
    val_dataset = HecktorPatientDataset(
        dataframe=dataframe,
        case_ids=split["val"],
        feature_columns=feature_columns,
        task_specs=task_specs,
        data_root=data_root,
        image_config=config.get("image", {}),
        load_images=load_images,
    )

    info = {
        "fold": fold_index,
        "split": split,
        "center_counts": split_center_counts(dataframe, split),
        "feature_columns": feature_columns,
        "tabular_preprocessor": preprocessor.to_dict(),
        "task_specs": task_specs,
        "data_root": str(data_root) if data_root else None,
        "images_available": images_available,
        "requires_images": requires_images,
    }

    return train_dataset, val_dataset, info


def resolve_data_root(config, required):
    paths_config = config.get("paths", {})
    explicit_root = paths_config.get("data_root")
    env_name = paths_config.get("data_root_env", "HECKTOR_DATA_ROOT")
    candidates = []

    if explicit_root:
        candidates.append(Path(explicit_root).expanduser())

    env_root = os.environ.get(env_name)

    if env_root:
        candidates.append(Path(env_root).expanduser())

    candidates.extend([
        PROJECT_ROOT.parent.parent / "HECKTOR 2026 Training Data",
        PROJECT_ROOT.parent / "HECKTOR 2026 Training Data",
        PROJECT_ROOT / "HECKTOR 2026 Training Data",
        PROJECT_ROOT / "hecktor2026_training",
        Path.home() / "HECKTOR 2026 Training Data",
    ])

    for candidate in candidates:
        candidate = candidate.resolve()

        if candidate.is_dir():
            return candidate

    if required:
        raise FileNotFoundError(
            "Image data root was not found. Set HECKTOR_DATA_ROOT or paths.data_root "
            "in the config before running imaging or fusion experiments."
        )

    return None

