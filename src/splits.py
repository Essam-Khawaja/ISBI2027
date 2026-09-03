import json
from pathlib import Path


def load_fold(splits_file, fold_index):
    splits_file = Path(splits_file)

    if not splits_file.exists():
        raise FileNotFoundError(f"Split file does not exist: {splits_file}")

    with open(splits_file, "r") as file:
        folds = json.load(file)

    if fold_index < 0 or fold_index >= len(folds):
        raise ValueError(
            f"Fold {fold_index} does not exist. Available folds: 0-{len(folds) - 1}"
        )

    fold = folds[fold_index]
    split = {
        "train": [str(patient_id) for patient_id in fold.get("train", [])],
        "val": [str(patient_id) for patient_id in fold.get("val", [])],
        "test": [str(patient_id) for patient_id in fold.get("test", [])],
    }

    validate_split(split, fold_index)
    return split


def validate_split(split, fold_index):
    names = ["train", "val", "test"]

    for index, left_name in enumerate(names):
        for right_name in names[index + 1:]:
            overlap = set(split[left_name]) & set(split[right_name])

            if overlap:
                preview = sorted(overlap)[:5]
                raise ValueError(
                    f"Fold {fold_index} has overlap between {left_name} and "
                    f"{right_name}: {preview}"
                )


def check_split_patients_exist(split, patient_ids):
    patient_ids = set(str(patient_id) for patient_id in patient_ids)
    missing = {}

    for split_name, split_ids in split.items():
        missing_ids = sorted(set(split_ids) - patient_ids)

        if missing_ids:
            missing[split_name] = missing_ids

    if missing:
        raise ValueError(f"Split contains patients missing from metadata: {missing}")


def split_center_counts(dataframe, split):
    counts = {}

    if "CenterID" not in dataframe.columns:
        return counts

    for split_name, patient_ids in split.items():
        rows = dataframe[dataframe["PatientID"].isin(patient_ids)]
        value_counts = rows["CenterID"].value_counts(dropna=False).sort_index()
        counts[split_name] = {
            str(center): int(count)
            for center, count in value_counts.items()
        }

    return counts

