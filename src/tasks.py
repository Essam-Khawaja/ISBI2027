import math

import pandas as pd


T_STAGE_MAPPING = {
    "T0": 0,
    "T1": 1,
    "T2": 2,
    "T3": 3,
    "T4": 4,
}

N_STAGE_MAPPING = {
    "N0": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
}

TASK_DEFINITIONS = {
    "relapse": {
        "column": "Relapse",
        "kind": "binary",
        "output_dim": 1,
    },
    "rfs": {
        "column": "RFS",
        "kind": "regression",
        "output_dim": 1,
        "transform": "log1p",
    },
    "t_stage": {
        "column": "T-stage",
        "kind": "multiclass",
        "output_dim": 5,
    },
    "n_stage": {
        "column": "N-stage",
        "kind": "multiclass",
        "output_dim": 4,
    },
}


def get_task_specs(config):
    task_names = config.get("targets", {}).get("tasks", ["t_stage", "n_stage"])
    specs = []

    for task_name in task_names:
        if task_name not in TASK_DEFINITIONS:
            valid = ", ".join(sorted(TASK_DEFINITIONS))
            raise ValueError(f"Unknown task '{task_name}'. Valid tasks: {valid}")

        spec = dict(TASK_DEFINITIONS[task_name])
        spec["name"] = task_name
        specs.append(spec)

    return specs


def target_columns_for_tasks(task_specs):
    return [task["column"] for task in task_specs]


def normalize_target_columns(dataframe):
    dataframe = dataframe.copy()

    if "T-stage" in dataframe.columns:
        dataframe["T-stage"] = dataframe["T-stage"].map(
            lambda value: map_stage_value(value, T_STAGE_MAPPING)
        )

    if "N-stage" in dataframe.columns:
        dataframe["N-stage"] = dataframe["N-stage"].map(
            lambda value: map_stage_value(value, N_STAGE_MAPPING)
        )

    for column in ["Relapse", "RFS", "T-stage", "N-stage"]:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    return dataframe


def map_stage_value(value, mapping):
    if pd.isna(value):
        return math.nan

    if isinstance(value, str):
        stripped = value.strip()

        if stripped in mapping:
            return mapping[stripped]

    return value

