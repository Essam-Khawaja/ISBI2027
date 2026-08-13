import json
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import HecktorDataset
from fusionModel import FusionModel

from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped

PROJECT_ROOT = Path(__file__).resolve().parent

CLINICAL_FILE = PROJECT_ROOT / "data" / "HECKTOR_2026_training_data.csv"
SPLITS_FILE = PROJECT_ROOT / "data" / "splits_final.json"

CLINICAL_COLUMNS = [
    "Age",
    "Gender",
    "Tobacco Consumption",
    "Alcohol Consumption",
    "Performance Status",
    "Treatment",
    "HPV Status",
]

# Order matters later:
# targets[0] = relapse/event indicator
# targets[1] = RFS time
# targets[2] = T-stage class
# targets[3] = N-stage class
TARGET_COLUMNS = [
    "Relapse",
    "RFS",
    "T-stage",
    "N-stage",
]

T_STAGE_MAPPING = {
    "T0": 0.0,
    "T1": 1.0,
    "T2": 2.0,
    "T3": 3.0,
    "T4": 4.0,
}

N_STAGE_MAPPING = {
    "N0": 0.0,
    "N1": 1.0,
    "N2": 2.0,
    "N3": 3.0,
}

def getDataFolder():
    envDataFolder = os.environ.get("HECKTOR_DATA_ROOT")

    if envDataFolder:
        dataFolder = Path(envDataFolder).expanduser().resolve()

        if not dataFolder.is_dir():
            raise FileNotFoundError(
                f"HECKTOR_DATA_ROOT points to a missing folder: {dataFolder}"
            )

        return dataFolder

    candidates = [
        PROJECT_ROOT.parent.parent / "HECKTOR 2026 Training Data",
        PROJECT_ROOT.parent / "HECKTOR 2026 Training Data",
        PROJECT_ROOT / "HECKTOR 2026 Training Data",
        PROJECT_ROOT / "hecktor2026_training",
        Path.home() / "HECKTOR 2026 Training Data",
    ]

    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    raise FileNotFoundError(
        "Dude I could not find the HECKTOR data folder. "
        "Set the HECKTOR_DATA_ROOT environment variable."
    )

DATA_FOLDER = getDataFolder()

def getCasePaths(dataFolder, caseId):
    caseFolder = Path(dataFolder) / caseId / "preprocessed"

    return {
        "ct": caseFolder / f"{caseId}__CT.nii.gz",
        "pet": caseFolder / f"{caseId}__PT.nii.gz",
        "label": caseFolder / f"{caseId}.nii.gz",
    }

def caseExists(dataFolder, caseId):
    paths = getCasePaths(dataFolder, caseId)
    return all(path.exists() for path in paths.values())

def filterAvailableCases(dataFolder, caseIds):
    availableCaseIds = [
        caseId
        for caseId in caseIds
        if caseExists(dataFolder, caseId)
    ]

    missingCaseIds = sorted(set(caseIds) - set(availableCaseIds))

    if missingCaseIds:
        preview = ", ".join(missingCaseIds[:5])

        if len(missingCaseIds) > 5:
            preview += f" (+{len(missingCaseIds) - 5} more)"

        print(
            f"Skipping {len(missingCaseIds)} cases missing from "
            f"{dataFolder}: {preview}"
        )

    return availableCaseIds

def loadSplit(foldIndex=0):
    if not SPLITS_FILE.exists():
        raise FileNotFoundError(
            f"Dude I could not find the splits JSON at {SPLITS_FILE}"
        )

    with open(SPLITS_FILE, "r") as splitFile:
        folds = json.load(splitFile)

    if foldIndex < 0 or foldIndex >= len(folds):
        raise ValueError(
            f"Fold {foldIndex} does not exist. "
            f"Available folds: 0-{len(folds) - 1}"
        )

    fold = folds[foldIndex]

    trainIds = filterAvailableCases(DATA_FOLDER, fold["train"])
    valIds = filterAvailableCases(DATA_FOLDER, fold["val"])

    overlap = set(trainIds) & set(valIds)

    if overlap:
        raise ValueError(
            f"Dude the train and validation splits overlap: "
            f"{sorted(overlap)[:5]}"
        )

    print("Using fold:", foldIndex)
    print("Training patients:", len(trainIds))
    print("Validation patients:", len(valIds))

    return trainIds, valIds

def loadClinicalData():
    if not CLINICAL_FILE.exists():
        raise FileNotFoundError(
            f"Dude I could not find the clinical CSV at {CLINICAL_FILE}"
        )

    clinicalData = pd.read_csv(CLINICAL_FILE)

    # Remove accidental whitespace from column names
    clinicalData.columns = clinicalData.columns.str.strip()

    requiredColumns = [
        "PatientID",
        *CLINICAL_COLUMNS,
        *TARGET_COLUMNS,
    ]

    missingColumns = [
        column
        for column in requiredColumns
        if column not in clinicalData.columns
    ]

    if missingColumns:
        raise ValueError(
            f"Dude these required CSV columns are missing: {missingColumns}"
        )

    clinicalData["PatientID"] = clinicalData["PatientID"].astype(str)

    # Convert staging strings into numeric ordinal values
    clinicalData["T-stage"] = clinicalData["T-stage"].map(T_STAGE_MAPPING)
    clinicalData["N-stage"] = clinicalData["N-stage"].map(N_STAGE_MAPPING)

    # Make sure all model columns are numeric
    for column in CLINICAL_COLUMNS + TARGET_COLUMNS:
        clinicalData[column] = pd.to_numeric(
            clinicalData[column],
            errors="coerce"
        )

    if clinicalData["PatientID"].duplicated().any():
        duplicateIds = clinicalData.loc[
            clinicalData["PatientID"].duplicated(),
            "PatientID"
        ].tolist()

        raise ValueError(
            f"Dude the clinical CSV contains duplicate patients: "
            f"{duplicateIds[:5]}"
        )

    print("Clinical rows:", len(clinicalData))
    print("Clinical columns:", len(clinicalData.columns))

    return clinicalData

def getLoadTransform():
    return Compose([
        LoadImaged(keys=["ct", "pet", "label"]),
        EnsureChannelFirstd(keys=["ct", "pet", "label"]),
        EnsureTyped(keys=["ct", "pet", "label"]),
    ])

def preprocessClinicalData(clinicalData, trainIds):
    clinicalData = clinicalData.copy()

    trainRows = clinicalData[
        clinicalData["PatientID"].isin(trainIds)
    ]

    if trainRows.empty:
        raise ValueError("Dude none of the training IDs matched the clinical CSV.")

    # Learn replacement values only from training patients.
    # This prevents validation information from leaking into training.
    featureMedians = {}

    for column in CLINICAL_COLUMNS:
        medianValue = trainRows[column].median()

        if pd.isna(medianValue):
            raise ValueError(
                f"Dude the training fold has no usable values for {column}"
            )

        featureMedians[column] = float(medianValue)
        clinicalData[column] = clinicalData[column].fillna(medianValue)

    remainingMissingFeatures = (
        clinicalData[CLINICAL_COLUMNS].isna().sum().sum()
    )

    if remainingMissingFeatures:
        raise ValueError(
            f"Dude there are still {remainingMissingFeatures} "
            f"missing clinical feature values."
        )

    print("\nFeature missing values replaced with training medians:")

    for column, medianValue in featureMedians.items():
        print(f"  {column}: {medianValue}")

    print("\nMissing target labels:")
    print(clinicalData[TARGET_COLUMNS].isna().sum())

    return clinicalData

def getDevice():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")

def main():
    device = getDevice()

    print("Using device:", device)
    print("Data folder:", DATA_FOLDER)
    print("Clinical columns:", CLINICAL_COLUMNS)
    print("Target columns:", TARGET_COLUMNS)

    trainIds, valIds = loadSplit(foldIndex=0)

    clinicalData = loadClinicalData()
    clinicalData = preprocessClinicalData(clinicalData, trainIds)

    dataset = HecktorDataset(
        dataFolder=DATA_FOLDER,
        clinicalData=clinicalData,
        patientIdColumn="PatientID",
        clinicalColumns=CLINICAL_COLUMNS,
        targetColumns=TARGET_COLUMNS,
        caseIds=trainIds,
        transform=getLoadTransform(),
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    batch = next(iter(loader))

    ct = batch["ct"].to(device).float()
    pet = batch["pet"].to(device).float()
    clinical = batch["clinical"].to(device).float()

    model = FusionModel().to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(ct, pet, clinical)

    print("\nCase:", batch["caseId"])
    print("CT:", ct.shape)
    print("PET:", pet.shape)
    print("Clinical:", clinical.shape)
    print("Targets:", batch["targets"])
    print("Target available:", batch["targetMask"])

    print("\nOutputs:")
    print("T-stage:", outputs["tStage"].shape)
    print("N-stage:", outputs["nStage"].shape)
    print("RFS:", outputs["rfs"].shape)


if __name__ == "__main__":
    main()
