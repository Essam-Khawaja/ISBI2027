import os
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np


def getDataRoot():
    envRoot = os.environ.get("HECKTOR_DATA_ROOT")

    if envRoot:
        return Path(envRoot).expanduser().resolve()

    return (Path.home() / "HECKTOR 2026 Training Data").resolve()

def getPatientFolders(root):
    return sorted(
        folder for folder in root.iterdir()
        if folder.is_dir() and "-" in folder.name
    )

def getPatientPaths(patientFolder):
    caseId = patientFolder.name
    preprocessedFolder = patientFolder / "preprocessed"

    return {
        "ct": preprocessedFolder / f"{caseId}__CT.nii.gz",
        "pet": preprocessedFolder / f"{caseId}__PT.nii.gz",
        "mask": preprocessedFolder / f"{caseId}.nii.gz",
    }

def getSpacing(image):
    return tuple(round(float(x), 4) for x in image.header.get_zooms()[:3])

def getArrayStats(data):
    return {
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "p01": float(np.percentile(data, 1)),
        "p50": float(np.percentile(data, 50)),
        "p99": float(np.percentile(data, 99)),
    }

def updateIntensitySummary(summary, modality, stats):
    if modality not in summary:
        summary[modality] = {
            "min": [],
            "max": [],
            "mean": [],
            "std": [],
            "p01": [],
            "p50": [],
            "p99": [],
        }

    for key, value in stats.items():
        summary[modality][key].append(value)

def printCounter(title, counter):
    print(f"\n{title}:")

    for value, count in counter.items():
        print(value, ":", count)

def printIntensitySummary(summary):
    print("\nIntensity summary across patients:")

    for modality, stats in summary.items():
        print(f"\n{modality.upper()}:")

        for statName, values in stats.items():
            values = np.array(values, dtype=np.float32)
            print(
                f"  {statName}: "
                f"mean={values.mean():.4f}, "
                f"min={values.min():.4f}, "
                f"max={values.max():.4f}"
            )

def checkPatient(patientFolder, counters, missing, mismatches, intensitySummary):
    caseId = patientFolder.name
    paths = getPatientPaths(patientFolder)

    missingModalities = [
        modality
        for modality, path in paths.items()
        if not path.exists()
    ]

    if missingModalities:
        missing.append((caseId, missingModalities))
        return

    images = {
        modality: nib.load(path)
        for modality, path in paths.items()
    }

    shapes = {
        modality: image.shape
        for modality, image in images.items()
    }

    spacings = {
        modality: getSpacing(image)
        for modality, image in images.items()
    }

    counters["shape"][tuple(shapes.values())] += 1
    counters["spacing"][tuple(spacings.values())] += 1
    counters["ctShape"][shapes["ct"]] += 1
    counters["petShape"][shapes["pet"]] += 1
    counters["maskShape"][shapes["mask"]] += 1
    counters["ctSpacing"][spacings["ct"]] += 1
    counters["petSpacing"][spacings["pet"]] += 1
    counters["maskSpacing"][spacings["mask"]] += 1

    if len(set(shapes.values())) != 1:
        mismatches.append((caseId, "shape", shapes))

    if len(set(spacings.values())) != 1:
        mismatches.append((caseId, "spacing", spacings))

    ctData = images["ct"].get_fdata(dtype=np.float32)
    petData = images["pet"].get_fdata(dtype=np.float32)
    maskData = images["mask"].get_fdata(dtype=np.float32)

    updateIntensitySummary(intensitySummary, "ct", getArrayStats(ctData))
    updateIntensitySummary(intensitySummary, "pet", getArrayStats(petData))

    maskValues, maskCounts = np.unique(maskData, return_counts=True)
    maskLabelTuple = tuple(float(value) for value in maskValues)
    counters["maskLabels"][maskLabelTuple] += 1

    for value, count in zip(maskValues, maskCounts):
        counters["maskVoxelCounts"][float(value)] += int(count)

def main():
    root = getDataRoot()

    if not root.is_dir():
        raise FileNotFoundError(
            f"Dude I could not find the HECKTOR data folder: {root}"
        )

    patientFolders = getPatientFolders(root)

    counters = {
        "shape": Counter(),
        "spacing": Counter(),
        "ctShape": Counter(),
        "petShape": Counter(),
        "maskShape": Counter(),
        "ctSpacing": Counter(),
        "petSpacing": Counter(),
        "maskSpacing": Counter(),
        "maskLabels": Counter(),
        "maskVoxelCounts": Counter(),
    }

    missing = []
    mismatches = []
    intensitySummary = {}

    for patientFolder in patientFolders:
        checkPatient(
            patientFolder=patientFolder,
            counters=counters,
            missing=missing,
            mismatches=mismatches,
            intensitySummary=intensitySummary,
        )

    print("Data root:", root)
    print("Patients checked:", len(patientFolders))
    print("Patients missing files:", len(missing))
    print("Patients with mismatches:", len(mismatches))

    printCounter("CT shapes", counters["ctShape"])
    printCounter("PET shapes", counters["petShape"])
    printCounter("Mask shapes", counters["maskShape"])
    printCounter("CT spacings", counters["ctSpacing"])
    printCounter("PET spacings", counters["petSpacing"])
    printCounter("Mask spacings", counters["maskSpacing"])
    printCounter("Mask label sets", counters["maskLabels"])
    printCounter("Total mask voxels per label", counters["maskVoxelCounts"])

    printIntensitySummary(intensitySummary)

    if missing:
        print("\nFirst missing cases:")
        for caseId, missingModalities in missing[:20]:
            print(caseId, missingModalities)

    if mismatches:
        print("\nFirst mismatch cases:")
        for caseId, mismatchType, values in mismatches[:20]:
            print(caseId, mismatchType, values)

if __name__ == "__main__":
    main()
