import nibabel as nib
from pathlib import Path
from collections import Counter

root = Path.home() / "HECKTOR 2026 Training Data"

shapes = Counter()
spacings = Counter()
missing = []

patientFolders = [
    folder for folder in root.iterdir()
    if folder.is_dir() and "-" in folder.name
]

for patientFolder in patientFolders:
    caseId = patientFolder.name
    ctPath = patientFolder / "preprocessed" / f"{caseId}__CT.nii.gz"

    if not ctPath.exists():
        missing.append(caseId)
        continue

    image = nib.load(ctPath)

    shapes[image.shape] += 1
    spacings[tuple(round(float(x), 4) for x in image.header.get_zooms()[:3])] += 1

print("Patients checked:", len(patientFolders))
print("Missing preprocessed CT:", len(missing))

print("\nShapes:")
for shape, count in shapes.items():
    print(shape, ":", count)

print("\nSpacings:")
for spacing, count in spacings.items():
    print(spacing, ":", count)

if missing:
    print("\nMissing cases:", missing[:20]) 
