from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class HecktorDataset(Dataset):
    def __init__(
        self,
        dataFolder: str,
        clinicalData: pd.DataFrame,
        patientIdColumn: str,
        clinicalColumns: List[str],
        targetColumns: List[str],
        caseIds: List[str],
        transform: Optional[Callable] = None
    ):
        self.dataFolder = Path(dataFolder)
        self.caseIds = [str(caseId) for caseId in caseIds]
        self.transform = transform

        self.clinicalColumns = clinicalColumns
        self.targetColumns = targetColumns

        clinicalData = clinicalData.copy()
        clinicalData[patientIdColumn] = clinicalData[patientIdColumn].astype(str)
        self.clinicalData = clinicalData.set_index(patientIdColumn)

        missingIds = [
            caseId
            for caseId in self.caseIds
            if caseId not in self.clinicalData.index
        ]

        if missingIds:
            raise ValueError(
                f"Dude {len(missingIds)} patients from the split are missing "
                f"from the clinical CSV: {missingIds[:5]}"
            )

    def __len__(self):
        return len(self.caseIds)

    def getPatientPaths(self, caseId):
        patientFolder = self.dataFolder / caseId

        ctPath = patientFolder / f"{caseId}__CT.nii.gz"
        petPath = patientFolder / f"{caseId}__PT.nii.gz"
        labelPath = patientFolder / f"{caseId}.nii.gz"

        if not ctPath.exists():
            raise FileNotFoundError(f"Dude I could not find CT: {ctPath}")

        if not petPath.exists():
            raise FileNotFoundError(f"Dude I could not find PET: {petPath}")

        if not labelPath.exists():
            raise FileNotFoundError(f"Dude I could not find label: {labelPath}")

        return ctPath, petPath, labelPath

    def __getitem__(self, index):
        caseId = self.caseIds[index]

        ctPath, petPath, labelPath = self.getPatientPaths(caseId)
        patientRow = self.clinicalData.loc[caseId]

        clinicalFeatures = torch.tensor(
            patientRow[self.clinicalColumns].to_numpy(dtype=np.float32),
            dtype=torch.float32
        )

        targetValues = patientRow[self.targetColumns].to_numpy(dtype=np.float32)

        targets = torch.tensor(targetValues,dtype=torch.float32)

        targetMask = ~torch.isnan(targets)

        data = {
            "ct": str(ctPath),
            "pet": str(petPath),
            "label": str(labelPath),
            "clinical": clinicalFeatures,
            "targets": targets,
            "targetMask": targetMask,
            "caseId": caseId
        }

        if self.transform:
            data = self.transform(data)

        return data
