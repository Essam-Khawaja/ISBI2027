from encoder import ClinicalEncoder, Encoder3D
from torch import nn, cat

class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.clinicalEncoder = ClinicalEncoder()
        self.petEncoder = Encoder3D()
        self.ctEncoder = Encoder3D()

        self.fusionLayers = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.tStageHead = nn.Linear(32, 5)
        self.nStageHead = nn.Linear(32, 4)
        self.rfsHead = nn.Linear(32, 1)

    def forward(self, ct, pet, clinical):
        ct = self.ctEncoder(ct)
        pet = self.petEncoder(pet)
        clinical = self.clinicalEncoder(clinical)

        x = cat([ct, pet, clinical], dim=1)
        x = self.fusionLayers(x)

        tStage = self.tStageHead(x)
        nStage = self.nStageHead(x)
        rfs = self.rfsHead(x)

        return {
            "tStage": tStage,
            "nStage": nStage,
            "rfs": rfs
        }
