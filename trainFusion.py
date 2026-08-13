from main import loadSplit, loadClinicalData, preprocessClinicalData, getLoadTransform, DATA_FOLDER
from main import CLINICAL_COLUMNS, TARGET_COLUMNS, getDevice
from dataset import HecktorDataset
from fusionModel import FusionModel

import torch
from torch import nn
from torch.utils.data import DataLoader

def makeDatasets(foldIndex):
    trainIds, valIds = loadSplit(foldIndex=foldIndex)

    clinicalData = loadClinicalData()
    clinicalData = preprocessClinicalData(clinicalData, trainIds)

    trainDataset = HecktorDataset(
        dataFolder=DATA_FOLDER,
        clinicalData=clinicalData,
        patientIdColumn="PatientID",
        clinicalColumns=CLINICAL_COLUMNS,
        targetColumns=TARGET_COLUMNS,
        caseIds=trainIds,
        transform=getLoadTransform(),
    )

    valDataset = HecktorDataset(
        dataFolder=DATA_FOLDER,
        clinicalData=clinicalData,
        patientIdColumn="PatientID",
        clinicalColumns=CLINICAL_COLUMNS,
        targetColumns=TARGET_COLUMNS,
        caseIds=valIds,
        transform=getLoadTransform(),
    )

    return trainDataset, valDataset

def makeLoaders(trainDataset, valDataset, batchSize):
    trainLoader = DataLoader(
        trainDataset,
        batch_size=batchSize,
        shuffle=True,
        num_workers=0,
    )

    valLoader = DataLoader(
        valDataset,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
    )

    return trainLoader, valLoader

def computeLoss(outputs, batch, device):
    targets = batch["targets"].to(device)
    targetMask = batch["targetMask"].to(device)

    relapse = targets[:, 0]
    rfs = targets[:, 1]
    tStage = targets[:, 2]
    nStage = targets[:, 3]

    tLossFn = nn.CrossEntropyLoss()
    nLossFn = nn.CrossEntropyLoss()
    rfsLossFn = nn.SmoothL1Loss()

    totalLoss = torch.tensor(0.0, device=device)
    lossParts = {}

    validT = targetMask[:, 2]
    if validT.any():
        tLoss = tLossFn(outputs["tStage"][validT], tStage[validT].long())
        totalLoss = totalLoss + tLoss
        lossParts["tLoss"] = tLoss.item()

    validN = targetMask[:, 3]
    if validN.any():
        nLoss = nLossFn(outputs["nStage"][validN], nStage[validN].long())
        totalLoss = totalLoss + nLoss
        lossParts["nLoss"] = nLoss.item()

    validRfs = targetMask[:, 1]
    if validRfs.any():
        rfsTarget = torch.log1p(rfs[validRfs]).unsqueeze(1)
        rfsLoss = rfsLossFn(outputs["rfs"][validRfs], rfsTarget)
        totalLoss = totalLoss + rfsLoss
        lossParts["rfsLoss"] = rfsLoss.item()

    lossParts["totalLoss"] = totalLoss.item()

    return totalLoss, lossParts

def trainOneBatch(model, trainLoader, optimizer, device):
    model.train()

    batch = next(iter(trainLoader))

    ct = batch["ct"].float().to(device)
    pet = batch["pet"].float().to(device)
    clinical = batch["clinical"].float().to(device)

    outputs = model(ct, pet, clinical)
    loss, lossParts = computeLoss(outputs, batch, device)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("One batch training worked.")
    print(lossParts)

def trainOneEpoch(model, trainLoader, optimizer, device):
    model.train()

    totalLoss = 0.0
    batchCount = 0

    for batch in trainLoader:
        ct = batch["ct"].float().to(device)
        pet = batch["pet"].float().to(device)
        clinical = batch["clinical"].float().to(device)

        outputs = model(ct, pet, clinical)
        loss, lossParts = computeLoss(outputs, batch, device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        totalLoss += loss.item()
        batchCount += 1

    return totalLoss / batchCount

def validateOneEpoch(model, valLoader, device):
    model.eval()

    totalLoss = 0.0
    batchCount = 0

    with torch.no_grad():
        for batch in valLoader:
            ct = batch["ct"].float().to(device)
            pet = batch["pet"].float().to(device)
            clinical = batch["clinical"].float().to(device)

            outputs = model(ct, pet, clinical)
            loss, lossParts = computeLoss(outputs, batch, device)

            totalLoss += loss.item()
            batchCount += 1

    return totalLoss / batchCount

def main():
    device = getDevice()

    print("Using device:", device)
    print("Clinical columns:", CLINICAL_COLUMNS)
    print("Target columns:", TARGET_COLUMNS)

    trainDataset, valDataset = makeDatasets(foldIndex=0)
    trainLoader, valLoader = makeLoaders(trainDataset, valDataset, batchSize=1)

    model = FusionModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    trainOneBatch(model, trainLoader, optimizer, device)

    trainLoss = trainOneEpoch(model, trainLoader, optimizer, device)
    valLoss = validateOneEpoch(model, valLoader, device)

    print("Train loss:", trainLoss)
    print("Val loss:", valLoss)


if __name__ == "__main__":
    main()
