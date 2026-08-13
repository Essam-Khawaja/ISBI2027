import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import HecktorDataset
from fusionModel import FusionModel
from main import CLINICAL_COLUMNS, DATA_FOLDER, TARGET_COLUMNS, getDevice, getLoadTransform
from main import loadClinicalData, loadSplit, preprocessClinicalData


RELAPSE_INDEX = 0
RFS_INDEX = 1
T_STAGE_INDEX = 2
N_STAGE_INDEX = 3

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

def makeLoaders(trainDataset, valDataset, batchSize, numWorkers):
    trainLoader = DataLoader(
        trainDataset,
        batch_size=batchSize,
        shuffle=True,
        num_workers=numWorkers,
    )

    valLoader = DataLoader(
        valDataset,
        batch_size=batchSize,
        shuffle=False,
        num_workers=numWorkers,
    )

    return trainLoader, valLoader

def makeLossFunctions():
    return {
        "tStage": nn.CrossEntropyLoss(),
        "nStage": nn.CrossEntropyLoss(),
        "rfs": nn.SmoothL1Loss(),
    }

def moveBatchToDevice(batch, device):
    ct = batch["ct"].float().to(device)
    pet = batch["pet"].float().to(device)
    clinical = batch["clinical"].float().to(device)

    return ct, pet, clinical

def addLoss(totalLoss, newLoss):
    if totalLoss is None:
        return newLoss

    return totalLoss + newLoss

def computeLoss(outputs, batch, lossFunctions, device, rfsWeight):
    targets = batch["targets"].to(device)
    targetMask = batch["targetMask"].to(device)

    rfs = targets[:, RFS_INDEX]
    tStage = targets[:, T_STAGE_INDEX]
    nStage = targets[:, N_STAGE_INDEX]

    totalLoss = None
    lossParts = {}

    validT = targetMask[:, T_STAGE_INDEX]
    if validT.any().item():
        tLoss = lossFunctions["tStage"](
            outputs["tStage"][validT],
            tStage[validT].long()
        )

        totalLoss = addLoss(totalLoss, tLoss)
        lossParts["tLoss"] = tLoss.item()

    validN = targetMask[:, N_STAGE_INDEX]
    if validN.any().item():
        nLoss = lossFunctions["nStage"](
            outputs["nStage"][validN],
            nStage[validN].long()
        )

        totalLoss = addLoss(totalLoss, nLoss)
        lossParts["nLoss"] = nLoss.item()

    validRfs = targetMask[:, RFS_INDEX]
    if validRfs.any().item():
        rfsTarget = torch.log1p(rfs[validRfs]).unsqueeze(1)
        rfsLoss = lossFunctions["rfs"](outputs["rfs"][validRfs], rfsTarget)
        weightedRfsLoss = rfsWeight * rfsLoss

        totalLoss = addLoss(totalLoss, weightedRfsLoss)
        lossParts["rfsLoss"] = rfsLoss.item()
        lossParts["weightedRfsLoss"] = weightedRfsLoss.item()

    if totalLoss is None:
        raise RuntimeError("Dude this batch had no usable targets for any loss.")

    lossParts["totalLoss"] = totalLoss.item()

    return totalLoss, lossParts

def trainOneBatch(model, trainLoader, optimizer, lossFunctions, device, rfsWeight):
    model.train()

    batch = next(iter(trainLoader))

    ct, pet, clinical = moveBatchToDevice(batch, device)

    outputs = model(ct, pet, clinical)
    loss, lossParts = computeLoss(outputs, batch, lossFunctions, device, rfsWeight)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("One batch training worked.")
    print(lossParts)

def trainOneEpoch(model, trainLoader, optimizer, lossFunctions, device, rfsWeight, maxBatches=None):
    model.train()

    totalLoss = 0.0
    batchCount = 0

    for batch in trainLoader:
        ct, pet, clinical = moveBatchToDevice(batch, device)

        outputs = model(ct, pet, clinical)
        loss, lossParts = computeLoss(outputs, batch, lossFunctions, device, rfsWeight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        totalLoss += loss.item()
        batchCount += 1

        if maxBatches and batchCount >= maxBatches:
            break

    return totalLoss / batchCount

def validateOneEpoch(model, valLoader, lossFunctions, device, rfsWeight, maxBatches=None):
    model.eval()

    totalLoss = 0.0
    batchCount = 0

    with torch.no_grad():
        for batch in valLoader:
            ct, pet, clinical = moveBatchToDevice(batch, device)

            outputs = model(ct, pet, clinical)
            loss, lossParts = computeLoss(outputs, batch, lossFunctions, device, rfsWeight)

            totalLoss += loss.item()
            batchCount += 1

            if maxBatches and batchCount >= maxBatches:
                break

    return totalLoss / batchCount

def saveCheckpoint(model, optimizer, foldIndex, epoch, trainLoss, valLoss, outputFolder):
    outputFolder.mkdir(parents=True, exist_ok=True)

    checkpointPath = outputFolder / f"fusion_fold{foldIndex}_epoch{epoch}.pt"

    torch.save(
        {
            "foldIndex": foldIndex,
            "epoch": epoch,
            "modelState": model.state_dict(),
            "optimizerState": optimizer.state_dict(),
            "trainLoss": trainLoss,
            "valLoss": valLoss,
            "clinicalColumns": CLINICAL_COLUMNS,
            "targetColumns": TARGET_COLUMNS,
        },
        checkpointPath
    )

    return checkpointPath

def trainFold(args, foldIndex, device):
    print("\n" + "=" * 60)
    print("Starting fold:", foldIndex)
    print("=" * 60)

    trainDataset, valDataset = makeDatasets(foldIndex=foldIndex)
    trainLoader, valLoader = makeLoaders(
        trainDataset=trainDataset,
        valDataset=valDataset,
        batchSize=args.batchSize,
        numWorkers=args.numWorkers
    )

    model = FusionModel().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learningRate,
        weight_decay=args.weightDecay
    )

    lossFunctions = makeLossFunctions()
    bestValLoss = None

    if args.oneBatch:
        trainOneBatch(
            model=model,
            trainLoader=trainLoader,
            optimizer=optimizer,
            lossFunctions=lossFunctions,
            device=device,
            rfsWeight=args.rfsWeight
        )
        return

    for epoch in range(1, args.epochs + 1):
        trainLoss = trainOneEpoch(
            model=model,
            trainLoader=trainLoader,
            optimizer=optimizer,
            lossFunctions=lossFunctions,
            device=device,
            rfsWeight=args.rfsWeight,
            maxBatches=args.maxTrainBatches
        )

        valLoss = validateOneEpoch(
            model=model,
            valLoader=valLoader,
            lossFunctions=lossFunctions,
            device=device,
            rfsWeight=args.rfsWeight,
            maxBatches=args.maxValBatches
        )

        print(
            f"Fold {foldIndex} | epoch {epoch}/{args.epochs} | "
            f"train loss {trainLoss:.4f} | val loss {valLoss:.4f}"
        )

        shouldSave = not args.saveBestOnly or bestValLoss is None or valLoss < bestValLoss

        if shouldSave:
            bestValLoss = valLoss
            checkpointPath = saveCheckpoint(
                model=model,
                optimizer=optimizer,
                foldIndex=foldIndex,
                epoch=epoch,
                trainLoss=trainLoss,
                valLoss=valLoss,
                outputFolder=Path(args.outputFolder)
            )
            print("Saved checkpoint:", checkpointPath)

def parseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--all-folds", dest="allFolds", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", dest="batchSize", type=int, default=1)
    parser.add_argument("--num-workers", dest="numWorkers", type=int, default=0)
    parser.add_argument("--learning-rate", dest="learningRate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", dest="weightDecay", type=float, default=1e-4)
    parser.add_argument("--rfs-weight", dest="rfsWeight", type=float, default=0.1)
    parser.add_argument("--max-train-batches", dest="maxTrainBatches", type=int, default=None)
    parser.add_argument("--max-val-batches", dest="maxValBatches", type=int, default=None)
    parser.add_argument("--output-folder", dest="outputFolder", default="checkpoints")
    parser.add_argument("--save-best-only", dest="saveBestOnly", action="store_true")
    parser.add_argument("--one-batch", dest="oneBatch", action="store_true")

    return parser.parse_args()

def main():
    args = parseArgs()
    device = getDevice()

    print("Using device:", device)
    print("Clinical columns:", CLINICAL_COLUMNS)
    print("Target columns:", TARGET_COLUMNS)
    print("RFS loss weight:", args.rfsWeight)

    if args.allFolds:
        foldIndices = range(5)
    else:
        foldIndices = [args.fold]

    for foldIndex in foldIndices:
        trainFold(args, foldIndex, device)

if __name__ == "__main__":
    main()
