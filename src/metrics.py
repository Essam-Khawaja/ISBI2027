import math

import numpy as np
import torch


class MetricAccumulator:
    def __init__(self, task_specs):
        self.task_specs = task_specs
        self.values = {
            task["name"]: {
                "predictions": [],
                "targets": [],
            }
            for task in task_specs
        }

    def update(self, outputs, batch):
        for task in self.task_specs:
            name = task["name"]
            mask = batch["target_masks"][name].bool()

            if not mask.any().item():
                continue

            output = outputs[name][mask].detach().cpu()
            target = batch["targets"][name][mask].detach().cpu()

            if task["kind"] == "multiclass":
                prediction = torch.argmax(output, dim=1)
            elif task["kind"] == "binary":
                prediction = torch.sigmoid(output.squeeze(1))
            elif task["kind"] == "regression":
                prediction = output.squeeze(1)

                if task.get("transform") == "log1p":
                    target = torch.log1p(target.float())
            else:
                raise ValueError(f"Unknown task kind: {task['kind']}")

            self.values[name]["predictions"].extend(prediction.numpy().tolist())
            self.values[name]["targets"].extend(target.numpy().tolist())

    def compute(self):
        metrics = {}

        for task in self.task_specs:
            name = task["name"]
            predictions = self.values[name]["predictions"]
            targets = self.values[name]["targets"]

            metrics[f"{name}_support"] = len(targets)

            if not targets:
                continue

            if task["kind"] == "multiclass":
                metrics.update(multiclass_metrics(name, predictions, targets, task["output_dim"]))
            elif task["kind"] == "binary":
                metrics.update(binary_metrics(name, predictions, targets))
            elif task["kind"] == "regression":
                metrics.update(regression_metrics(name, predictions, targets))

        return metrics


def multiclass_metrics(name, predictions, targets, num_classes):
    predictions = [int(value) for value in predictions]
    targets = [int(value) for value in targets]
    correct = sum(int(pred == target) for pred, target in zip(predictions, targets))
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    for prediction, target in zip(predictions, targets):
        if 0 <= target < num_classes:
            class_total[target] += 1

            if prediction == target:
                class_correct[target] += 1

    return {
        f"{name}_accuracy": safe_divide(correct, len(targets)),
        f"{name}_balanced_accuracy": balanced_accuracy(class_correct, class_total),
    }


def binary_metrics(name, probabilities, targets):
    labels = [int(round(float(value))) for value in targets]
    predictions = [int(float(probability) >= 0.5) for probability in probabilities]
    correct = sum(int(pred == label) for pred, label in zip(predictions, labels))
    class_correct = [0, 0]
    class_total = [0, 0]

    for prediction, label in zip(predictions, labels):
        if label in (0, 1):
            class_total[label] += 1

            if prediction == label:
                class_correct[label] += 1

    return {
        f"{name}_accuracy": safe_divide(correct, len(labels)),
        f"{name}_balanced_accuracy": balanced_accuracy(class_correct, class_total),
        f"{name}_auroc": binary_auroc(probabilities, labels),
    }


def regression_metrics(name, predictions, targets):
    predictions = np.asarray(predictions, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    errors = predictions - targets

    return {
        f"{name}_mae": float(np.mean(np.abs(errors))),
        f"{name}_rmse": float(np.sqrt(np.mean(errors ** 2))),
    }


def safe_divide(numerator, denominator):
    if denominator == 0:
        return math.nan

    return numerator / denominator


def balanced_accuracy(class_correct, class_total):
    accuracies = [
        safe_divide(correct, total)
        for correct, total in zip(class_correct, class_total)
        if total > 0
    ]

    if not accuracies:
        return math.nan

    return sum(accuracies) / len(accuracies)


def binary_auroc(scores, labels):
    pairs = sorted(
        (float(score), int(label))
        for score, label in zip(scores, labels)
        if int(label) in (0, 1)
    )
    pos_count = sum(label == 1 for _, label in pairs)
    neg_count = sum(label == 0 for _, label in pairs)

    if pos_count == 0 or neg_count == 0:
        return math.nan

    ranks = [0.0] * len(pairs)
    index = 0

    while index < len(pairs):
        end = index + 1

        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1

        average_rank = (index + 1 + end) / 2.0

        for rank_index in range(index, end):
            ranks[rank_index] = average_rank

        index = end

    pos_rank_sum = sum(rank for rank, (_, label) in zip(ranks, pairs) if label == 1)
    auc = (pos_rank_sum - pos_count * (pos_count + 1) / 2.0) / (pos_count * neg_count)

    return float(auc)

