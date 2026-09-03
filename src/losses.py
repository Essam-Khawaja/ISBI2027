import torch
from torch import nn


def make_loss_functions(task_specs):
    loss_functions = {}

    for task in task_specs:
        if task["kind"] == "multiclass":
            loss_functions[task["name"]] = nn.CrossEntropyLoss()
        elif task["kind"] == "binary":
            loss_functions[task["name"]] = nn.BCEWithLogitsLoss()
        elif task["kind"] == "regression":
            loss_functions[task["name"]] = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown task kind: {task['kind']}")

    return loss_functions


def compute_loss(outputs, batch, task_specs, loss_functions, loss_weights):
    total_loss = None
    loss_parts = {}

    for task in task_specs:
        name = task["name"]
        mask = batch["target_masks"][name].bool()

        if not mask.any().item():
            continue

        output = outputs[name][mask]
        target = batch["targets"][name][mask]
        raw_loss = compute_task_loss(
            output=output,
            target=target,
            task=task,
            loss_function=loss_functions[name],
        )
        weight = float(loss_weights.get(name, 1.0))
        weighted_loss = raw_loss * weight

        total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
        loss_parts[f"{name}_loss"] = float(raw_loss.detach().cpu().item())
        loss_parts[f"{name}_weighted_loss"] = float(weighted_loss.detach().cpu().item())

    if total_loss is None:
        return None, loss_parts

    loss_parts["total_loss"] = float(total_loss.detach().cpu().item())

    return total_loss, loss_parts


def compute_task_loss(output, target, task, loss_function):
    if task["kind"] == "multiclass":
        return loss_function(output, target.long())

    if task["kind"] == "binary":
        return loss_function(output.squeeze(1), target.float())

    if task["kind"] == "regression":
        if task.get("transform") == "log1p":
            target = torch.log1p(target.float())

        return loss_function(output.squeeze(1), target.float())

    raise ValueError(f"Unknown task kind: {task['kind']}")

