import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import apply_cli_overrides, load_config
from src.data import make_datasets
from src.losses import compute_loss, make_loss_functions
from src.metrics import MetricAccumulator
from src.models import build_model
from src.utils import (
    append_metrics_csv,
    batch_to_device,
    count_parameters,
    get_device,
    make_run_dir,
    seed_worker,
    set_seed,
    write_json,
)


def main():
    args = parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    train(config)


def train(config):
    seed = int(config.get("seed", 42))
    set_seed(seed, deterministic=bool(config.get("deterministic", True)))
    device = get_device(config.get("device", {}).get("name", "auto"))

    train_dataset, val_dataset, data_info = make_datasets(config)
    model = build_model(
        config=config,
        tabular_dim=len(data_info["feature_columns"]),
        task_specs=data_info["task_specs"],
    ).to(device)

    training_config = config.get("training", {})
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training_config.get("batch_size", 1)),
        shuffle=True,
        num_workers=int(training_config.get("num_workers", 0)),
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(training_config.get("batch_size", 1)),
        shuffle=False,
        num_workers=int(training_config.get("num_workers", 0)),
        worker_init_fn=seed_worker,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 1e-4)),
        weight_decay=float(training_config.get("weight_decay", 1e-4)),
    )
    loss_functions = make_loss_functions(data_info["task_specs"])
    loss_weights = config.get("loss_weights", {})
    run_dir = make_run_dir(config)

    save_run_setup(run_dir, config, data_info, model)
    print(f"Run directory: {run_dir}")
    print(f"Device: {device}")
    print(f"Model: {config.get('model', {}).get('name')}")
    print(f"Train patients: {len(train_dataset)}")
    print(f"Validation patients: {len(val_dataset)}")
    print(f"Tabular features: {len(data_info['feature_columns'])}")
    print(f"Trainable parameters: {count_parameters(model):,}")

    best_val_loss = None
    final_metrics = {}
    epochs = int(training_config.get("epochs", 1))
    max_train_batches = none_if_non_positive(training_config.get("max_train_batches", 0))
    max_val_batches = none_if_non_positive(training_config.get("max_val_batches", 0))

    for epoch in range(1, epochs + 1):
        train_loss, _ = run_epoch(
            model=model,
            loader=train_loader,
            task_specs=data_info["task_specs"],
            loss_functions=loss_functions,
            loss_weights=loss_weights,
            device=device,
            optimizer=optimizer,
            max_batches=max_train_batches,
        )
        val_loss, val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            task_specs=data_info["task_specs"],
            loss_functions=loss_functions,
            loss_weights=loss_weights,
            device=device,
            optimizer=None,
            max_batches=max_val_batches,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **val_metrics,
        }
        append_metrics_csv(run_dir / "metrics.csv", row)
        final_metrics = row

        print(format_epoch(epoch, epochs, row))

        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                run_dir=run_dir,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_metrics=val_metrics,
                config=config,
                data_info=data_info,
            )

    write_json(run_dir / "final_metrics.json", final_metrics)
    print(f"Best checkpoint: {run_dir / 'best_checkpoint.pt'}")
    return run_dir


def run_epoch(
    model,
    loader,
    task_specs,
    loss_functions,
    loss_weights,
    device,
    optimizer=None,
    max_batches=None,
):
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    batch_count = 0
    metrics = MetricAccumulator(task_specs)

    grad_context = torch.enable_grad() if is_training else torch.no_grad()

    with grad_context:
        for batch_index, batch in enumerate(loader, start=1):
            batch = batch_to_device(batch, device)
            outputs = model(
                image=batch.get("image"),
                tabular=batch.get("tabular"),
            )
            loss, _ = compute_loss(
                outputs=outputs,
                batch=batch,
                task_specs=task_specs,
                loss_functions=loss_functions,
                loss_weights=loss_weights,
            )

            if loss is None:
                continue

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += float(loss.detach().cpu().item())
            batch_count += 1
            metrics.update(outputs, batch)

            if max_batches is not None and batch_index >= max_batches:
                break

    if batch_count == 0:
        raise RuntimeError("No batches produced a usable loss.")

    return total_loss / batch_count, metrics.compute()


def save_run_setup(run_dir, config, data_info, model):
    config_text = config.get("_config_text")

    if config_text:
        (run_dir / "config.toml").write_text(config_text)

    resolved_config = {
        key: value
        for key, value in config.items()
        if not key.startswith("_")
    }
    write_json(run_dir / "resolved_config.json", resolved_config)
    write_json(run_dir / "split_ids.json", data_info["split"])
    write_json(run_dir / "data_info.json", {
        key: value
        for key, value in data_info.items()
        if key != "split"
    })
    write_json(run_dir / "model_info.json", {
        "trainable_parameters": count_parameters(model),
        "model": str(model),
    })


def save_checkpoint(
    run_dir,
    model,
    optimizer,
    epoch,
    train_loss,
    val_loss,
    val_metrics,
    config,
    data_info,
):
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_metrics": val_metrics,
        "feature_columns": data_info["feature_columns"],
        "task_specs": data_info["task_specs"],
        "config": {
            key: value
            for key, value in config.items()
            if not key.startswith("_")
        },
    }
    torch.save(checkpoint, run_dir / "best_checkpoint.pt")


def format_epoch(epoch, epochs, row):
    metric_parts = [
        f"{key}={value:.4f}"
        for key, value in row.items()
        if key not in {"epoch"} and isinstance(value, float)
    ]
    return f"Epoch {epoch}/{epochs} | " + " | ".join(metric_parts)


def none_if_non_positive(value):
    if value is None:
        return None

    value = int(value)

    if value <= 0:
        return None

    return value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
