import argparse

import torch
from torch.utils.data import DataLoader

from src.config import apply_cli_overrides, load_config
from src.data import make_datasets
from src.losses import compute_loss, make_loss_functions
from src.models import build_model
from src.utils import batch_to_device, count_parameters, get_device, seed_worker, set_seed


def main():
    args = parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    smoke_check(config, require_images=args.require_images)


def smoke_check(config, require_images=False):
    seed = int(config.get("seed", 42))
    set_seed(seed, deterministic=bool(config.get("deterministic", True)))
    device = get_device(config.get("device", {}).get("name", "auto"))

    train_dataset, val_dataset, data_info = make_datasets(
        config,
        allow_missing_images=not require_images,
    )
    model = build_model(
        config=config,
        tabular_dim=len(data_info["feature_columns"]),
        task_specs=data_info["task_specs"],
    ).to(device)
    model.eval()

    loader = DataLoader(
        train_dataset,
        batch_size=min(int(config.get("training", {}).get("batch_size", 1)), 2),
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker,
    )
    batch = next(iter(loader))

    if data_info["requires_images"] and "image" not in batch:
        batch["image"] = make_fake_image_batch(config, batch)
        print("Image data root not found; used a fake image tensor for model wiring.")

    batch = batch_to_device(batch, device)

    with torch.no_grad():
        outputs = model(
            image=batch.get("image"),
            tabular=batch.get("tabular"),
        )
        loss_functions = make_loss_functions(data_info["task_specs"])
        loss, loss_parts = compute_loss(
            outputs=outputs,
            batch=batch,
            task_specs=data_info["task_specs"],
            loss_functions=loss_functions,
            loss_weights=config.get("loss_weights", {}),
        )

    print("Smoke check passed.")
    print(f"Device: {device}")
    print(f"Model: {config.get('model', {}).get('name')}")
    print(f"Train patients: {len(train_dataset)}")
    print(f"Validation patients: {len(val_dataset)}")
    print(f"Tabular features: {len(data_info['feature_columns'])}")
    print(f"Images available: {data_info['images_available']}")
    print(f"Trainable parameters: {count_parameters(model):,}")

    if "tabular" in batch:
        print(f"Tabular batch shape: {tuple(batch['tabular'].shape)}")

    if "image" in batch:
        print(f"Image batch shape: {tuple(batch['image'].shape)}")

    for name, output in outputs.items():
        print(f"Output {name}: {tuple(output.shape)}")

    if loss is not None:
        print(f"Loss: {float(loss.detach().cpu().item()):.4f}")
        print(f"Loss parts: {loss_parts}")


def make_fake_image_batch(config, batch):
    batch_size = len(batch["case_id"])
    image_config = config.get("image", {})
    channels = len(image_config.get("modalities", ["ct", "pet"]))
    crop_shape = image_config.get("crop_shape", [128, 128, 128])

    return torch.zeros(
        (batch_size, channels, *crop_shape),
        dtype=torch.float32,
    )


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
    parser.add_argument("--require-images", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()

