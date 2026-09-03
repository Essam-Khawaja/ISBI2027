from pathlib import Path

import nibabel as nib
import numpy as np
import torch


MODALITY_FILENAMES = {
    "ct": "{case_id}__CT.nii.gz",
    "pet": "{case_id}__PT.nii.gz",
    "mask": "{case_id}.nii.gz",
}


def get_case_paths(data_root, case_id):
    case_dir = Path(data_root) / case_id / "preprocessed"

    return {
        modality: case_dir / template.format(case_id=case_id)
        for modality, template in MODALITY_FILENAMES.items()
    }


def load_nifti_volume(path):
    image = nib.load(str(path))
    volume = image.get_fdata(dtype=np.float32)
    volume = np.asarray(volume, dtype=np.float32)

    # NIfTI arrives as x, y, z. The model receives channels, depth, height, width.
    return np.transpose(volume, (2, 1, 0))


def center_crop_or_pad(volume, target_shape):
    output = volume

    for axis, target_size in enumerate(target_shape):
        current_size = output.shape[axis]

        if current_size > target_size:
            start = (current_size - target_size) // 2
            stop = start + target_size
            slices = [slice(None)] * output.ndim
            slices[axis] = slice(start, stop)
            output = output[tuple(slices)]

    pad_width = []

    for current_size, target_size in zip(output.shape, target_shape):
        total_padding = max(target_size - current_size, 0)
        before = total_padding // 2
        after = total_padding - before
        pad_width.append((before, after))

    if any(before or after for before, after in pad_width):
        output = np.pad(output, pad_width, mode="constant", constant_values=0)

    return output.astype(np.float32, copy=False)


def normalize_volume(volume, mode):
    if mode == "none":
        return volume.astype(np.float32, copy=False)

    if mode != "zscore":
        raise ValueError(f"Unknown image normalization mode: {mode}")

    mean = float(volume.mean())
    std = float(volume.std())

    if std < 1e-6:
        return np.zeros_like(volume, dtype=np.float32)

    return ((volume - mean) / std).astype(np.float32, copy=False)


def load_image_tensor(case_id, data_root, image_config):
    modalities = image_config.get("modalities", ["ct", "pet"])
    crop_shape = image_config.get("crop_shape")
    normalization = image_config.get("normalization", "zscore")
    paths = get_case_paths(data_root, case_id)
    channels = []

    for modality in modalities:
        if modality not in paths:
            raise ValueError(f"Unknown image modality: {modality}")

        path = paths[modality]

        if not path.exists():
            raise FileNotFoundError(f"Missing {modality.upper()} image for {case_id}: {path}")

        volume = load_nifti_volume(path)

        if crop_shape:
            volume = center_crop_or_pad(volume, crop_shape)

        volume = normalize_volume(volume, normalization)
        channels.append(torch.from_numpy(volume))

    # image: [channels, depth, height, width]
    return torch.stack(channels, dim=0).float()

