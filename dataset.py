"""Minimal image-folder data pipeline for the 10-class Imagenette bonus.

Deliberately avoids torchvision: the whole pipeline is ~40 lines of PIL +
torch.utils.data, and the repo's other deps stay unchanged.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}


class ImageFolderDataset(Dataset):
    """Every image under root, shorter side resized to image_size, center-cropped square,
    random horizontal flip, scaled to [-1, 1] -- the same range utils.load_image produces.

    Labels are ignored: the model is unconditional, so the class subfolders are just a
    directory layout, not supervision.
    """

    def __init__(self, root: str, image_size: int):
        self.paths = sorted(
            path for path in Path(root).rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise FileNotFoundError(f"no images with extensions {IMAGE_EXTENSIONS} under {root}")
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.paths[index]).convert("RGB")
        scale = self.image_size / min(image.size)
        image = image.resize(
            (round(image.width * scale), round(image.height * scale)), Image.Resampling.BICUBIC
        )
        left = (image.width - self.image_size) // 2
        top = (image.height - self.image_size) // 2
        image = image.crop((left, top, left + self.image_size, top + self.image_size))
        # torch.rand (not python random) so DataLoader workers are seeded independently.
        if torch.rand(()) < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1) * 2.0 - 1.0


def make_dataloader(root: str, image_size: int, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        ImageFolderDataset(root, image_size),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def infinite_batches(loader: DataLoader):
    """Loop the dataloader forever; the training loop counts steps, not epochs."""
    while True:
        for images in loader:
            yield images
