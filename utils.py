import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# open and convert an image
def load_image(path: str, image_size: int, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    array = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    tensor = tensor * 2.0 - 1.0

    # return a tensor
    return tensor.unsqueeze(0).to(device=device, dtype=torch.float32)


def save_image(tensor: torch.Tensor, path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    image = tensor.detach().float().cpu().clamp(-1.0, 1.0)
    image = (image + 1.0) * 0.5
    image = (image[0].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(image).save(path)


def append_loss_csv(path: str, step: int, loss: float, sample_mse: float) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if needs_header:
            writer.writerow(["step", "loss", "sample_mse"])
        writer.writerow([step, loss, sample_mse])


def all_finite(*tensors: torch.Tensor) -> bool:
    return all(torch.isfinite(tensor).all().item() for tensor in tensors)
