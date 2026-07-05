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

def load_image(path: str, image_size: int, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    array = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    tensor = tensor * 2.0 - 1.0
    return tensor.unsqueeze(0).to(device=device, dtype=torch.float32)


def save_image(tensor: torch.Tensor, path: str) -> None:
    """Save the first image of a (N, C, H, W) batch as a PNG."""
    save_image_grid(tensor[:1], path)


def save_image_grid(tensor: torch.Tensor, path: str) -> None:
    """Save a batch of images (N, C, H, W) side by side in one PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    images = tensor.detach().float().cpu().clamp(-1.0, 1.0)
    images = (images + 1.0) * 0.5
    num_images, _, height, width = images.shape
    grid = Image.new("RGB", (width * num_images, height))
    for index in range(num_images):
        array = (images[index].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        grid.paste(Image.fromarray(array), (index * width, 0))
    grid.save(path)


def nearest_image_eval(samples: torch.Tensor, targets: torch.Tensor):
    """Assignment-free scoring of one-step samples against the training images.

    MeanFlow never promises which noise maps to which image -- the learned flow picks its
    own noise->image basins -- so each sample is scored against its *nearest* target, not
    a fixed pairing. Returns:
      pairwise_mse: (num_samples, num_targets) MSE between every sample and every target
      nearest_mse, nearest_image: each sample's distance to / index of its nearest target
      order: sample indices sorted by (nearest target, error), so grids read basin by basin
    """
    pairwise_mse = ((samples[:, None] - targets[None, :]) ** 2).mean(dim=(2, 3, 4))
    nearest_mse, nearest_image = pairwise_mse.min(dim=1)
    order = sorted(
        range(samples.shape[0]), key=lambda i: (nearest_image[i].item(), nearest_mse[i].item())
    )
    return pairwise_mse, nearest_mse, nearest_image, order


def append_loss_csv(path: str, step: int, loss: float, sample_mse: float, per_image_mse=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists()
    per_image_mse = per_image_mse or []
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if needs_header:
            header = ["step", "loss", "sample_mse"] + [f"sample_mse_{i}" for i in range(len(per_image_mse))]
            writer.writerow(header)
        writer.writerow([step, loss, sample_mse, *per_image_mse])


def all_finite(*tensors: torch.Tensor) -> bool:
    return all(torch.isfinite(tensor).all().item() for tensor in tensors)


class EMA:
    """Exponential moving average of a model's parameters, used only for sampling."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {name: param.detach().clone() for name, param in model.state_dict().items()}

    def update(self, model: torch.nn.Module) -> None:
        for name, param in model.state_dict().items():
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def copy_to(self, model: torch.nn.Module) -> None:
        model.load_state_dict(self.shadow)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict) -> None:
        self.shadow = {name: value.clone() for name, value in state_dict.items()}


def save_loss_curve(csv_path: str, out_path: str) -> None:
    """Read a loss_history.csv and save a loss/sample_mse-vs-step plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps, losses, sample_mses = [], [], []
    per_image_columns = []
    with Path(csv_path).open() as handle:
        reader = csv.DictReader(handle)
        per_image_columns = [name for name in (reader.fieldnames or []) if name.startswith("sample_mse_")]
        per_image_series = {name: [] for name in per_image_columns}
        for row in reader:
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))
            sample_mses.append(float(row["sample_mse"]))
            for name in per_image_columns:
                per_image_series[name].append(float(row[name]))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, losses, label="loss", alpha=0.8)
    ax.plot(steps, sample_mses, label="sample_mse", color="black", linewidth=2)
    for name in per_image_columns:
        ax.plot(steps, per_image_series[name], label=name, alpha=0.6, linestyle="--")
    ax.set_xlabel("step")
    ax.set_yscale("log")
    ax.set_title("Training curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
