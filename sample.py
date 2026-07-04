"""One-step sample from many fresh noises and measure reproduction assignment-free.

train.py's sample_mse pairs fixed_noise[i] with image[i], but MeanFlow never enforces
that pairing -- the learned marginal flow chooses its own noise->image basins. This
script asks the question the assignment actually cares about: from N random noises,
does each one-step sample land cleanly on *some* training image, and is every
training image produced by at least one noise?
"""

import argparse
from pathlib import Path

import torch

from meanflow import one_step_sample
from model import TinyTimeConditionedCNN
from utils import load_image, save_image_grid, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Assignment-free one-step sampling evaluation.")
    parser.add_argument("--checkpoint", type=str, required=True, help="e.g. outputs/run/checkpoint_best.pt")
    parser.add_argument("--images", type=str, required=True, nargs="+", help="The training images to compare against.")
    parser.add_argument("--num-samples", type=int, default=16, help="Number of fresh noises to sample from.")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234, help="Different from training's default on purpose.")
    parser.add_argument("--output-dir", type=str, default=None, help="Defaults to the checkpoint's directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    saved_args = checkpoint["args"]
    model = TinyTimeConditionedCNN(
        hidden_channels=saved_args["hidden_channels"],
        time_dim=saved_args["time_dim"],
        num_blocks=saved_args["num_blocks"],
    ).to(device=device, dtype=torch.float32)
    # Prefer EMA weights: they are what produced the checkpointed sample_mse.
    state = checkpoint["ema"] if checkpoint.get("ema") is not None else checkpoint["model"]
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded {args.checkpoint} (step {checkpoint['step']}, paired sample_mse={checkpoint['sample_mse']:.6f})")

    targets = torch.cat([load_image(path, args.image_size, device) for path in args.images], dim=0)
    num_images = targets.shape[0]

    noise = torch.randn(args.num_samples, 3, args.image_size, args.image_size, device=device)
    with torch.no_grad():
        samples = one_step_sample(model, noise)

    # (num_samples, num_images) MSE between every sample and every training image.
    pairwise_mse = ((samples[:, None] - targets[None, :]) ** 2).mean(dim=(2, 3, 4))
    nearest_mse, nearest_image = pairwise_mse.min(dim=1)

    print(f"\nPer-sample nearest training image ({args.num_samples} fresh noises):")
    for index in range(args.num_samples):
        print(f"  sample {index:02d} -> img{nearest_image[index].item()}  mse={nearest_mse[index].item():.4f}")

    print("\nCoverage (assignment-free):")
    for image_index in range(num_images):
        count = int((nearest_image == image_index).sum().item())
        best = pairwise_mse[:, image_index].min().item()
        print(f"  img{image_index}: nearest for {count}/{args.num_samples} samples, best mse over all samples={best:.4f}")
    print(f"  mean nearest-image mse: {nearest_mse.mean().item():.4f}")

    output_dir = Path(args.output_dir or Path(args.checkpoint).parent)
    # Sort the grid by (assigned image, error) so basins read left to right.
    order = sorted(range(args.num_samples), key=lambda i: (nearest_image[i].item(), nearest_mse[i].item()))
    save_image_grid(samples[order], str(output_dir / "samples_many_noises.png"))
    save_image_grid(targets, str(output_dir / "samples_many_noises_targets.png"))
    print(f"\nSaved grid (sorted by nearest image) to {output_dir}/samples_many_noises.png")


if __name__ == "__main__":
    main()
