import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.func import jvp

from meanflow import MeanFlowBatch, make_meanflow_batch, meanflow_loss, one_step_sample
from model import TinyTimeConditionedCNN
from utils import all_finite, append_loss_csv, load_image, save_image, set_seed


def analytical_jvp_check(device: torch.device) -> None:
    x = torch.tensor([2.0], device=device)
    t = torch.tensor([3.0], device=device)

    def simple_function(x_value, t_value):
        return x_value * t_value**2

    value, tangent = jvp(
        simple_function,
        (x, t),
        (torch.ones_like(x), torch.ones_like(t)),
    )
    expected_value = torch.tensor([18.0], device=device)
    expected_tangent = t**2 + 2.0 * x * t

    assert torch.allclose(value, expected_value), f"value check failed: {value}"
    assert torch.allclose(tangent, expected_tangent), f"JVP check failed: {tangent}"


def first_parameter_snapshot(model: torch.nn.Module) -> torch.Tensor:
    return next(parameter for parameter in model.parameters() if parameter.requires_grad).detach().clone()


def save_checkpoint(path: Path, step: int, model, optimizer, args, sample_mse: float) -> None:
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "sample_mse": sample_mse,
        },
        path,
    )


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def run_sanity_checks(model, optimizer, clean_image: torch.Tensor) -> None:
    model.train()
    batch_size = clean_image.shape[0]
    batch = make_meanflow_batch(clean_image, equal_time_probability=1.0, endpoint_probability=0.0)
    result = meanflow_loss(model, batch)

    assert result.mean_velocity.shape == clean_image.shape
    assert result.jvp_term.shape == clean_image.shape
    assert result.target.shape == clean_image.shape
    assert all_finite(batch.z_t, batch.velocity, result.mean_velocity, result.jvp_term, result.target, result.loss)

    # When r == t, the MeanFlow target becomes velocity because (t - r) is zero.
    assert torch.allclose(result.target, batch.velocity, atol=1e-5)
    assert not result.target.requires_grad

    before = first_parameter_snapshot(model)
    optimizer.zero_grad(set_to_none=True)
    result.loss.backward()

    finite_nonzero_gradient = False
    for parameter in model.parameters():
        if parameter.grad is not None:
            gradient = parameter.grad
            if torch.isfinite(gradient).all().item() and gradient.abs().sum().item() > 0.0:
                finite_nonzero_gradient = True
                break
    assert finite_nonzero_gradient, "no finite nonzero gradient found"

    optimizer.step()
    after = first_parameter_snapshot(model)
    assert not torch.allclose(before, after), "optimizer step did not change the first parameter"

    print("Sanity checks passed.")


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal one-image MeanFlow training prototype.")
    parser.add_argument("--image", type=str, required=True, help="Path to one RGB image.")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-every", type=int, default=200)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--equal-time-probability", type=float, default=0.1)
    parser.add_argument("--endpoint-probability", type=float, default=0.25)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--time-dim", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type != "cuda":
        print("CUDA is not available; this prototype will run, but the final assignment requires GPU.")

    analytical_jvp_check(device)
    print("Analytical JVP check passed.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_image = load_image(args.image, args.image_size, device)
    clean_image = clean_image.repeat(args.batch_size, 1, 1, 1)
    print(
        "Loaded image:",
        tuple(clean_image.shape),
        f"min={clean_image.min().item():.3f}",
        f"max={clean_image.max().item():.3f}",
    )

    model = TinyTimeConditionedCNN(
        hidden_channels=args.hidden_channels,
        time_dim=args.time_dim,
        num_blocks=args.num_blocks,
    ).to(device=device, dtype=torch.float32)
    print(f"Model trainable parameters: {count_trainable_parameters(model):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.01)

    run_sanity_checks(model, optimizer, clean_image)

    fixed_eval_noise = torch.randn(1, 3, args.image_size, args.image_size, device=device)
    torch.save(fixed_eval_noise.detach().cpu(), output_dir / "fixed_eval_noise.pt")
    save_image(clean_image[:1], str(output_dir / "clean.png"))
    save_image(fixed_eval_noise, str(output_dir / "fixed_noise.png"))

    loss_csv = output_dir / "loss_history.csv"
    best_sample_mse = float("inf")

    for step in range(1, args.steps + 1):
        batch = make_meanflow_batch(clean_image, args.equal_time_probability, args.endpoint_probability)
        result = meanflow_loss(model, batch)

        optimizer.zero_grad(set_to_none=True)
        result.loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            sample = one_step_sample(model, fixed_eval_noise)
            sample_mse = F.mse_loss(sample, clean_image[:1]).item()

        loss_value = result.loss.item()
        append_loss_csv(str(loss_csv), step, loss_value, sample_mse)

        if sample_mse < best_sample_mse:
            best_sample_mse = sample_mse
            save_checkpoint(output_dir / "checkpoint_best.pt", step, model, optimizer, args, sample_mse)
            save_image(sample, str(output_dir / "sample_best.png"))

        if step == 1 or step % 50 == 0:
            finite = all_finite(
                batch.z_t,
                batch.velocity,
                result.mean_velocity,
                result.jvp_term,
                result.target,
                result.loss,
                sample,
            )
            print(
                f"step={step:04d}",
                f"lr={scheduler.get_last_lr()[0]:.6f}",
                f"loss={loss_value:.6f}",
                f"sample_mse={sample_mse:.6f}",
                f"|u|={result.mean_velocity.abs().mean().item():.4f}",
                f"|jvp|={result.jvp_term.abs().mean().item():.4f}",
                f"finite={finite}",
            )

        if step == 1 or step % args.sample_every == 0:
            save_image(sample, str(output_dir / f"sample_step_{step:04d}.png"))

        if step % args.checkpoint_every == 0 or step == args.steps:
            save_checkpoint(output_dir / "checkpoint.pt", step, model, optimizer, args, sample_mse)

    print(f"Best sample_mse={best_sample_mse:.6f}")
    print(f"Done. Outputs saved in {output_dir}")


if __name__ == "__main__":
    main()
