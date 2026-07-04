import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.func import jvp

from meanflow import MeanFlowBatch, make_meanflow_batch, meanflow_loss, one_step_sample
from model import TinyTimeConditionedCNN
from utils import EMA, all_finite, append_loss_csv, load_image, save_image, save_image_grid, save_loss_curve, set_seed


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


def save_checkpoint(path: Path, step: int, model, optimizer, args, sample_mse: float, ema: "EMA | None" = None) -> None:
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "ema": ema.state_dict() if ema is not None else None,
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
    parser = argparse.ArgumentParser(description="Minimal MeanFlow overfit training prototype.")
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        nargs="+",
        help="Path(s) to one or more fixed RGB images to overfit, e.g. --images a.png b.png c.png.",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="AdamW weight decay. Default 0 since the goal here is to overfit exactly, not regularize.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-every", type=int, default=200)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--equal-time-probability", type=float, default=0.1)
    parser.add_argument("--endpoint-probability", type=float, default=0.25)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--time-dim", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Max gradient norm. Set to 0 to disable.")
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to a checkpoint to warm-start from (e.g. checkpoint_best.pt).",
    )
    parser.add_argument(
        "--no-lr-decay",
        action="store_true",
        help="Use a constant --lr instead of cosine decay. Useful when warm-starting from --resume-from.",
    )
    parser.add_argument(
        "--lr-decay-steps",
        type=int,
        default=None,
        help="Cosine decay reaches its floor after this many steps. Defaults to --steps. Set higher than "
        "--steps to decay more slowly (training ends before the schedule bottoms out).",
    )
    parser.add_argument(
        "--adaptive-lr",
        action="store_true",
        help="Use ReduceLROnPlateau on sample_mse instead of a fixed schedule. Overrides --no-lr-decay "
        "and --lr-decay-steps: the LR only drops when sample_mse actually stalls, instead of on a preset timetable.",
    )
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=200,
        help="Steps with no sample_mse improvement before --adaptive-lr cuts the LR.",
    )
    parser.add_argument(
        "--lr-factor",
        type=float,
        default=0.5,
        help="Multiply LR by this factor when --adaptive-lr detects a plateau.",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.999,
        help="Decay for an exponential moving average of weights, used for sampling. Set to 0 to disable.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=10,
        help="Compute the one-step sample / sample_mse only every N steps (still trains every step). "
        "The sample forward pass + EMA copy is a full extra model evaluation on top of the JVP training "
        "step, so skipping it on most steps meaningfully speeds up training.",
    )
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

    base_images = torch.cat([load_image(path, args.image_size, device) for path in args.images], dim=0)
    num_images = base_images.shape[0]
    repeats = -(-args.batch_size // num_images)  # ceil division
    clean_image = base_images.repeat(repeats, 1, 1, 1)[: args.batch_size]
    print(
        "Loaded images:",
        num_images,
        tuple(base_images.shape),
        f"min={base_images.min().item():.3f}",
        f"max={base_images.max().item():.3f}",
    )

    model = TinyTimeConditionedCNN(
        hidden_channels=args.hidden_channels,
        time_dim=args.time_dim,
        num_blocks=args.num_blocks,
    ).to(device=device, dtype=torch.float32)
    print(f"Model trainable parameters: {count_trainable_parameters(model):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location=device)
        if checkpoint.get("ema") is not None:
            # The saved "model" is the raw weights at that instant, which can be noisier than
            # the EMA-smoothed weights that actually produced the checkpointed sample_mse.
            # Continue training from the smoothed point, not the noisy one.
            model.load_state_dict(checkpoint["ema"])
            print(
                f"Resumed model weights from EMA in {args.resume_from} "
                f"(saved at step {checkpoint['step']}, sample_mse={checkpoint['sample_mse']:.6f})"
            )
        else:
            model.load_state_dict(checkpoint["model"])
            print(
                f"Resumed model weights from {args.resume_from} "
                f"(saved at step {checkpoint['step']}, sample_mse={checkpoint['sample_mse']:.6f})"
            )
        # Restore Adam's per-parameter momentum/variance too. Without this, every resume
        # started from a cold, zero-initialized optimizer state; Adam's bias-correction then
        # takes unusually large effective steps for the first several hundred steps regardless
        # of how small --lr is set, which was producing large loss spikes right after resuming.
        optimizer.load_state_dict(checkpoint["optimizer"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr

    ema = EMA(model, args.ema_decay) if args.ema_decay > 0 else None
    if ema is not None and args.resume_from and checkpoint.get("ema") is not None:
        ema.load_state_dict(checkpoint["ema"])
        print("Resumed EMA weights too.")

    eval_model = None
    if ema is not None:
        eval_model = TinyTimeConditionedCNN(
            hidden_channels=args.hidden_channels,
            time_dim=args.time_dim,
            num_blocks=args.num_blocks,
        ).to(device=device, dtype=torch.float32)
        for parameter in eval_model.parameters():
            parameter.requires_grad_(False)

    plateau_scheduler = None
    if args.adaptive_lr:
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience
        )
        scheduler = None
    elif args.no_lr_decay:
        scheduler = None
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.lr_decay_steps or args.steps, eta_min=args.lr * 0.01
        )

    run_sanity_checks(model, optimizer, clean_image)

    fixed_eval_noise = torch.randn(num_images, 3, args.image_size, args.image_size, device=device)
    torch.save(fixed_eval_noise.detach().cpu(), output_dir / "fixed_eval_noise.pt")
    for index in range(num_images):
        save_image(base_images[index : index + 1], str(output_dir / f"clean_{index}.png"))
        save_image(fixed_eval_noise[index : index + 1], str(output_dir / f"fixed_noise_{index}.png"))
    save_image_grid(base_images, str(output_dir / "clean_grid.png"))

    loss_csv = output_dir / "loss_history.csv"
    best_sample_mse = float("inf")

    for step in range(1, args.steps + 1):
        batch = make_meanflow_batch(clean_image, args.equal_time_probability, args.endpoint_probability)
        result = meanflow_loss(model, batch)

        optimizer.zero_grad(set_to_none=True)
        result.loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update(model)

        loss_value = result.loss.item()

        needs_eval = (
            step == 1
            or step % args.eval_every == 0
            or step % 50 == 0
            or step % args.sample_every == 0
            or step % args.checkpoint_every == 0
            or step == args.steps
        )

        if needs_eval:
            with torch.no_grad():
                if ema is not None:
                    ema.copy_to(eval_model)
                    sample = one_step_sample(eval_model, fixed_eval_noise)
                else:
                    sample = one_step_sample(model, fixed_eval_noise)
                sample_mse = F.mse_loss(sample, base_images).item()
                per_image_mse = F.mse_loss(sample, base_images, reduction="none").mean(dim=(1, 2, 3)).tolist()

            if plateau_scheduler is not None:
                plateau_scheduler.step(sample_mse)

            append_loss_csv(str(loss_csv), step, loss_value, sample_mse, per_image_mse)

            if sample_mse < best_sample_mse:
                best_sample_mse = sample_mse
                save_checkpoint(output_dir / "checkpoint_best.pt", step, model, optimizer, args, sample_mse, ema)
                for index in range(num_images):
                    save_image(sample[index : index + 1], str(output_dir / f"sample_best_{index}.png"))
                save_image_grid(sample, str(output_dir / "sample_best_grid.png"))

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
                current_lr = optimizer.param_groups[0]["lr"]
                per_image_str = " ".join(f"img{i}={mse:.4f}" for i, mse in enumerate(per_image_mse))
                print(
                    f"step={step:04d}",
                    f"lr={current_lr:.6f}",
                    f"loss={loss_value:.6f}",
                    f"sample_mse={sample_mse:.6f}",
                    f"[{per_image_str}]",
                    f"|u|={result.mean_velocity.abs().mean().item():.4f}",
                    f"|jvp|={result.jvp_term.abs().mean().item():.4f}",
                    f"finite={finite}",
                )

            if step == 1 or step % args.sample_every == 0:
                save_image_grid(sample, str(output_dir / f"sample_step_{step:04d}.png"))

            if step % args.checkpoint_every == 0 or step == args.steps:
                save_checkpoint(output_dir / "checkpoint.pt", step, model, optimizer, args, sample_mse, ema)

    save_loss_curve(str(loss_csv), str(output_dir / "loss_curve.png"))

    print(f"Best sample_mse={best_sample_mse:.6f}")
    print(f"Done. Outputs saved in {output_dir}")


if __name__ == "__main__":
    main()
