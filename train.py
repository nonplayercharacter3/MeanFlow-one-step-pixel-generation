import argparse
from pathlib import Path

import torch
from torch.func import jvp

from meanflow import make_meanflow_batch, meanflow_loss, one_step_sample
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


def parameter_snapshot(model: torch.nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in model.parameters() if parameter.requires_grad]


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
    batch = make_meanflow_batch(clean_image, equal_time_probability=1.0, endpoint_probability=0.0)
    result = meanflow_loss(model, batch)

    assert result.mean_velocity.shape == clean_image.shape
    assert result.jvp_term.shape == clean_image.shape
    assert result.target.shape == clean_image.shape
    assert all_finite(batch.z_t, batch.velocity, result.mean_velocity, result.jvp_term, result.target, result.loss)

    # When r == t, the MeanFlow target becomes velocity because (t - r) is zero.
    assert torch.allclose(result.target, batch.velocity, atol=1e-5)
    assert not result.target.requires_grad

    before = parameter_snapshot(model)
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
    after = parameter_snapshot(model)
    # Not every parameter moves on step 1: the zero-initialized FiLM projections block the
    # gradient to the time MLP at init, so check the model as a whole rather than any one tensor.
    assert any(
        not torch.allclose(b, a) for b, a in zip(before, after)
    ), "optimizer step did not change any parameter"

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
    parser.add_argument(
        "--time-sampling",
        type=str,
        choices=["uniform", "logit_normal"],
        default="uniform",
        help="Distribution for r, t before the equal-time/endpoint overrides. 'logit_normal' is the "
        "MeanFlow paper's sigmoid(N(-0.4, 1.0)), concentrating training at mid-range noise levels.",
    )
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
        help="Training steps with no sample_mse improvement before --adaptive-lr cuts the LR "
        "(internally converted to eval-call units, since sample_mse is only measured every "
        "--eval-every steps).",
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
    parser.add_argument(
        "--reweight-images",
        action="store_true",
        help="Weight each image's contribution to the loss by (its sample_mse / mean sample_mse), "
        "updated every --eval-every steps. Gives a currently-harder-to-fit image more gradient signal "
        "instead of letting well-fit images dominate the averaged loss.",
    )
    parser.add_argument(
        "--loss-weight-power",
        type=float,
        default=0.0,
        help="MeanFlow paper's adaptive loss weighting power p (Appendix B.2): weight = "
        "1/(per_sample_error + loss_weight_c)^p, stop-gradiented. Down-weights samples with an "
        "unusually large current error (e.g. a noisy JVP estimate) instead of letting them dominate "
        "plain MSE. Paper uses p=1 for ImageNet, p=0.75 for pixel-space CIFAR-10. Default 0 = plain MSE.",
    )
    parser.add_argument(
        "--loss-weight-c",
        type=float,
        default=1e-3,
        help="Small constant in the adaptive loss weighting denominator, avoids division by zero.",
    )
    parser.add_argument(
        "--num-eval-noises",
        type=int,
        default=8,
        help="Number of fixed noises used for the assignment-free eval. More noises make it likelier "
        "that every training image has at least one eval noise inside its basin.",
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
        # plateau_scheduler.step() only runs on eval steps (~every --eval-every training
        # steps), and ReduceLROnPlateau counts patience in step() calls. Convert the
        # user-facing training-step patience into eval-call units here; previously the raw
        # value was passed through, making the effective patience --eval-every times longer
        # than documented (e.g. 300 -> ~3000 training steps), so the LR never dropped.
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_factor,
            patience=max(1, args.lr_patience // args.eval_every),
        )
        scheduler = None
    elif args.no_lr_decay:
        scheduler = None
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.lr_decay_steps or args.steps, eta_min=args.lr * 0.01
        )

    run_sanity_checks(model, optimizer, clean_image)

    # The eval is assignment-free: MeanFlow never promises that noise i maps to image i --
    # the learned marginal flow picks its own noise->image basins. Pairing fixed_noise[i]
    # with image[i] (the old scheme) reported a large "error" whenever a noise sat in
    # another image's basin, even while every sample was a clean copy of *some* training
    # image. So we sample more noises than images and score each image by its best
    # reproduction across all of them.
    num_eval_noises = max(args.num_eval_noises, num_images)
    fixed_eval_noise = torch.randn(num_eval_noises, 3, args.image_size, args.image_size, device=device)
    for index in range(num_images):
        save_image(base_images[index : index + 1], str(output_dir / f"clean_{index}.png"))
    save_image_grid(base_images, str(output_dir / "clean_grid.png"))
    save_image_grid(fixed_eval_noise, str(output_dir / "fixed_eval_noise_grid.png"))

    loss_csv = output_dir / "loss_history.csv"
    # append_loss_csv only appends, so a reused output dir would interleave this run's rows
    # with a previous run's and corrupt every downstream plot. One run, one CSV.
    loss_csv.unlink(missing_ok=True)
    best_sample_mse = float("inf")

    image_indices = torch.arange(args.batch_size, device=device) % num_images
    per_image_weight = torch.ones(num_images, device=device)

    for step in range(1, args.steps + 1):
        batch = make_meanflow_batch(
            clean_image, args.equal_time_probability, args.endpoint_probability, args.time_sampling
        )
        sample_weight = per_image_weight[image_indices] if args.reweight_images else None
        result = meanflow_loss(model, batch, sample_weight, args.loss_weight_power, args.loss_weight_c)

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
                # (num_eval_noises, num_images) MSE between every sample and every image.
                pairwise_mse = ((sample[:, None] - base_images[None, :]) ** 2).mean(dim=(2, 3, 4))
                nearest_mse, nearest_image = pairwise_mse.min(dim=1)
                # sample_mse: are all samples clean copies of some training image?
                sample_mse = nearest_mse.mean().item()
                # per_image_mse: how well is each image reproduced by its best-matching noise?
                per_image_mse, best_noise_index = pairwise_mse.min(dim=0)
                per_image_mse = per_image_mse.tolist()

            if args.reweight_images:
                per_image_mse_tensor = torch.tensor(per_image_mse, device=device)
                per_image_weight = (per_image_mse_tensor / per_image_mse_tensor.mean().clamp_min(1e-8)).clamp(
                    0.2, 5.0
                )

            if plateau_scheduler is not None:
                plateau_scheduler.step(sample_mse)

            append_loss_csv(str(loss_csv), step, loss_value, sample_mse, per_image_mse)

            if sample_mse < best_sample_mse:
                best_sample_mse = sample_mse
                save_checkpoint(output_dir / "checkpoint_best.pt", step, model, optimizer, args, sample_mse, ema)
                for index in range(num_images):
                    best_sample = sample[best_noise_index[index] : best_noise_index[index] + 1]
                    save_image(best_sample, str(output_dir / f"sample_best_{index}.png"))
                # Sort by (assigned image, error) so the basins read left to right.
                order = sorted(
                    range(num_eval_noises), key=lambda i: (nearest_image[i].item(), nearest_mse[i].item())
                )
                save_image_grid(sample[order], str(output_dir / "sample_best_grid.png"))

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
