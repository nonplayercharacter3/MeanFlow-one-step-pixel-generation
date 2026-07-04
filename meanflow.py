from dataclasses import dataclass

import torch
from torch.func import jvp


@dataclass
class MeanFlowBatch:
    z_t: torch.Tensor
    velocity: torch.Tensor
    r: torch.Tensor
    t: torch.Tensor


@dataclass
class MeanFlowLoss:
    loss: torch.Tensor
    mean_velocity: torch.Tensor
    jvp_term: torch.Tensor
    target: torch.Tensor


def sample_times(
    batch_size: int,
    device: torch.device,
    equal_time_probability: float = 0.1,
    endpoint_probability: float = 0.25,
    time_sampling: str = "uniform",
    logit_normal_mean: float = -0.4,
    logit_normal_std: float = 1.0,
):
    """Sample scalar times with 0 <= r <= t <= 1.

    Sometimes forces r == t (plain flow-matching stabilizer), and separately
    forces r == 0, t == 1 for a fraction of the batch. The (r=0, t=1) corner
    is exactly what one-step sampling evaluates at inference, but independent
    uniform r, t give it zero density (most sampled gaps t-r cluster near 0),
    so without this the model rarely practices the case it is actually judged on.

    time_sampling="logit_normal" draws each time as sigmoid(N(mean, std)) instead of
    uniform -- the MeanFlow paper's scheme (mu=-0.4, sigma=1.0), which concentrates
    training at mid-range noise levels where the velocity field is hardest, instead of
    spending a third of the batch on the nearly-trivial regions near t=0 and t=1.
    """
    if time_sampling == "logit_normal":
        samples = torch.sigmoid(
            torch.randn(2, batch_size, device=device) * logit_normal_std + logit_normal_mean
        )
        r, t = samples[0], samples[1]
    elif time_sampling == "uniform":
        r = torch.rand(batch_size, device=device)
        t = torch.rand(batch_size, device=device)
    else:
        raise ValueError(f"unknown time_sampling: {time_sampling!r}")
    r, t = torch.minimum(r, t), torch.maximum(r, t)

    equal_mask = torch.rand(batch_size, device=device) < equal_time_probability
    r = torch.where(equal_mask, t, r)

    endpoint_mask = torch.rand(batch_size, device=device) < endpoint_probability
    r = torch.where(endpoint_mask, torch.zeros_like(r), r)
    t = torch.where(endpoint_mask, torch.ones_like(t), t)

    return r[:, None], t[:, None]


def make_meanflow_batch(
    clean_image: torch.Tensor,
    equal_time_probability: float = 0.1,
    endpoint_probability: float = 0.25,
    time_sampling: str = "uniform",
) -> MeanFlowBatch:
    """Create one training batch from the fixed clean image and fresh Gaussian noise."""
    batch_size = clean_image.shape[0]
    noise = torch.randn_like(clean_image)
    r, t = sample_times(
        batch_size, clean_image.device, equal_time_probability, endpoint_probability, time_sampling
    )

    t_image = t[:, :, None, None]
    z_t = (1.0 - t_image) * clean_image + t_image * noise
    velocity = noise - clean_image

    return MeanFlowBatch(z_t=z_t, velocity=velocity, r=r, t=t)


def meanflow_loss(
    model,
    batch: MeanFlowBatch,
    sample_weight: torch.Tensor = None,
    loss_weight_power: float = 0.0,
    loss_weight_c: float = 1e-3,
) -> MeanFlowLoss:
    """Compute MSE(u_theta, stopgrad(v - (t-r) du_theta/dt)).

    sample_weight, if given, is a per-batch-element weight (shape (batch,)) applied to the
    squared error -- used to give harder-to-fit images more gradient signal when a batch mixes
    multiple fixed targets (see --reweight-images in train.py).

    loss_weight_power (p > 0) applies the adaptive loss weighting from the MeanFlow paper
    (Appendix B.2): w = 1 / (per-sample squared error + loss_weight_c) ** p, stop-gradiented.
    This down-weights samples whose current error is unusually large (e.g. from a noisy JVP
    estimate on an endpoint-biased sample), similar to a Pseudo-Huber loss, instead of letting
    a few large-error samples dominate the plain-MSE gradient. p=0 (default) is plain MSE.
    """

    def model_fn(z_t, r, t):
        return model(z_t, r, t)

    zeros_like_time = torch.zeros_like(batch.r)
    mean_velocity, jvp_term = jvp(
        model_fn,
        (batch.z_t, batch.r, batch.t),
        (batch.velocity, zeros_like_time, torch.ones_like(batch.t)),
    )

    time_gap = (batch.t - batch.r)[:, :, None, None]
    target = batch.velocity - time_gap * jvp_term
    target = target.detach()

    squared_error = (mean_velocity - target) ** 2
    per_sample_error = squared_error.mean(dim=(1, 2, 3))

    weight = torch.ones_like(per_sample_error)
    if loss_weight_power > 0:
        weight = weight * (per_sample_error.detach() + loss_weight_c).pow(-loss_weight_power)
    if sample_weight is not None:
        weight = weight * sample_weight

    loss = (per_sample_error * weight).mean()

    return MeanFlowLoss(
        loss=loss,
        mean_velocity=mean_velocity,
        jvp_term=jvp_term,
        target=target,
    )


def one_step_sample(model, noise: torch.Tensor) -> torch.Tensor:
    """MeanFlow one-step sample: x_hat = epsilon - u_theta(epsilon, 0, 1)."""
    batch_size = noise.shape[0]
    r = torch.zeros(batch_size, 1, device=noise.device, dtype=noise.dtype)
    t = torch.ones(batch_size, 1, device=noise.device, dtype=noise.dtype)
    mean_velocity = model(noise, r, t)
    return noise - mean_velocity
