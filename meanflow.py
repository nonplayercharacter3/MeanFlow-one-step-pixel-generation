from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.func import jvp


@dataclass
class MeanFlowBatch:
    clean_image: torch.Tensor
    noise: torch.Tensor
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
):
    """Sample scalar times with 0 <= r <= t <= 1.

    Sometimes forces r == t (plain flow-matching stabilizer), and separately
    forces r == 0, t == 1 for a fraction of the batch. The (r=0, t=1) corner
    is exactly what one-step sampling evaluates at inference, but independent
    uniform r, t give it zero density (most sampled gaps t-r cluster near 0),
    so without this the model rarely practices the case it is actually judged on.
    """
    r = torch.rand(batch_size, device=device)
    t = torch.rand(batch_size, device=device)
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
) -> MeanFlowBatch:
    """Create one training batch from the fixed clean image and fresh Gaussian noise."""
    batch_size = clean_image.shape[0]
    noise = torch.randn_like(clean_image)
    r, t = sample_times(batch_size, clean_image.device, equal_time_probability, endpoint_probability)

    t_image = t[:, :, None, None]
    z_t = (1.0 - t_image) * clean_image + t_image * noise
    velocity = noise - clean_image

    return MeanFlowBatch(
        clean_image=clean_image,
        noise=noise,
        z_t=z_t,
        velocity=velocity,
        r=r,
        t=t,
    )


def meanflow_loss(model, batch: MeanFlowBatch) -> MeanFlowLoss:
    """Compute MSE(u_theta, stopgrad(v - (t-r) du_theta/dt))."""

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
    loss = F.mse_loss(mean_velocity, target)

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
