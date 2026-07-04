# MeanFlow in PyTorch

A minimal PyTorch implementation of MeanFlow for one-step image generation in pixel space.

This project ports the core MeanFlow training objective from the reference JAX implementation into a small GPU-compatible PyTorch training loop. The current implementation focuses on verifying the method through tiny overfitting experiments before scaling to a 10-class Imagenette subset.

## Project Goal

The main goal is to train a model that predicts the average velocity

```text
u_theta(z_t, r, t)
```

and generates an image from noise in one network evaluation:

```text
x_hat = noise - u_theta(noise, 0, 1)
```

The required milestone is to overfit three fixed training images and reproduce them using one-step sampling.

## Current Status

**The required milestone is reached**: the 3-image overfit converges to mean nearest-image `sample_mse = 0.0094` (per image: 0.0021 / 0.0050 / 0.0053) in ~900 steps, and one-step samples from held-out noise are visually indistinguishable copies of the training images. See `report.html` for the full debugging journey.

Implemented:

* image loading and normalization;
* linear interpolation between image and noise;
* MeanFlow target construction with `torch.func.jvp` and a detached target;
* the paper's adaptive loss weighting (`--loss-weight-power`) and logit-normal time sampling (`--time-sampling logit_normal`);
* mini U-Net with sinusoidal `(t, t-r)` embeddings, per-block FiLM conditioning, GroupNorm, and bottleneck self-attention;
* one-step sampler;
* EMA weights for evaluation, adaptive LR (`ReduceLROnPlateau` on sample quality), gradient clipping;
* assignment-free evaluation: samples are scored against their *nearest* training image, since MeanFlow's marginal flow chooses its own noise-to-image assignment;
* `sample.py` for held-out-noise evaluation of a trained checkpoint;
* analytical JVP test, gradient and numerical sanity checks, checkpoint and sample saving.

## Repository Structure

```text
MeanFlow/
├── train.py          # Training entry point
├── sample.py         # Held-out-noise evaluation of a trained checkpoint
├── model.py          # Mini U-Net with FiLM time conditioning
├── meanflow.py       # MeanFlow objective, JVP, and sampler
├── utils.py          # Image loading and output utilities
├── README.md
├── Experiments.md    # Chronological experiment log
├── report.html       # Assignment report
├── data/
└── outputs/
```

## Requirements

* Python 3.10 or newer
* PyTorch 2.x
* torchvision
* Pillow
* matplotlib

Install dependencies:

```bash
pip install torch torchvision pillow matplotlib
```

For the final experiment, an NVIDIA GPU is recommended. The code can run on CPU for small debugging experiments, but training is substantially slower.

## Data

The overfit experiments use fixed images from Imagenette (`data/overfit1/`, `data/overfit3/`).

Each image is:

* converted to RGB;
* resized to `32 x 32`;
* converted to a PyTorch tensor;
* normalized from `[0, 1]` to `[-1, 1]`.

The full assignment target uses either Imagenette or another fixed 10-class ImageNet subset.

## Run a Syntax Check

```bash
python -m py_compile train.py sample.py model.py meanflow.py utils.py
```

## Run a Short Smoke Test

```bash
python train.py \
  --images data/smoke_test.png \
  --steps 100 \
  --batch-size 8 \
  --sample-every 50 \
  --checkpoint-every 100 \
  --output-dir outputs/smoke_100
```

This verifies that:

* the model runs;
* the JVP has the correct shape;
* gradients flow;
* parameters update;
* no NaN or infinite values appear.

## Run the One-Image Overfit Experiment

```bash
python train.py \
  --images data/overfit1/imagenette_one.png \
  --steps 5000 \
  --batch-size 8 \
  --lr 3e-4 \
  --hidden-channels 256 \
  --num-blocks 6 \
  --sample-every 500 \
  --checkpoint-every 1000 \
  --output-dir outputs/imagenette_one_overfit
```

The same clean image is repeated across the batch, but every batch element receives fresh random noise and newly sampled times.

## Run the Three-Image Overfit Experiment (required milestone)

This is the final recipe (~10 minutes on a Colab L4, reaches `sample_mse ≈ 0.009`):

```bash
python train.py \
  --images data/overfit3/image_0.png data/overfit3/image_1.png data/overfit3/image_2.png \
  --steps 2500 \
  --batch-size 64 \
  --lr 5e-4 \
  --hidden-channels 128 \
  --num-blocks 2 \
  --time-sampling logit_normal \
  --equal-time-probability 0.5 \
  --endpoint-probability 0.1 \
  --loss-weight-power 0.75 \
  --adaptive-lr \
  --lr-patience 150 \
  --ema-decay 0.99 \
  --reweight-images \
  --sample-every 500 \
  --checkpoint-every 1000 \
  --output-dir outputs/overfit3_final
```

`--images` accepts any number of paths; the batch is filled by cycling through them.

Evaluation is assignment-free: MeanFlow's learned marginal flow chooses its own noise-to-image mapping, so pairing a fixed noise with a fixed image would report spurious errors. Instead, each of 8 fixed eval noises is scored against its *nearest* training image (`sample_mse`), and each image is scored by its best reproduction across the noises (the per-image `img0=...` numbers in the log and CSV). Outputs include `clean_grid.png`, `sample_best_grid.png` (samples sorted by nearest image), per-image `clean_{i}.png` / `sample_best_{i}.png` pairs, `loss_history.csv`, and `loss_curve.png`.

## Evaluate a Checkpoint on Held-Out Noise

```bash
python sample.py \
  --checkpoint outputs/overfit3_final/checkpoint_best.pt \
  --images data/overfit3/image_0.png data/overfit3/image_1.png data/overfit3/image_2.png \
  --num-samples 16
```

Samples 16 fresh noises (a seed never used in training), prints each sample's nearest training image and MSE plus per-image coverage, and saves `samples_many_noises.png` sorted by nearest image. This is the confirmation that the model reproduces all three images from noise it has never been evaluated on.

## Training Procedure

For each training step:

1. Load the fixed clean image.
2. Sample Gaussian noise.
3. Sample times `r` and `t` such that `r <= t`.
4. Sometimes force `r == t` for ordinary flow-matching stabilization.
5. Construct

```text
z_t = (1 - t) * clean_image + t * noise
```

6. Compute the path velocity

```text
velocity = noise - clean_image
```

7. Evaluate the model and its directional derivative using

```python
torch.func.jvp
```

with tangent

```text
(velocity, 0, 1)
```

8. Construct the detached MeanFlow target

```text
target = velocity - (t - r) * jvp_term
```

9. Minimize mean-squared error between the model prediction and target.

## One-Step Sampling

Sampling uses:

```text
x_hat = noise - model(noise, r=0, t=1)
```

This requires only one network evaluation.

Generated samples are saved periodically in the selected output directory.

## Outputs

Each experiment directory contains:

```text
outputs/experiment_name/
├── clean_grid.png            # the training images side by side
├── clean_{i}.png             # each training image
├── sample_best_grid.png      # best one-step samples, sorted by nearest image
├── sample_best_{i}.png       # best reproduction of image i
├── sample_step_{n}.png       # periodic sample grids
├── samples_many_noises.png   # written by sample.py
├── checkpoint.pt             # periodic checkpoint (model, EMA, optimizer, args)
├── checkpoint_best.pt        # checkpoint at the best sample_mse
├── loss_history.csv          # per-eval loss, sample_mse, per-image mse
└── loss_curve.png            # plotted at the end of training
```

## Reproducibility

For each experiment, record:

* exact command;
* random seed;
* learning rate;
* number of steps;
* batch size;
* image resolution;
* model configuration;
* time-sampling strategy;
* probability of `r == t`;
* device used.

Detailed experiment notes are stored in `Experiments.md`; the debugging narrative is in `report.html`.

## Known Limitations

* Unconditional only; basin coverage across the 3 images is imbalanced (most random noises map to one image, though all three are reliably produced).
* CPU training is slow; the final recipe assumes a GPU.
* A decreasing loss does not by itself prove successful one-step generation — hence the assignment-free `sample_mse` and `sample.py` held-out check.
* The optional 10-class Imagenette extension was not attempted.

## References

* Mean Flows for One-step Generative Modeling
* PyTorch `torch.func.jvp`
* Imagenette dataset
