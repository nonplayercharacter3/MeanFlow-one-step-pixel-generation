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

Implemented:

* image loading and normalization;
* linear interpolation between image and noise;
* MeanFlow target construction;
* JVP computation with `torch.func.jvp`;
* detached MeanFlow target;
* tiny time-conditioned CNN;
* one-step sampler;
* analytical JVP test;
* gradient and numerical sanity checks;
* checkpoint and sample saving.

Current experiment:

* one fixed `32 x 32` Imagenette image;
* CPU development test;
* one-step reconstruction still being debugged.

## Repository Structure

```text
MeanFlow/
├── train.py          # Training entry point
├── model.py          # Tiny time-conditioned CNN
├── meanflow.py       # MeanFlow objective, JVP, and sampler
├── utils.py          # Image loading and output utilities
├── README.md
├── EXPERIMENTS.md    # Chronological experiment log
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

The current debugging experiment uses one fixed image from Imagenette.

The image is:

* converted to RGB;
* resized to `32 x 32`;
* converted to a PyTorch tensor;
* normalized from `[0, 1]` to `[-1, 1]`.

The full assignment target uses either Imagenette or another fixed 10-class ImageNet subset.

## Run a Syntax Check

```bash
python -m py_compile train.py model.py meanflow.py utils.py
```

## Run a Short Smoke Test

```bash
python train.py \
  --image data/smoke_test.png \
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
  --image data/overfit1/imagenette_one.png \
  --steps 5000 \
  --batch-size 8 \
  --sample-every 500 \
  --checkpoint-every 1000 \
  --output-dir outputs/imagenette_one_overfit
```

The same clean image is repeated across the batch, but every batch element receives fresh random noise and newly sampled times.

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

Each experiment directory may contain:

```text
outputs/experiment_name/
├── sample_step_0500.png
├── sample_step_1000.png
├── checkpoint_step_1000.pt
├── training.log
└── loss_curve.png
```

Exact filenames may differ depending on the current implementation.

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

Detailed experiment notes are stored in `EXPERIMENTS.md`.

## Known Limitations

* The current model is intentionally very small.
* The current debugging setup uses one image rather than the final three-image dataset.
* CPU training is slow.
* A decreasing loss does not by itself prove successful one-step generation.
* The final correctness test is whether generated samples reproduce the fixed training images.

## Final Target

The required final result is:

* train on three fixed images;
* overfit them successfully;
* generate recognizable reconstructions in one model evaluation;
* report the loss curve and generated samples.

The optional extension is training on a 10-class Imagenette subset.

## References

* Mean Flows for One-step Generative Modeling
* PyTorch `torch.func.jvp`
* Imagenette dataset
