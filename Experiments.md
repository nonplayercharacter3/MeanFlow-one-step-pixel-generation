## Experiment 1: Smoke test

Date: June 23, 2026

Goal:
Verify that the training loop, JVP, gradients, and optimizer work.

Dataset:
One fixed 32×32 RGB image.

Configuration:
- Steps: 100
- Batch size: 8
- Learning rate: ...
- Model: TinyTimeConditionedCNN
- Device: CPU
- Time sampling: ...
- Probability of r == t: ...

Results:
- Initial loss: 1.354688
- Final loss: 0.599790
- finite=True throughout
- Analytical JVP check passed

Observations:
- Loss decreased.
- No NaN or infinity values.
- Generated image was still noisy.

Conclusion:
The mechanical training pipeline runs, but image reconstruction has not yet been demonstrated.

Next step:
Train on one real Imagenette image for 5,000 steps.