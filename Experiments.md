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

## Experiment 2: One-image overfit debugging

Date: July 3, 2026

Goal:
Debug why the one-image overfit learns rough structure but does not produce a clean one-step reconstruction.

Code changes:
- Saved `clean.png` in each output directory.
- Saved `fixed_noise.png` in each output directory.
- Logged `sample_mse`, the MSE between `one_step_sample(fixed_noise)` and the clean image.
- Saved `checkpoint_best.pt` based on lowest `sample_mse`.

Experiment A: lower learning rate with full MeanFlow target

Command:
```bash
python train.py \
  --image data/overfit1/imagenette_one.png \
  --steps 10000 \
  --batch-size 8 \
  --lr 3e-4 \
  --sample-every 500 \
  --checkpoint-every 1000 \
  --output-dir outputs/debug_lr3e4
```

Results:
- First: step 1, loss 1.430307, sample_mse 1.405791.
- Final: step 10000, loss 0.610777, sample_mse 0.264352.
- Best loss: step 8436, loss 0.148561, sample_mse 0.270739.
- Best sample_mse: step 2441, loss 0.524021, sample_mse 0.250019.
- Best checkpoint: `outputs/debug_lr3e4/checkpoint_best.pt`.
- Contact sheet: `outputs/debug_lr3e4/contact_sheet.png`.

Observation:
Lower learning rate made training stable and the sample learned rough image structure, but the one-step sample stayed blurry/noisy. Training longer with this tiny model gave diminishing returns.

Experiment B: force `r == t`

Command:
```bash
python train.py \
  --image data/overfit1/imagenette_one.png \
  --steps 5000 \
  --batch-size 8 \
  --equal-time-probability 1.0 \
  --sample-every 500 \
  --checkpoint-every 1000 \
  --output-dir outputs/debug_equal_times
```

Results:
- First: step 1, loss 1.409987, sample_mse 1.365403.
- Final: step 5000, loss 0.126825, sample_mse 0.788535.
- Best loss: step 4757, loss 0.110413, sample_mse 0.816218.
- Best sample_mse: step 167, loss 0.255502, sample_mse 0.271434.
- Best checkpoint: `outputs/debug_equal_times/checkpoint_best.pt`.
- Contact sheet: `outputs/debug_equal_times/contact_sheet.png`.

Observation:
The equal-time run achieved lower training loss, but endpoint one-step sample quality got worse after the early steps. This shows that low velocity-target loss at random times is not enough to guarantee good one-step generation from fixed noise.

Conclusion:
The full MeanFlow target is better for one-step sampling than the `r == t` diagnostic, but the current tiny CNN still lacks enough endpoint reconstruction quality. Next likely debugging moves: save/reload the exact fixed noise tensor in checkpoints, add a slightly larger CNN, or add an explicit endpoint diagnostic term/metric for `model(noise, 0, 1)` versus `noise - clean_image`.
