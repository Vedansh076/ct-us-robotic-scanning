# CT → SimUS Translation — Design Notes

## 1. Normalisation Choices

### CT input  →  `[-1, 1]`

| Decision | Rationale |
|---|---|
| **Clip to `[-150, 1250]` HU** | The relevant anatomy for spinal ultrasound simulation spans soft tissue (~0 HU), cortical bone (~400–1000 HU) and the surrounding musculature (~−80 to −20 HU). Values below −150 are mostly air pockets irrelevant to the simulation; values above 1250 are artefacts. Clipping prevents these extremes from stretching the normalisation range. |
| **Scale to `[-1, 1]`** | Matches the natural output range of `tanh` activations and is the standard convention for image-translation networks trained with L1/GAN losses. Centring at 0 also benefits batch statistics (though we use InstanceNorm, which re-normalises per sample). |

### SimUS output  →  `[0, 1]`

| Decision | Rationale |
|---|---|
| **Clip to `[0, 220]`** | Ultrasound intensities are non-negative by physics; the upper clip removes the very small tail of bright specular reflections that would otherwise dominate the range. |
| **Scale to `[0, 1]`** | Matches the final `Sigmoid` activation in the model head. L1 loss then operates in a bounded, well-conditioned space. |

> **Runtime adjustment**: if the actual distribution of your SimUS values differs materially after exploring the dataset, update `SIMUS_MAX` in `dataset.py` and re-run. The clipping percentile that discards ≤ 0.5 % of pixels on each tail is a safe heuristic.

---

## 2. Loss Function — L1

**L1 (mean absolute error)** was chosen over L2 (MSE) because:

* **Sharpness**: L2 penalises large errors quadratically, which biases the model toward blurry mean predictions. L1's linear penalty tolerates localised high-frequency details better.
* **Outlier robustness**: Spine CT slices may contain metal implants or reconstruction artefacts with extreme HU values. L1 is less sensitive to these than L2.
* **Common practice**: L1 is the standard reconstruction loss in paired image-translation literature (Pix2Pix uses L1 + adversarial loss; for a supervised baseline without a discriminator, L1 alone is well-validated).

**Future extension**: Adding a perceptual loss (VGG feature-space L1) or a frequency-domain loss (FFT magnitude L1) could improve texture fidelity without needing a GAN.

---

## 3. Architecture Hyperparameters

| Hyperparameter | Default | Reasoning |
|---|---|---|
| `base_features` | 64 | Standard U-Net width. Gives ~31 M parameters — powerful enough for medical image translation, trainable on 8 GB VRAM with batch 8. |
| `dropout` | 0.1 | Light `Dropout2d` on the bottleneck + deepest skip. ~9 k training samples is modest; this helps regularisation without degrading convergence. |
| InstanceNorm `affine=True` | — | Learnable scale/shift restores representational capacity removed by normalisation. |

---

## 4. Recommended Batch Size & Learning Rate for 8 GB GPU

### Memory budget (256 × 256 inputs, fp16 AMP)

| Component | Approx. VRAM |
|---|---|
| Model weights (fp32 master copy) | ~120 MB |
| Model weights (fp16 forward copy) | ~60 MB |
| Activations at batch 8 | ~1.8 GB |
| Gradients | ~120 MB |
| Optimiser states (Adam: 2× params) | ~240 MB |
| **Total** | **~2.4 GB** |

At batch 8 you have comfortable headroom on an 8 GB card. **Batch 16 is also feasible** (~3.5 GB) and will slightly stabilise gradient estimates.

### Recommended settings

```
--batch_size 8          # safe & tested; try 12 if memory permits
--lr        2e-4        # Adam sweet spot for U-Nets on medical images
--epochs    100         # cosine-annealing decay reaches ~2e-6 by epoch 100
--weight_decay 1e-5     # light L2 regularisation on weights
--grad_clip 1.0         # prevents rare gradient spikes from metal artefacts
```

**Learning rate schedule**: CosineAnnealingLR from `2e-4` to `2e-6` over 100 epochs. This avoids the abrupt loss plateau of step-wise decay and typically yields 0.5–1.0 % lower final validation loss compared to constant LR.

**Effective batch tricks**: if you need to simulate a larger batch without fitting it in VRAM, add gradient accumulation (sum gradients over N mini-batches before stepping). For this dataset size it is unlikely to be necessary.

---

## 5. Subject-Based Split Rationale

Splitting by **subject** (2 train / 1 val) rather than by random slice ensures:

* The validation set contains **anatomical variation unseen during training** — the model must generalise across inter-subject differences in bone density, spinal curvature, and soft-tissue morphology.
* No slice from a given subject appears in both splits (which random splitting would risk, creating data leakage through near-identical adjacent slices).

With only 3 subjects the split is coarse (67 / 33 %), but it is the statistically honest choice. Use subject `tcga-qq-asvc` as a held-out test set; consider cross-validation (leave-one-subject-out) for final reporting.

---

## 6. Project Structure

```
ct_simus/
├── dataset.py       # CTSimUSDataset + normalisation utils
├── model.py         # UNet (encoder-bottleneck-decoder + skip connections)
├── train.py         # Full training loop with AMP, checkpointing, plots
├── inference.py     # Single-file and batch inference + optional metrics
├── requirements.txt
└── NOTES.md         # This file
```

---

## 7. Quick-Start Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train
python train.py \
    --data_root /path/to/dataset \
    --output_dir ./runs/exp1 \
    --epochs 100 \
    --batch_size 8 \
    --lr 2e-4

# Inference on the validation subject
python inference.py \
    --checkpoint ./runs/exp1/best_model.pth \
    --ct_dir /path/to/dataset/ct/ \
    --simus_dir /path/to/dataset/simus/ \
    --out_dir ./runs/exp1/preds/

# Resume interrupted training
python train.py \
    --data_root /path/to/dataset \
    --output_dir ./runs/exp1 \
    --resume ./runs/exp1/latest_checkpoint.pth
```
