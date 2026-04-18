# Task 11 — Hyperparameter Tuning

## Task Info

| Item | Detail |
|------|--------|
| **Sprint** | 3 |
| **Task ID** | S3-T11 |
| **Priority** | High |
| **Story Points** | 5 |
| **Status** | Belum Mulai |

## Description
Fine-tune hyperparameters pada **top 3 model** yang menunjukkan performa terbaik dari initial training (kemungkinan besar EfficientNetV2, ConvNeXt, atau Swin).

## Hyperparameters to Tune

| Parameter | Search Space |
|-----------|-------------|
| Learning Rate | [1e-5, 5e-5, 1e-4, 5e-4] |
| Batch Size | [16, 32] |
| Optimizer | [Adam, AdamW] |
| Dropout | [0.3, 0.4, 0.5] |
| Dense Units | [256, 512] |
| Weight Decay | [1e-4, 1e-5, 0] |
| Label Smoothing | [0, 0.1, 0.2] |

## Acceptance Criteria
- [ ] Top 3 model dipilih berdasarkan initial training
- [ ] Minimal 5 hyperparameter configs per model
- [ ] Best config per model didokumentasikan
- [ ] Results table dengan semua experiments

## Estimated Time
~8-12 jam (multiple runs)

## Dependencies
- S3-T02 sampai S3-T09 (initial training done)
