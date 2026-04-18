# Task 16 — Grad-CAM Visualization

## Task Info

| Item | Detail |
|------|--------|
| **Sprint** | 3 |
| **Task ID** | S3-T16 |
| **Priority** | Medium |
| **Story Points** | 3 |
| **Status** | Belum Mulai |

## Description
Implementasi interpretability untuk CNN dan Transformer (Attention Maps/Grad-CAM).

## Last Layer per Model

| Model | Last Layer/Block Name |
|-------|---------------------|
| EfficientNetV2-S | `top_activation` |
| ConvNeXt-Tiny | `convnext_tiny_stage_3` |
| Swin-Tiny | *Attention Map visualization* |
| DenseNet121 | `relu` (final) |
| ResNet50V2 | `post_relu` |
| MobileNetV2 | `out_relu` |

## Acceptance Criteria
- [ ] Grad-CAM/Attention Map untuk semua top models
- [ ] Comparison grid: same image, different model heatmaps
- [ ] Verifikasi model fokus pada lesi

## Estimated Time
~3 jam

## Dependencies
- Top models trained
