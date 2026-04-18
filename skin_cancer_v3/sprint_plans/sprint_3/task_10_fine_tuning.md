# Task 10 — Fine-Tuning Strategy (All Models)

## Task Info

| Item | Detail |
|------|--------|
| **Sprint** | 3 |
| **Task ID** | S3-T10 |
| **Priority** | High |
| **Story Points** | 3 |
| **Status** | Belum Mulai |

## Description
Menerapkan strategi fine-tuning konsisten untuk semua 8 model: Phase 1 (frozen base) → Phase 2 (partial unfreeze) → Phase 3 (optional full). Mendokumentasikan layer-layer mana yang di-unfreeze per model.

## Fine-Tuning Map

| Model | Phase 2: Unfreeze From | # Layers Unfrozen | Phase 2 LR |
|-------|----------------------|-------------------|------------|
| **EfficientNetV2-S** | Last 60 layers | ~60 layers | 1e-5 |
| **ConvNeXt-Tiny** | Stage 3 + 4 | ~40 layers | 1e-5 |
| **Swin-Tiny** | Stage 3 + 4 | ~40 layers | 1e-5 |
| DenseNet121 | `conv5_block1_0_bn` | ~40 layers | 1e-5 |
| ResNet50V2 | Last 30 layers | ~30 layers | 1e-5 |
| InceptionV3 | `mixed7` | ~40 layers | 1e-5 |
| MobileNetV2 | Last 30 layers | ~30 layers | 1e-5 |
| Xception | Last 40 layers | ~40 layers | 1e-5 |

## Acceptance Criteria
- [ ] Semua model melewati minimal 2-phase training
- [ ] Phase 1 vs Phase 2 improvement dicatat per model
- [ ] Learning rate warmup verified untuk Phase 2
- [ ] Dokumentasi layer unfreeze per arsitektur

## Estimated Time
Included in individual model training times

## Dependencies
- S3-T02 sampai S3-T09 (semua model)
