# Task 12 — Advanced Augmentation (MixUp / CutMix)

## Task Info

| Item | Detail |
|------|--------|
| **Sprint** | 3 |
| **Task ID** | S3-T12 |
| **Priority** | Medium |
| **Story Points** | 3 |
| **Status** | Belum Mulai |

## Description
Implementasi advanced augmentation techniques dan A/B test terhadap top models. Sangat penting untuk model modern seperti Swin dan ConvNeXt yang membutuhkan strong augmentation.

## Techniques
1. **MixUp** — Linear interpolation antara 2 gambar+label
2. **CutMix** — Patch dari gambar A paste ke gambar B
3. **Albumentations** — CLAHE, ColorJitter, GridDistortion

## Acceptance Criteria
- [ ] Minimal 2 advanced augmentation diimplementasi
- [ ] A/B test: standard augmentation vs advanced pada top model
- [ ] Best augmentation pipeline dipilih

## Estimated Time
~3 jam

## Dependencies
- S2-T02 (basic augmentation)
