# Task 13 — Ensemble Model

## Task Info

| Item | Detail |
|------|--------|
| **Sprint** | 3 |
| **Task ID** | S3-T13 |
| **Priority** | Medium |
| **Story Points** | 5 |
| **Status** | Belum Mulai |

## Description
Membuat ensemble dari top 2-3 model terbaik. ISIC winners 2024 membuktikan bahwa **ensemble CNN + Transformer (hybrid)** memberikan hasil terbaik.

## Ensemble Combinations to Test

| Ensemble | Models | Strategy |
|----------|--------|----------|
| Hybrid-1 | EfficientNetV2-S + Swin-Tiny | Weighted Avg |
| Hybrid-2 | ConvNeXt-Tiny + Swin-Tiny | Weighted Avg |
| CNN-Duo | EfficientNetV2-S + ConvNeXt-Tiny | Weighted Avg |
| Vote | Top 3 models | Majority/Soft Voting |

## Acceptance Criteria
- [ ] Minimal 2 ensemble strategies tested
- [ ] Ensemble vs single-model comparison
- [ ] Final decision: deploy ensemble atau single model (pertimbangkan inference time vs accuracy)

## Estimated Time
~3 jam

## Dependencies
- S3-T11 (optimal models trained)
