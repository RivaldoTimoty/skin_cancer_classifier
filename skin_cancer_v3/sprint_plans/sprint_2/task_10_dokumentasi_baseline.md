# Task 10 — Dokumentasi Baseline Performance

## Task Info

| Item | Detail |
|------|--------|
| **Sprint** | 2 |
| **Task ID** | S2-T10 |
| **Priority** | High |
| **Story Points** | 1 |
| **Status** | Belum Mulai |
| **Assignee** | - |

## Description
Mendokumentasikan semua hasil baseline model sebagai benchmark untuk SprinT 3 (Transfer Learning).

## Acceptance Criteria
- [ ] Notebook `notebooks/02_Baseline_Model.ipynb` dibuat
- [ ] Semua metrics tercatat dalam tabel
- [ ] Training curves & confusion matrix terintegrasi
- [ ] Kesimpulan dan next steps didokumentasikan
- [ ] Report `reports/baseline_report.md` dibuat

## Report Template

```markdown
# Baseline Model Performance Report

## Model Architecture
- Type: Custom CNN (4 Conv blocks)
- Total Parameters: X
- Input Size: 224×224×3

## Training Configuration
- Epochs: X (Early stopped at epoch Y)
- Batch Size: 32
- Optimizer: Adam (lr=1e-3)
- Loss: Categorical Crossentropy
- Class Weights: Balanced
- Training Time: X minutes

## Results

### Overall Metrics
| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| Accuracy | X% | X% | X% |
| Loss | X | X | X |

### Per-Class Metrics (Test Set)
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| ... | ... | ... | ... | ... |

## Key Observations
1. ...
2. ...

## Next Steps (Sprint 3)
- Apply Transfer Learning (EfficientNet, MobileNet)
- Hyperparameter tuning
- Advanced augmentation
```

## Estimated Time
~30 menit

## Dependencies
- S2-T06 sampai S2-T08 selesai
