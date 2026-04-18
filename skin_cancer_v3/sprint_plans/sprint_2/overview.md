# Sprint 2 — Data Pipeline & Baseline Model

## Sprint Goal
*Membuat data pipeline yang robust dan model baseline sebagai benchmark.*

## Sprint Info

| Item | Detail |
|------|--------|
| **Sprint** | 2 |
| **Durasi** | 3-4 hari |
| **Total Story Points** | 22 SP |
| **Start Date** | TBD |
| **End Date** | TBD |
| **Status** | Belum Mulai |

## Task List

| # | Task | Priority | SP | Status | Assignee |
|---|------|----------|-----|--------|----------|
| 1 | [Implementasi Data Pipeline](./task_01_data_pipeline.md) | High | 3 | | - |
| 2 | [Implementasi Data Augmentation](./task_02_data_augmentation.md) | High | 3 | | - |
| 3 | [Handle Class Imbalance](./task_03_class_imbalance.md) | High | 3 | | - |
| 4 | [Build Baseline CNN Model](./task_04_baseline_cnn.md) | High | 3 | | - |
| 5 | [Setup Training Callbacks](./task_05_training_callbacks.md) | High | 2 | | - |
| 6 | [Train Baseline Model](./task_06_train_baseline.md) | High | 2 | | - |
| 7 | [Evaluasi Metrics](./task_07_evaluation_metrics.md) | High | 2 | | - |
| 8 | [Confusion Matrix](./task_08_confusion_matrix.md) | Medium | 2 | | - |
| 9 | [Setup TensorBoard](./task_09_tensorboard.md) | Medium | 1 | | - |
| 10 | [Dokumentasi Baseline](./task_10_dokumentasi_baseline.md) | High | 1 | | - |

## Definition of Done
- [ ] Data pipeline (`tf.data`) berjalan efisien
- [ ] Augmentation pipeline terimplementasi
- [ ] Baseline CNN model trained dan metrics tercatat
- [ ] Confusion matrix menunjukkan per-class performance
- [ ] Baseline accuracy menjadi benchmark untuk improvement

## Dependencies
- Sprint 1 selesai (EDA & data split ready)

## Target Baseline Metrics
| Metric | Target Minimum |
|--------|---------------|
| Accuracy | ≥ 60% |
| Weighted F1 | ≥ 0.55 |

> **Note:** Baseline tidak diharapkan bagus — tujuannya adalah benchmark.
