# Task 07 — Evaluasi Metrics

## Task Info

| Item | Detail |
|------|--------|
| **Sprint** | 2 |
| **Task ID** | S2-T07 |
| **Priority** | High |
| **Story Points** | 2 |
| **Status** | Belum Mulai |
| **Assignee** | - |

## Description
Menghitung dan mendokumentasikan semua metrics evaluasi untuk baseline model.

## Acceptance Criteria
- [ ] Accuracy (overall) dihitung
- [ ] Precision, Recall, F1-Score per kelas dihitung
- [ ] Weighted & Macro averages dihitung
- [ ] Classification report disimpan
- [ ] Module `src/evaluate.py` dibuat

## Implementation

```python
# src/evaluate.py
from sklearn.metrics import (
    classification_report, accuracy_score, 
    precision_recall_fscore_support
)
import numpy as np

def evaluate_model(model, test_dataset, class_names):
    """Evaluate model and return detailed metrics."""
    
    # Get predictions
    y_pred_probs = model.predict(test_dataset)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Get true labels
    y_true = []
    for _, labels in test_dataset:
        y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_true = np.array(y_true)
    
    # Classification report
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4
    )
    print(report)
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'report': report,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_probs': y_pred_probs
    }
```

## Metrics Explained

| Metric | Apa yang diukur | Penting untuk |
|--------|----------------|---------------|
| **Accuracy** | Overall correct predictions | General performance |
| **Precision** | Dari yang diprediksi positif, berapa yang benar | Mengurangi false positives |
| **Recall** | Dari yang sebenarnya positif, berapa yang terdeteksi | Mengurangi false negatives |
| **F1-Score** | Harmonic mean precision & recall | Balanced metric |
| **Melanoma Recall** | Deteksi rate melanoma | **KRITIS — melanoma harus terdeteksi!** |

## Estimated Time
~1 jam

## Dependencies
- S2-T06 (model sudah di-train)
