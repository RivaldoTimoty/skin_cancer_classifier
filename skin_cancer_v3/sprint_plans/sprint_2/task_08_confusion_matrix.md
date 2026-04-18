# Task 08 — Confusion Matrix Visualization

## Task Info

| Item | Detail |
|------|--------|
| **Sprint** | 2 |
| **Task ID** | S2-T08 |
| **Priority** | Medium |
| **Story Points** | 2 |
| **Status** | Belum Mulai |
| **Assignee** | - |

## Description
Membuat visualisasi confusion matrix untuk melihat pola kesalahan klasifikasi model.

## Acceptance Criteria
- [ ] Confusion matrix (counts) divisualisasikan sebagai heatmap
- [ ] Normalized confusion matrix (percentage) juga dibuat
- [ ] Kelas yang sering tertukar diidentifikasi
- [ ] Plot disimpan di `reports/figures/`

## Implementation

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(y_true, y_pred, class_names, normalize=False, 
                          save_path=None):
    """Plot confusion matrix as heatmap."""
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(title, fontsize=14)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

# Usage
plot_confusion_matrix(
    y_true, y_pred, class_names,
    save_path='reports/figures/baseline_confusion_matrix.png'
)
plot_confusion_matrix(
    y_true, y_pred, class_names, normalize=True,
    save_path='reports/figures/baseline_confusion_matrix_normalized.png'
)
```

## Key Insights to Look For
- Kelas mana yang paling sering salah?
- Apakah ada kelas yang sering tertukar satu sama lain?
- Apakah melanoma recall sudah cukup tinggi?
- Apakah model bias ke kelas mayoritas?

## Estimated Time
~30 menit

## Dependencies
- S2-T07 (evaluasi metrics — y_true, y_pred sudah ada)
