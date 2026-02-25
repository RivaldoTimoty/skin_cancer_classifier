# Skin Cancer Classification — Evaluation Report

**Model**: EfficientNet-B3 (Full Fine-tuning via Google Colab T4 GPU)  
**Dataset**: ISIC 9-Class Skin Cancer  
**Date**: Updated Phase 10

---

## Overall Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.7738 |
| Precision (Macro) | 0.6849 |
| Recall (Macro) | 0.6773 |
| F1-Score (Macro) | 0.6714 |
| F1-Score (Weighted) | 0.7639 |
| Precision (Weighted) | 0.7656 |
| Recall (Weighted) | 0.7738 |
| ROC-AUC (Macro) | 0.9420 |
| ROC-AUC (Weighted) | 0.9545 |

## Classification Report

```
                            precision    recall  f1-score   support

         actinic keratosis       0.32      0.59      0.42        17
      basal cell carcinoma       0.95      0.95      0.95        56
            dermatofibroma       0.89      0.57      0.70        14
                  melanoma       0.76      0.89      0.82        66
                     nevus       0.75      0.61      0.67        54
pigmented benign keratosis       0.82      0.86      0.84        69
      seborrheic keratosis       0.00      0.00      0.00        12
   squamous cell carcinoma       0.68      0.63      0.65        27
           vascular lesion       1.00      1.00      1.00        21

                  accuracy                           0.77       336
                 macro avg       0.68      0.68      0.67       336
              weighted avg       0.77      0.77      0.76       336

```

## Confusion Matrix

See `outputs/confusion_matrix.png`

## ROC Curves

See `outputs/roc_curves.png`
