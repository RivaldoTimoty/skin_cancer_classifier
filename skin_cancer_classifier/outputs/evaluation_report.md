# Skin Cancer Classification — Evaluation Report

**Model**: EfficientNet-B0 (Transfer Learning)  
**Dataset**: ISIC 9-Class Skin Cancer  
**Date**: Generated automatically  

---

## Overall Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.6220 |
| Precision (Macro) | 0.5319 |
| Recall (Macro) | 0.5447 |
| F1-Score (Macro) | 0.5339 |
| F1-Score (Weighted) | 0.6218 |
| Precision (Weighted) | 0.6302 |
| Recall (Weighted) | 0.6220 |
| ROC-AUC (Macro) | 0.9176 |
| ROC-AUC (Weighted) | 0.9157 |

## Classification Report

```
                            precision    recall  f1-score   support

         actinic keratosis       0.22      0.24      0.23        17
      basal cell carcinoma       0.90      0.82      0.86        56
            dermatofibroma       0.45      0.64      0.53        14
                  melanoma       0.60      0.65      0.62        66
                     nevus       0.66      0.46      0.54        54
pigmented benign keratosis       0.68      0.74      0.71        69
      seborrheic keratosis       0.00      0.00      0.00        12
   squamous cell carcinoma       0.41      0.44      0.43        27
           vascular lesion       0.86      0.90      0.88        21

                  accuracy                           0.62       336
                 macro avg       0.53      0.54      0.53       336
              weighted avg       0.63      0.62      0.62       336

```

## Confusion Matrix

See `outputs/confusion_matrix.png`

## ROC Curves

See `outputs/roc_curves.png`
