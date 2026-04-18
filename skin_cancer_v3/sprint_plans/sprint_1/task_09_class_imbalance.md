# Task 09 — Strategi Penanganan Class Imbalance

## Task Info

| Item | Detail |
|------|--------|
| **Sprint** | 1 |
| **Task ID** | S1-T09 |
| **Priority** | High |
| **Story Points** | 2 |
| **Status** | Belum Mulai |
| **Assignee** | - |

## Description
Menentukan dan mendokumentasikan strategi terbaik untuk menangani class imbalance yang terdeteksi di dataset (rasio 6:1 antara kelas terbanyak dan tersedikit).

## Acceptance Criteria
- [ ] Minimal 3 strategi class imbalance dievaluasi
- [ ] Pro/cons setiap strategi didokumentasikan
- [ ] Strategi final dipilih dengan justifikasi
- [ ] Parameter strategi ditentukan

## Current Class Distribution

| Kelas | Jumlah | Rasio |
|-------|--------|-------|
| pigmented benign keratosis | 462 | 1.00× |
| seborrheic keratosis | 77 | 0.17× |
| **Imbalance Ratio** | - | **6:1** |

## Strategi yang Dievaluasi

### 1. Class Weights (Weighted Loss)
```python
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
```
| Pro | Con |
|-----|-----|
| Mudah diimplementasi | Mungkin tidak cukup untuk imbalance berat |
| Tidak mengubah data | Bisa membuat model tidak stabil |
| Komputasi ringan | - |

### 2. Data Augmentation (Oversampling Kelas Minoritas)
```python
# Augment lebih agresif untuk kelas dengan sedikit data
# Target: semua kelas punya jumlah gambar yang setara
```
| Pro | Con |
|-----|-----|
| Meningkatkan variasi data | Bisa overfitting ke augmented data |
| Proven efektif untuk images | Waktu training lebih lama |
| Menyeimbangkan kelas | Membutuhkan disk space lebih |

### 3. Focal Loss
```python
# Modifikasi cross-entropy yang fokus ke sample sulit
def focal_loss(gamma=2.0, alpha=0.25):
    ...
```
| Pro | Con |
|-----|-----|
| Efektif untuk imbalance berat | Lebih kompleks |
| Fokus ke hard examples | Perlu tuning gamma & alpha |
| State-of-the-art untuk detection | - |

### 4. Kombinasi (Recommended)
- **Class Weights** + **Targeted Augmentation** + **Stratified Sampling**
- Augment kelas minoritas 3-5× lebih banyak
- Gunakan class weights di loss function
- Pastikan validation set tetap stratified

## Estimated Time
~1 jam

## Dependencies
- S1-T02 (distribusi kelas sudah dianalisis)

## Decision
> Keputusan final akan didokumentasikan setelah analisis selesai.
