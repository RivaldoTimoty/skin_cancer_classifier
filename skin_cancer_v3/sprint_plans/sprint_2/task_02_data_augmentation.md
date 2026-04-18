# Task 02 — Implementasi Data Augmentation

## Task Info

| Item | Detail |
|------|--------|
| **Sprint** | 2 |
| **Task ID** | S2-T02 |
| **Priority** | High |
| **Story Points** | 3 |
| **Status** | Belum Mulai |
| **Assignee** | - |

## Description
Implementasi data augmentation pipeline untuk meningkatkan variasi training data dan mengurangi overfitting.

## Acceptance Criteria
- [ ] Module `src/augmentation.py` dibuat
- [ ] Minimal 5 jenis augmentasi diimplementasi
- [ ] Augmentasi hanya diterapkan pada training data (bukan validation/test)
- [ ] Visual comparison sebelum dan sesudah augmentasi dibuat
- [ ] Kelas minoritas mendapat augmentasi lebih agresif

## Augmentation Techniques

| Technique | Parameter | Alasan |
|-----------|-----------|--------|
| Random Horizontal Flip | p=0.5 | Lesi bisa muncul di orientasi apapun |
| Random Rotation | ±20° | Rotasi natural |
| Random Zoom | ±15% | Variasi scale |
| Random Brightness | ±20% | Variasi lighting |
| Random Contrast | ±20% | Variasi kontras kamera |
| Random Translation | ±10% | Posisi lesi tidak selalu center |

## Code Skeleton
```python
# src/augmentation.py
import tensorflow as tf

def get_augmentation_layer():
    """Create a Keras augmentation layer."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
    ], name='augmentation')

def augment_dataset(dataset, augmentation_layer):
    """Apply augmentation to a dataset."""
    return dataset.map(
        lambda x, y: (augmentation_layer(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
```

## Visual Verification
- Tampilkan 5 versi augmented dari 1 gambar asli per kelas
- Pastikan augmentasi tidak merusak informasi diagnostik penting

## Important Notes
- **JANGAN** augment validation/test data
- Jangan terlalu agresif — lesi harus tetap bisa dikenali
- Pertimbangkan augmentasi yang medically valid

## Estimated Time
~2 jam

## Dependencies
- S2-T01 (data pipeline sudah ada)
