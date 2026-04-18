# Task 01 — Implementasi Data Pipeline (tf.data)

## Task Info

| Item | Detail |
|------|--------|
| **Sprint** | 2 |
| **Task ID** | S2-T01 |
| **Priority** | High |
| **Story Points** | 3 |
| **Status** | Belum Mulai |
| **Assignee** | - |

## Description
Membangun data pipeline menggunakan `tf.data` API yang efisien untuk loading, preprocessing, dan batching gambar.

## Acceptance Criteria
- [ ] Pipeline bisa load gambar dari folder structure
- [ ] Gambar di-resize ke target size (224×224)
- [ ] Pixel values dinormalisasi ke [0, 1] atau [-1, 1]
- [ ] Labels di-encode (one-hot atau integer)
- [ ] Pipeline menggunakan prefetch dan cache untuk performance
- [ ] Module `src/data_loader.py` dibuat

## Implementation Steps
1. Buat fungsi `create_dataset()` di `src/data_loader.py`
2. Gunakan `tf.data.Dataset` pipeline
3. Implementasi: load → decode → resize → (optional) normalize → batch → prefetch
4. Dilarang hardcode normalization `/255` karena model seperti EfficientNetV2 dan ConvNeXt sudah memiliki preprocessing layer internal.
5. Support train/validation/test mode

## Code Skeleton
```python
# src/data_loader.py
import tensorflow as tf
import os

IMG_SIZE = 224
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

def get_class_names(data_dir):
    return sorted(os.listdir(data_dir))

def create_dataset(data_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE, 
                   shuffle=True, augment=False, normalize=False):
    """Create a tf.data pipeline from directory."""
    
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=shuffle,
        seed=42
    )
    
    # Normalize pixel values (ONLY if explicitly asked, otherwise skip for modern models)
    if normalize:
        normalization = tf.keras.layers.Rescaling(1./255)
        dataset = dataset.map(lambda x, y: (normalization(x), y), 
                              num_parallel_calls=AUTOTUNE)
    
    # Performance optimization
    if shuffle:
        dataset = dataset.shuffle(1000)
    dataset = dataset.prefetch(AUTOTUNE)
    
    return dataset
```

## Pipeline Flow
```
Raw Images → Decode JPEG → Resize (img_size) → [Normalize] → Batch (32) → Prefetch
```

## Estimated Time
~2 jam

## Dependencies
- Sprint 1 selesai (data split sudah ada)
- TensorFlow terinstall

## Notes
- `tf.data.AUTOTUNE` untuk otomatis optimize buffer sizes
- `prefetch` penting untuk overlap CPU dan GPU processing
- Jika memory terbatas, jangan gunakan `.cache()`
- **CRITICAL:** Biarkan `normalize=False` secara default. Biarkan Preprocessing layer dari tiap model yang meng-handle rescaling sesuai kebutuhan arsitektur masing-masing.
