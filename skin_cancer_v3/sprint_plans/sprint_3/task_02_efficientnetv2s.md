# Task 02 — EfficientNetV2-S: Training & Evaluation

## Task Info
| Item | Detail |
|------|--------|
| **Sprint** | 3 |
| **Task ID** | S3-T02 |
| **Priority** | High |
| **Story Points** | 4 |
| **Status** | Belum Mulai |

## Description
Melatih **EfficientNetV2-S** — evolusi EfficientNet yang **3-4× lebih cepat training** dengan akurasi lebih tinggi. Model ini menjadi **backbone utama pemenang ISIC 2024 Kaggle Challenge**. Menggunakan Fused-MBConv di early stages dan progressive learning.

**📄 Paper:** *"EfficientNetV2: Smaller Models and Faster Training"* — Tan & Le, Google Brain, ICML 2021

## 🏗️ Architecture
```
EfficientNetV2-S (Frozen) → GlobalAveragePooling2D → BatchNorm
→ Dense(512, relu) → Dropout(0.5) → Dense(256, relu) → Dropout(0.3)
→ Dense(9, softmax)
```
**Input: 300×300×3** | **Params: 21.5M** | **Built-in preprocessing:**

## 🔑 Key Advantages vs EfficientNet V1
- **Fused-MBConv** di early stages → 3-4× faster training
- **Progressive Learning** → augmentasi naik bertahap, mengurangi overfitting
- **ImageNet Top-1: 84.9%** (vs 84.0% B3)
- **ISIC 2024 winner backbone**

## Acceptance Criteria
- [ ] Model di-load: `tf.keras.applications.EfficientNetV2S(weights='imagenet')`
- [ ] Phase 1: Frozen training (10 epochs, LR=1e-3)
- [ ] Phase 2: Unfreeze last 60 layers, fine-tune (30 epochs, LR=1e-5)
- [ ] Phase 3: Full fine-tune (10 epochs, LR=1e-6)
- [ ] **Target: accuracy ≥ 88%, melanoma recall ≥ 90%**
- [ ] Model disimpan: `models/efficientnetv2s_best.h5`

## Implementation
```python
def build_efficientnetv2s(num_classes=9, input_shape=(300, 300, 3)):
    base_model = tf.keras.applications.EfficientNetV2S(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        include_preprocessing=True
    )
    base_model.trainable = False
    
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs, name='efficientnetv2s_skin')
```

## Expected Performance
| Metric | Expected |
|--------|----------|
| Accuracy | **87-92%** |
| F1-Score | **0.85-0.90** |
| Melanoma Recall | **88-93%** |
| Training Time | ~2-3 jam (3× faster than V1!) |

## Notes
- `include_preprocessing=True` → jangan tambahkan Rescaling/normalization manual
- Input 300×300 (bukan 384 pretrained), masih optimal untuk fine-tuning
- Progressive learning (increase image size selama training) bisa diterapkan secara manual

## Estimated Time: ~4 jam
## Dependencies: S2-T01 (data pipeline)
