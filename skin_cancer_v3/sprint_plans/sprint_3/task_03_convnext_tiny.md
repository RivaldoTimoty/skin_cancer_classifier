# Task 03 — ConvNeXt-Tiny: Training & Evaluation

## Task Info
| Item | Detail |
|------|--------|
| **Sprint** | 3 |
| **Task ID** | S3-T03 |
| **Priority** | High |
| **Story Points** | 4 |
| **Status** | Belum Mulai |

## Description
Melatih **ConvNeXt-Tiny** — CNN modern (2022) yang mengadopsi design principles dari Swin Transformer namun tetap pure CNN. Terbukti **93-97% accuracy pada HAM10000** dan menjadi backbone di top ISIC 2024 solutions.

**📄 Paper:** *"A ConvNet for the 2020s"* — Liu et al., Meta AI, CVPR 2022

## 🏗️ Architecture
```
ConvNeXt-Tiny (Frozen) → GlobalAveragePooling2D → LayerNorm
→ Dense(512, gelu) → Dropout(0.5) → Dense(256, gelu) → Dropout(0.3)
→ Dense(9, softmax)
```
**Input: 224×224×3** | **Params: 28.6M** | **Built-in preprocessing:**

## 🔑 What Makes ConvNeXt Special
- **Patchify Stem** (4×4 conv stride 4) — like ViT patch embedding
- **7×7 Depthwise Conv** — large receptive field for global context
- **LayerNorm** instead of BatchNorm — more stable training
- **GELU** instead of ReLU — smoother gradients
- **Inverted Bottleneck** — narrow → wide → narrow
- Outperforms Swin Transformer on ImageNet while being pure CNN!

## Acceptance Criteria
- [ ] Model di-load: `tf.keras.applications.ConvNeXtTiny(weights='imagenet')`
- [ ] Phase 1: Frozen training (10 epochs, LR=1e-3)
- [ ] Phase 2: Unfreeze Stage 3+4, fine-tune (25 epochs, LR=1e-5)
- [ ] Phase 3: Full fine-tune (10 epochs, LR=1e-6)
- [ ] **Target: accuracy ≥ 87%, balanced precision/recall**
- [ ] Model disimpan: `models/convnext_tiny_best.h5`

## Implementation
```python
def build_convnext_tiny(num_classes=9, input_shape=(224, 224, 3)):
    base_model = tf.keras.applications.ConvNeXtTiny(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        include_preprocessing=True
    )
    base_model.trainable = False
    
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.LayerNormalization()(x)  # ConvNeXt uses LN!
    x = tf.keras.layers.Dense(512, activation='gelu')(x)  # GELU!
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation='gelu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs, name='convnext_tiny_skin')
```

> **PENTING:** ConvNeXt menggunakan **LayerNorm + GELU**, bukan BatchNorm + ReLU!

## Expected Performance
| Metric | Expected |
|--------|----------|
| Accuracy | **87-93%** |
| F1-Score | **0.85-0.91** |
| Balanced Precision/Recall | Kelebihan utama |

## Estimated Time: ~4 jam
## Dependencies: S2-T01 (data pipeline)
