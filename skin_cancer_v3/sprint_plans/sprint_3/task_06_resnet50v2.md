# Task 06 — ResNet50V2: Training & Evaluation

## Task Info
| Item | Detail |
|------|--------|
| **Sprint** | 3 |
| **Task ID** | S3-T06 |
| **Priority** | High |
| **Story Points** | 3 |
| **Status** | Belum Mulai |

## Description
Melatih ResNet50V2 sebagai **strong baseline** transfer learning. Skip connections memungkinkan fitur low-level (warna, edge) dan high-level (shape, pattern) digunakan bersamaan. Pre-activation design (V2) memberikan gradient flow yang lebih baik.

**📄 Paper:** *"Identity Mappings in Deep Residual Networks"* — He et al., 2016

## 🏗️ Architecture
```
ResNet50V2 (Frozen) → GlobalAveragePooling2D → BatchNorm
→ Dense(512, relu) → Dropout(0.5) → Dense(256, relu) → Dropout(0.3) → Dense(9, softmax)
```
**Input: 224×224×3** | **Params: 25.6M**

## Acceptance Criteria
- [ ] Phase 1: Frozen (10 epochs), Phase 2: Unfreeze last 30 layers (25 epochs)
- [ ] **Target: accuracy ≥ 80%**
- [ ] Model disimpan: `models/resnet50v2_best.h5`

## Implementation
```python
def build_resnet50v2(num_classes=9, input_shape=(224, 224, 3)):
    base_model = tf.keras.applications.ResNet50V2(
        include_top=False, weights='imagenet', input_shape=input_shape
    )
    base_model.trainable = False
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.resnet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs, name='resnet50v2_skin')
```

## Expected: Accuracy 78-85%, F1 0.75-0.83
## Estimated Time: ~3 jam
## Dependencies: S2-T01
