# Task 09 — Xception: Training & Evaluation

## Task Info
| Item | Detail |
|------|--------|
| **Sprint** | 3 |
| **Task ID** | S3-T09 |
| **Priority** | Medium |
| **Story Points** | 3 |
| **Status** | Belum Mulai |

## Description
Melatih Xception — **texture recognition specialist** berkat depthwise separable convolutions. Dermoskopi sangat bergantung pada analisis tekstur kulit (ABCDE rule). Dibuat oleh **François Chollet** (creator of Keras).

**📄 Paper:** *"Xception: Deep Learning with Depthwise Separable Convolutions"* — Chollet, CVPR 2017

## 🏗️ Architecture
```
Xception (Frozen) → GlobalAveragePooling2D → BatchNorm
→ Dense(512, relu) → Dropout(0.5) → Dense(256, relu) → Dropout(0.3) → Dense(9, softmax)
```
**Input: 299×299×3** | **Params: 22.9M**

## Acceptance Criteria
- [ ] Phase 1: Frozen (10 epochs), Phase 2: Unfreeze exit flow (25 epochs)
- [ ] **Target: accuracy ≥ 83%**
- [ ] Model disimpan: `models/xception_best.h5`

## Implementation
```python
def build_xception(num_classes=9, input_shape=(299, 299, 3)):
    base_model = tf.keras.applications.Xception(
        include_top=False, weights='imagenet', input_shape=input_shape
    )
    base_model.trainable = False
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.xception.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs, name='xception_skin')
```

## Expected: Accuracy 82-88%, F1 0.80-0.86
## Estimated Time: ~3 jam
## Dependencies: S2-T01 (adjust input 299×299)
