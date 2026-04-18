# Task 07 — InceptionV3: Training & Evaluation

## Task Info
| Item | Detail |
|------|--------|
| **Sprint** | 3 |
| **Task ID** | S3-T07 |
| **Priority** | Medium |
| **Story Points** | 3 |
| **Status** | Belum Mulai |

## Description
Melatih InceptionV3 dengan **multi-scale feature extraction** — penting untuk lesi kulit yang punya detail lokal (tekstur) DAN pola global (bentuk, asimetri). Digunakan dalam **Google Derm AI** (Nature 2017).

**📄 Paper:** *"Rethinking the Inception Architecture"* — Szegedy et al., 2015

## 🏗️ Architecture
```
InceptionV3 (Frozen) → GlobalAveragePooling2D → BatchNorm
→ Dense(512, relu) → Dropout(0.5) → Dense(256, relu) → Dropout(0.3) → Dense(9, softmax)
```
**Input: 299×299×3** | **Params: 23.9M**

## Acceptance Criteria
- [ ] Data pipeline adjust ke 299×299
- [ ] Phase 1: Frozen (10 epochs), Phase 2: Unfreeze dari `mixed7` (25 epochs)
- [ ] **Target: accuracy ≥ 82%**
- [ ] Model disimpan: `models/inceptionv3_best.h5`

## Implementation
```python
def build_inceptionv3(num_classes=9, input_shape=(299, 299, 3)):
    base_model = tf.keras.applications.InceptionV3(
        include_top=False, weights='imagenet', input_shape=input_shape
    )
    base_model.trainable = False
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.inception_v3.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs, name='inceptionv3_skin')
```

## Expected: Accuracy 80-87%, F1 0.78-0.85
## Estimated Time: ~3 jam
## Dependencies: S2-T01 (adjust input 299×299)
