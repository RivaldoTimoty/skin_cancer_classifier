# Task 05 — DenseNet121: Training & Evaluation

## Task Info
| Item | Detail |
|------|--------|
| **Sprint** | 3 |
| **Task ID** | S3-T05 |
| **Priority** | High |
| **Story Points** | 3 |
| **Status** | Belum Mulai |

## Description
Melatih DenseNet121 — arsitektur **favorit #1 medical imaging** (CheXNet, DermNet). Dense connectivity memastikan feature reuse maximum. Hanya 8M params namun performa setara model 25M+. **Ideal untuk dataset kecil (2,239 images).**

**📄 Paper:** *"Densely Connected Convolutional Networks"* — Huang et al., CVPR 2017

## 🏗️ Architecture
```
DenseNet121 (Frozen) → GlobalAveragePooling2D → BatchNorm
→ Dense(512, relu) → Dropout(0.5) → Dense(256, relu) → Dropout(0.3) → Dense(9, softmax)
```
**Input: 224×224×3** | **Params: 8.0M** | **Size: 33 MB**

## 🔑 Why DenseNet for Medical Imaging
- **Feature reuse** → fitur warna/tekstur dari early layers langsung diakses oleh classifier
- **Parameter efficient** → 8M params, tapi performa setara 25M+ models (ResNet)
- **CheXNet** (Stanford) = DenseNet121 → radiologist-level pneumonia detection
- **Implicit regularization** → mengurangi overfitting pada dataset kecil

## Acceptance Criteria
- [ ] DenseNet121 di-load: `tf.keras.applications.DenseNet121(weights='imagenet')`
- [ ] Phase 1: Frozen (10 epochs), Phase 2: Unfreeze block 4 (30 epochs)
- [ ] **Target: accuracy ≥ 83%**
- [ ] Model disimpan: `models/densenet121_best.h5`

## Implementation
```python
def build_densenet121(num_classes=9, input_shape=(224, 224, 3)):
    base_model = tf.keras.applications.DenseNet121(
        include_top=False, weights='imagenet', input_shape=input_shape
    )
    base_model.trainable = False
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.densenet.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs, name='densenet121_skin')
```

## Expected: Accuracy 80-88%, F1 0.78-0.86
## Estimated Time: ~3 jam
## Dependencies: S2-T01
