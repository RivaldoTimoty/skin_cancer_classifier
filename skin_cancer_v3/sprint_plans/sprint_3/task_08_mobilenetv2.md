# Task 08 — MobileNetV2: Training & Evaluation

## Task Info
| Item | Detail |
|------|--------|
| **Sprint** | 3 |
| **Task ID** | S3-T08 |
| **Priority** | Medium |
| **Story Points** | 3 |
| **Status** | Belum Mulai |

## Description
Melatih MobileNetV2 sebagai **model ringan untuk deployment/mobile**. Hanya 3.4M params (14MB). Penting untuk skenario telemedicine — pasien scan lesi kulit di smartphone tanpa internet.

**📄 Paper:** *"MobileNetV2: Inverted Residuals and Linear Bottlenecks"* — Sandler et al., 2018

## 🏗️ Architecture
```
MobileNetV2 (Frozen) → GlobalAveragePooling2D → BatchNorm
→ Dense(256, relu) → Dropout(0.5) → Dense(128, relu) → Dropout(0.3) → Dense(9, softmax)
```
**Input: 224×224×3** | **Params: 3.4M** | **Size: 14 MB** | **Inference: ~50ms CPU**

## Acceptance Criteria
- [ ] Phase 1: Frozen (10 epochs), Phase 2: Unfreeze last 30 layers (20 epochs)
- [ ] Inference time benchmark (target < 100ms/image on CPU)
- [ ] TFLite conversion test
- [ ] Model disimpan: `models/mobilenetv2_best.h5` + `models/mobilenetv2.tflite`

## Implementation
```python
def build_mobilenetv2(num_classes=9, input_shape=(224, 224, 3)):
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False, weights='imagenet', input_shape=input_shape, alpha=1.0
    )
    base_model.trainable = False
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs, name='mobilenetv2_skin')
```

## Expected: Accuracy 75-82%, F1 0.72-0.80, Inference <100ms CPU
## Estimated Time: ~2 jam
## Dependencies: S2-T01
