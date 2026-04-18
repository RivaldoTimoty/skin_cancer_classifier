# Task 04 — Swin Transformer: Training & Evaluation

## Task Info
| Item | Detail |
|------|--------|
| **Sprint** | 3 |
| **Task ID** | S3-T04 |
| **Priority** | High |
| **Story Points** | 5 |
| **Status** | Belum Mulai |

## Description
Melatih **Swin Transformer** — Vision Transformer terbaik untuk image classification (**ICCV 2021 Best Paper**). Menggunakan Shifted Window self-attention yang efisien. Terbukti **89.36% pada ISIC 2019 (9-class, dataset yang sama dengan kita!)**.

**📄 Paper:** *"Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"* — Liu et al., Microsoft Research, ICCV 2021

## 🏗️ Architecture
```
Swin-Tiny (Frozen) → GlobalAveragePooling2D → LayerNorm
→ Dense(512, gelu) → Dropout(0.5) → Dense(256, gelu) → Dropout(0.3)
→ Dense(9, softmax)
```
**Input: 224×224×3** | **Params: 28.3M** | **Window Size: 7×7**

## 🔑 Why Transformer for Skin Cancer
- **Self-attention** melihat seluruh gambar → mendeteksi **asimetri lesi** (ABCDE rule "A")
- **Border irregularity** → perlu analisis seluruh boundary, bukan lokal saja
- **Shifted windows** → cross-window information flow (efisien O(n) bukan O(n²))
- **89.36% pada ISIC 2019** — dataset yang sama (9 classes)!

## Acceptance Criteria
- [ ] Swin Transformer di-load via keras-cv atau TF Hub
- [ ] Phase 1: Frozen training (10 epochs, LR=1e-3)
- [ ] Phase 2: Unfreeze Stage 3+4, fine-tune (25 epochs, LR=1e-5)
- [ ] **Target: accuracy ≥ 85%**
- [ ] Perbandingan attention maps vs CNN Grad-CAM
- [ ] Model disimpan: `models/swin_tiny_best.h5`

## Implementation
```python
# Option 1: keras-cv (recommended)
# pip install keras-cv tensorflow-addons
import keras_cv

def build_swin_tiny(num_classes=9, input_shape=(224, 224, 3)):
    # Load pretrained Swin backbone
    backbone = keras_cv.models.SwinTiny(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
    )
    backbone.trainable = False
    
    inputs = tf.keras.Input(shape=input_shape)
    x = backbone(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dense(512, activation='gelu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation='gelu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs, name='swin_tiny_skin')

# Option 2: TensorFlow Hub
import tensorflow_hub as hub
def build_swin_hub(num_classes=9):
    swin_url = "https://tfhub.dev/sayakpaul/swin_tiny_patch4_window7_224/1"
    inputs = tf.keras.Input(shape=(224, 224, 3))
    swin_layer = hub.KerasLayer(swin_url, trainable=False)
    x = swin_layer(inputs)
    x = tf.keras.layers.Dense(512, activation='gelu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs, name='swin_hub_skin')
```

## Fallback Plan
Jika keras-cv / TF Hub tidak bisa diinstall:
1. Gunakan **PyTorch timm** → convert ke ONNX → load di TF
2. Atau gunakan **ConvNeXt-Tiny** sebagai pengganti (mengadopsi desain Swin)
3. Atau gunakan **HuggingFace Transformers** `AutoModelForImageClassification`

## Expected Performance
| Metric | Expected |
|--------|----------|
| Accuracy | **85-92%** |
| F1-Score | **0.83-0.90** |
| Asimetri detection | Kelebihan utama (global attention) |

## Estimated Time: ~5 jam (termasuk setup keras-cv)
## Dependencies: S2-T01 (data pipeline), keras-cv installation
