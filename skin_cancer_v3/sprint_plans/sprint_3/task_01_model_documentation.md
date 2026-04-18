# Task 01 — Model Architecture Documentation (Updated: State-of-the-Art 2024-2025)

## Task Info

| Item | Detail |
|------|--------|
| **Sprint** | 3 |
| **Task ID** | S3-T01 |
| **Priority** | High |
| **Story Points** | 3 |
| **Status** | Belum Mulai |

## Description
Dokumentasi lengkap semua arsitektur model yang digunakan — **campuran model klasik yang terbukti dan model modern state-of-the-art (2024-2025)** yang telah di-benchmark pada dataset dermoskopi (ISIC, HAM10000).

> **Sumber Riset:** Kaggle ISIC 2024 Challenge solutions, PubMed/NIH papers, arXiv 2024-2025, GitHub implementations, Medium technical posts.

---

## Hasil Riset: Tren Model Terbaru untuk Skin Cancer Detection

### Temuan Utama dari Riset

1. **Transformer-based models** (Swin Transformer, DINOv2-ViT) sudah **terbukti outperform CNN klasik** pada ISIC dataset
2. **ConvNeXt** — CNN modern yang mengadopsi desain Transformer, **97%+ accuracy** pada HAM10000
3. **EfficientNetV2** — evolusi EfficientNet dengan **training 3-4× lebih cepat** dan akurasi lebih tinggi
4. **ISIC 2024 Kaggle Challenge winners** menggunakan: EfficientNetV2 + DINOv2-ViT + ConvNeXt + LightGBM ensemble
5. **DINOv2** — self-supervised ViT dari Meta AI, **96.48% accuracy** pada 31-class skin disease dataset

### Model Selection Criteria untuk Project Ini
- **Tersedia di `tf.keras.applications`** (ConvNeXt, EfficientNetV2) atau mudah di-import
- **Pretrained ImageNet weights** tersedia
- **Terbukti di skin cancer classification** melalui paper/Kaggle
- **Feasible** dengan dataset kita (2,239 images, 9 classes)
- **Campuran** — classic proven + modern SOTA

---

## Model Lineup Final (8 Model)

| # | Model | Generasi | Tahun | Kategori | Status |
|---|-------|----------|-------|----------|--------|
| 1 | EfficientNetV2-S | **Modern** | 2021 | CNN (NAS) | ISIC 2024 winner backbone |
| 2 | ConvNeXt-Tiny | **Modern** | 2022 | CNN (Transformer-inspired) | 93-97% on HAM10000 |
| 3 | DenseNet121 | Classic | 2016 | CNN (Dense) | Medical imaging champion |
| 4 | Swin Transformer (via KerasCV) | **Modern** | 2021 | Vision Transformer | 89-95% on ISIC |
| 5 | ResNet50V2 | Classic | 2016 | CNN (Residual) | Strong baseline |
| 6 | InceptionV3 | Classic | 2015 | CNN (Multi-scale) | Google Derm AI backbone |
| 7 | MobileNetV2 | Classic | 2018 | CNN (Mobile) | Deployment/lightweight |
| 8 | Xception | Classic | 2016 | CNN (Depthwise Sep.) | Texture specialist |

> **3 model baru ditambahkan**: EfficientNetV2-S, ConvNeXt-Tiny, Swin Transformer
> **1 model dihapus**: VGG16 (terlalu besar, tidak efisien, outperformed oleh semua model di atas)

---

## MODEL BARU #1: EfficientNetV2-S (2021)

**📄 Paper:** *"EfficientNetV2: Smaller Models and Faster Training"* — Tan & Le, Google Brain, **ICML 2021**

### Apa yang Baru vs EfficientNetB3?

| Aspek | EfficientNet (V1/B3) | EfficientNetV2-S |
|-------|---------------------|------------------|
| Training Speed | Baseline | **3-4× lebih cepat** |
| Building Block | MBConv saja | **MBConv + Fused-MBConv** |
| Scaling | Compound (fixed) | **Progressive Learning** |
| Regularization | Dropout | **Progressive Dropout + Augmentation** |
| ImageNet Top-1 | 84.0% (B3) | **84.9%** (V2-S) |
| Params | 12M (B3) | **21.5M** (V2-S) |
| Training Time | ~30h | **~10h** (3× faster!) |

### 🏗️ Architecture

```
Input (384×384×3) — or 300×300 for fine-tuning

EfficientNetV2-S Structure:
Stage 0: Conv3×3, 24 (stride 2)
Stage 1: Fused-MBConv1, k3×3, ch24  (×2)   ← NEW: Fused-MBConv di early stages
Stage 2: Fused-MBConv4, k3×3, ch48  (×4)   ← Fused = regular conv (faster!)
Stage 3: Fused-MBConv4, k3×3, ch64  (×4)
Stage 4: MBConv4,  k3×3, ch128, SE (×6)    ← Standard MBConv di deep stages
Stage 5: MBConv6,  k3×3, ch160, SE (×9)
Stage 6: MBConv6,  k3×3, ch256, SE (×15)
↓
Conv1×1, 1280 → GlobalAvgPool → Dense(num_classes)
```

**Fused-MBConv** = Menggabungkan expansion conv + depthwise conv menjadi single regular conv.
- Pada **early stages** (resolusi tinggi), regular conv lebih cepat di hardware modern
- Pada **late stages** (resolusi rendah), depthwise separable conv lebih efisien

### Specifications

| Property | Value |
|----------|-------|
| Total Parameters | 21.5M |
| Input Size | 384×384 (pretrained), 300×300 (fine-tune OK) |
| ImageNet Top-1 | 84.9% |
| Model Size | ~88 MB |
| Training Speed | 3-4× faster than V1 |
| TF Keras | `tf.keras.applications.EfficientNetV2S` |

### Bukti di Skin Cancer

| Source | Result |
|--------|--------|
| **ISIC 2024 Kaggle** (winners) | Backbone utama di ensemble pemenang |
| NIH PubMed 2024 | 88-99% accuracy pada HAM10000/ISIC (varies by setup) |
| EfficientNetV2-L fine-tuned | Hingga 96-99% accuracy pada task binary |
| Multi-class (9-class) | Expected ~87-92% |

### Mengapa Cocok untuk Project Ini
1. **ISIC 2024 champion backbone** — terbukti di kompetisi klasifikasi kanker kulit terbaru
2. **Progressive learning** — augmentasi dan image size naik bertahap selama training → mengurangi overfitting pada dataset kecil
3. **Fused-MBConv** — 3-4× lebih cepat training dari EfficientNet V1
4. **Native di TF/Keras** — `tf.keras.applications.EfficientNetV2S`, mudah digunakan
5. **Sweet spot** antara V2-S (compact) dan V2-L (monster)

### Kekurangan
- Lebih besar dari EfficientNetB3 (21.5M vs 12M params)
- Progressive learning membutuhkan custom training loop
- Pretrained pada 384×384, perlu adjust

### Implementation

```python
def build_efficientnetv2s(num_classes=9, input_shape=(300, 300, 3)):
    base_model = tf.keras.applications.EfficientNetV2S(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        include_preprocessing=True  # Built-in preprocessing!
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

---

## MODEL BARU #2: ConvNeXt-Tiny (2022)

**📄 Paper:** *"A ConvNet for the 2020s"* — Liu et al., Meta AI / Facebook Research, **CVPR 2022**

### Konsep Revolusioner
ConvNeXt menjawab pertanyaan: *"Apakah pure CNN masih bisa mengalahkan Vision Transformer jika kita adopt design principles-nya?"* **Jawabannya: YA!**

ConvNeXt di-build dengan cara **"modernize" ResNet50** step-by-step menggunakan teknik dari Swin Transformer:

```
ResNet50 (76.1%) 
  → Macro design changes (+2.5%)
    → ResNeXt-ify (grouped conv) (+0.6%)
      → Inverted bottleneck (+0.1%)
        → Large kernel (7×7) (+0.7%)
          → Micro design (LayerNorm, GELU, fewer activations) (+0.6%)
= ConvNeXt-T (82.1%)  ← +6% improvement, masih PURE CNN!
```

### 🏗️ Architecture

```
Input (224×224×3)
    ↓
Patchify Stem: Conv 4×4, stride 4, 96ch      ← Like ViT patch embedding!
    ↓
Stage 1: ConvNeXt Block × 3, ch=96
    DownSample (LayerNorm + Conv 2×2 stride 2)
Stage 2: ConvNeXt Block × 3, ch=192
    DownSample
Stage 3: ConvNeXt Block × 9, ch=384          ← Deepest stage (like Swin)
    DownSample
Stage 4: ConvNeXt Block × 3, ch=768
    ↓
GlobalAvgPool → LayerNorm → Dense(classes)

ConvNeXt Block:
  x → Depthwise Conv 7×7 → LayerNorm → Conv 1×1 (expand ×4)
    → GELU → Conv 1×1 (project back) → + x (residual)
```

### Desain yang Diadopsi dari Transformer

| Dari Transformer | Di ConvNeXt | Efeknya |
|------------------|-------------|---------|
| Patch embedding | Patchify stem (4×4 conv stride 4) | Non-overlapping patches |
| LayerNorm | LayerNorm (bukan BatchNorm) | Lebih stabil |
| GELU activation | GELU (bukan ReLU) | Smoother gradient |
| Fewer activations | Hanya 1 GELU per block | Lebih efisien |
| Large receptive field | 7×7 depthwise conv | Global-like context |
| Inverted bottleneck | Narrow → wide → narrow | Efficient compute |

### Specifications

| Property | Value |
|----------|-------|
| Total Parameters | 28.6M |
| Input Size | 224 × 224 × 3 |
| ImageNet Top-1 | **82.1%** (Tiny), 83.1% (Small) |
| Model Size | ~109 MB |
| TF Keras | `tf.keras.applications.ConvNeXtTiny` |

### Bukti di Skin Cancer

| Source | Result |
|--------|--------|
| ResearchGate 2024 | **96.3% accuracy** pada HAM10000 (ConvNeXtV2 + attention) |
| NIH 2024 study | **93-97%** sebagai individual classifier di skin lesion |
| ISIC 2024 Kaggle | Digunakan oleh top teams sebagai salah satu backbone |
| Benchmark vs CNN | Outperforms ResNet, DenseNet, EfficientNet V1 pada skin data |

### Mengapa Cocok untuk Project Ini
1. **93-97% accuracy** terbukti pada dataset dermoskopi — tertinggi di antara individual CNN
2. **Balanced precision/recall** — penting untuk medical diagnosis (mengurangi false negative DAN false positive)
3. **Modern design** → mengadopsi kelebihan Transformer tanpa overhead attention computation
4. **7×7 depthwise conv** memberikan receptive field yang besar → menangkap konteks global lesi
5. **Native TF/Keras** — `tf.keras.applications.ConvNeXtTiny`
6. **No special preprocessing** — ConvNeXt meng-handle normalization internally

### Kekurangan
- Lebih besar dari DenseNet121 (28.6M vs 8M)
- Patchify stem (stride 4) bisa kehilangan detail sangat halus
- Relatif baru, kurang studi di Indonesia untuk dermoskopi

### Implementation

```python
def build_convnext_tiny(num_classes=9, input_shape=(224, 224, 3)):
    base_model = tf.keras.applications.ConvNeXtTiny(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        include_preprocessing=True  # Built-in normalization
    )
    base_model.trainable = False
    
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.LayerNormalization()(x)  # ConvNeXt uses LN, not BN
    x = tf.keras.layers.Dense(512, activation='gelu')(x)  # GELU like ConvNeXt
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation='gelu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return tf.keras.Model(inputs, outputs, name='convnext_tiny_skin')
```

> **Catatan penting**: ConvNeXt menggunakan **LayerNorm** dan **GELU**, bukan BatchNorm dan ReLU seperti model lama. Classification head harus konsisten!

---

## MODEL BARU #3: Swin Transformer (2021)

**📄 Paper:** *"Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"* — Liu et al., Microsoft Research, **ICCV 2021 (Best Paper!)**

### Mengapa Transformer untuk Gambar?
Standard CNN hanya melihat "tetangga lokal" (3×3 atau 7×7). Transformer melihat **seluruh gambar sekaligus** melalui self-attention. Ini penting untuk skin cancer karena:
- Mendeteksi **asimetri** membutuhkan perbandingan sisi kiri vs kanan (global view)
- **Border irregularity** membutuhkan analisis seluruh boundary (bukan local patch)

Masalah ViT standar: Attention pada seluruh gambar = **O(n²)** complexity → sangat lambat.
**Solusi Swin:** Attention dalam **windows** yang bergeser (shifted) → **O(n)** complexity!

### 🏗️ Architecture

```
Input (224×224×3)
    ↓
Patch Partition: 4×4 patches → 56×56 × 96ch
    ↓
┌──────────────────────────────────────────────┐
│ Swin Transformer Block:                      │
│                                              │
│ 1. Window-based Multi-Head Self-Attention   │
│    (W-MSA) — attention dalam 7×7 windows    │
│                                              │
│ 2. Shifted Window MSA (SW-MSA)              │
│    windows digeser 3 pixel → cross-window    │
│    communication!                            │
│                                              │
│ + LayerNorm + MLP + Residual connections    │
└──────────────────────────────────────────────┘
    ↓
Stage 1: Swin Block × 2, ch=96,  res=56×56
Patch Merging (2×2 → downsample)
Stage 2: Swin Block × 2, ch=192, res=28×28
Patch Merging
Stage 3: Swin Block × 6, ch=384, res=14×14
Patch Merging
Stage 4: Swin Block × 2, ch=768, res=7×7
    ↓
GlobalAvgPool → LayerNorm → Dense(classes)
```

### Shifted Window — Key Innovation

```
Window MSA:           Shifted Window MSA:
┌──┬──┬──┬──┐        ┌─┬───┬───┬─┐
│  │  │  │  │        │ │   │   │ │
├──┼──┼──┼──┤   →    ├─┼───┼───┼─┤
│  │  │  │  │        │ │   │   │ │
├──┼──┼──┼──┤        ├─┼───┼───┼─┤
│  │  │  │  │        │ │   │   │ │
└──┴──┴──┴──┘        └─┴───┴───┴─┘

Regular windows        Windows shifted by (M/2, M/2)
→ No cross-window      → Cross-window connections!
  communication           (inter-window context)
```

### Specifications

| Property | Value |
|----------|-------|
| Total Parameters | 28.3M (Tiny) |
| Input Size | 224 × 224 × 3 |
| ImageNet Top-1 | **81.3%** (Tiny), 83.0% (Small) |
| Window Size | 7 × 7 |
| Heads | [3, 6, 12, 24] per stage |

### Bukti di Skin Cancer

| Source | Result |
|--------|--------|
| NIH 2024 (Enhanced Swin) | **89.36% accuracy** pada ISIC 2019 (9-class!) |
| Hybrid Swin + CNN | Hingga **98%+** pada HAM10000 |
| ISIC 2024 Kaggle | Banyak top solutions menggunakan Swin sebagai backbone |
| LoRA fine-tuning study | Efficient fine-tuning dengan parameter 90% lebih sedikit |

### Mengapa Cocok untuk Project Ini
1. **Global context** — Self-attention melihat seluruh gambar → mendeteksi asimetri lesi
2. **ICCV 2021 Best Paper** — proven arsitektur, bukan experimental
3. **89.36% pada ISIC 2019** — dataset yang **sama persis** dengan data kita (9 classes!)
4. **Hierarchical** — seperti CNN, menghasilkan feature maps multi-scale
5. **Shifted windows** — efisien dibanding ViT standar
6. Bisa diakses via **keras-cv** atau **timm** (PyTorch) / **tensorflow-hub**

### Kekurangan
- **Tidak native di `tf.keras.applications`** — perlu install `keras-cv` atau convert dari PyTorch
- Lebih lambat inference dari CNN (attention overhead)
- Membutuhkan lebih banyak data untuk optimal — bisa mitigasi dengan strong augmentation
- GPU memori lebih tinggi

### Implementation (via keras_cv)

```python
# Option 1: keras_cv (recommended)
# pip install keras-cv
import keras_cv

def build_swin_tiny(num_classes=9, input_shape=(224, 224, 3)):
    backbone = keras_cv.models.SwinTransformerTiny(
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
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return tf.keras.Model(inputs, outputs, name='swin_tiny_skin')

# Option 2: TensorFlow Hub
import tensorflow_hub as hub
swin_url = "https://tfhub.dev/sayakpaul/swin_tiny_patch4_window7_224/1"
```

> **Fallback plan:** Jika Swin Transformer sulit diimplementasi (dependency issues), maka ConvNeXt-Tiny menjadi representasi dari "Transformer-inspired architecture" — karena ConvNeXt mengadopsi banyak desain Swin.

---

## Master Comparison Table (Updated dengan Model Baru)

### Modern vs Classic Lineup

| Model | Gen. | Params | Input | ImageNet | Skin Cancer* | Speed | TF Native |
|-------|------|--------|-------|----------|-------------|-------|-----------|
| **EfficientNetV2-S** | 2021 | 21.5M | 300² | 84.9% | 87-92% || `EfficientNetV2S` |
| **ConvNeXt-Tiny** | 2022 | 28.6M | 224² | 82.1% | 93-97% || `ConvNeXtTiny` |
| **Swin Transformer** | 2021 | 28.3M | 224² | 81.3% | 89-95% || keras-cv |
| DenseNet121 | 2016 | 8.0M | 224² | 92.3%† | 80-88% || `DenseNet121` |
| ResNet50V2 | 2016 | 25.6M | 224² | 93.0%† | 78-85% || `ResNet50V2` |
| InceptionV3 | 2015 | 23.9M | 299² | 93.7%† | 80-87% || `InceptionV3` |
| MobileNetV2 | 2018 | 3.4M | 224² | 90.1%† | 75-82% || `MobileNetV2` |
| Xception | 2016 | 22.9M | 299² | 94.5%† | 82-88% || `Xception` |

*Skin Cancer = expected accuracy range (9-class, berdasarkan literatur 2024)
†ImageNet Top-5 (classic models reported Top-5)

### Performance Tier Prediction

| Tier | Models | Expected Accuracy |
|------|--------|-------------------|
| **S-Tier** (SOTA) | ConvNeXt-Tiny, EfficientNetV2-S | 87-97% |
| **A-Tier** (Excellent) | Swin Transformer, DenseNet121, Xception | 82-90% |
| **B-Tier** (Good) | ResNet50V2, InceptionV3 | 78-87% |
| **C-Tier** (Baseline+) | MobileNetV2 | 75-82% |

---

## Why This Model Lineup is Optimal

### Diversity of Approaches

```
                    Global Context
                         ↑
            Swin         |
         Transformer     |
                         |
     ConvNeXt-Tiny ------+------ EfficientNetV2-S
     (CNN+Transformer    |       (NAS-optimized CNN)
      hybrid design)     |
                         |
     Xception            |       DenseNet121
     (Texture)           |       (Feature Reuse)
                         |
     ResNet50V2          |       InceptionV3
     (Skip Connections)  |       (Multi-scale)
                         |
                    Local Features
                         
        MobileNetV2 ←── Lightweight axis ──→ Heavy models
```

### Coverage Checklist

| Requirement | Covered By |
|-------------|-----------|
| SOTA accuracy (proven 2024) | ConvNeXt-Tiny, EfficientNetV2-S |
| Global attention (for asymmetry detection) | Swin Transformer |
| Texture recognition (ABCDE rule) | Xception, ConvNeXt |
| Feature reuse (small dataset) | DenseNet121 |
| Multi-scale features | InceptionV3, EfficientNetV2 |
| Mobile deployment | MobileNetV2 |
| Strong baseline comparison | ResNet50V2 |
| NAS-optimized | EfficientNetV2-S |
| Transformer-inspired CNN | ConvNeXt-Tiny |
| Pure Transformer | Swin Transformer |

---

## 📚 References (2024-2025)

### Papers
1. Tan, M. & Le, Q. (2021). *EfficientNetV2: Smaller Models and Faster Training.* ICML 2021
2. Liu, Z. et al. (2022). *A ConvNet for the 2020s.* CVPR 2022
3. Liu, Z. et al. (2021). *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.* ICCV 2021 (**Best Paper**)
4. Oquab, M. et al. (2024). *DINOv2: Learning Robust Visual Features without Supervision.* TMLR 2024

### Skin Cancer Specific Studies (2024)
5. NIH/PubMed 2024 — Enhanced Swin Transformer: **89.36% on ISIC 2019** (9-class)
6. ResearchGate 2024 — ConvNeXtV2 + separable self-attention: **96.3% on HAM10000**
7. arXiv 2024 — DINOv2 fine-tuned: **96.48% accuracy, 0.97 F1** pada 31-class skin disease
8. Kaggle ISIC 2024 — Top solutions: EfficientNetV2 + DINOv2-ViT + ConvNeXt + LightGBM ensemble

### Implementation Sources
9. TensorFlow: `tf.keras.applications` — ConvNeXtTiny, EfficientNetV2S (native)
10. Keras-CV: SwinTransformer (via `keras_cv.models`)
11. Kaggle Notebooks: Multiple high-scoring implementations for ISIC 2024
