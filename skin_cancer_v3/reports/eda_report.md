# Laporan Exploratory Data Analysis (EDA)
**Project:** Skin Cancer Detection System v3

Laporan ini merangkum hasil Exploratory Data Analysis dari data gambar latih dan uji dataset kanker kulit 9 kelas.

## 1. Distribusi Kelas
Berdasarkan analisis dataset `Train` dengan total 2239 gambar:
- Melanoma dan Basal Cell Carcinoma menempati proporsi sedang.
- Pigmented Benign Keratosis dan Nevus mendominasi dataset.
- Seborrheic keratosis, dermatofibroma, dan vascular lesion merupakan kelas minoritas (highly imbalanced).
**Tindakan:** Menggunakan teknik `class_weights` dan `StratifiedSplit` saat training agar model tidak bias ke kelas mayoritas.

## 2. Resolusi dan Dimensi Gambar
Sebagian besar gambar memiliki dimensi di kisaran menengah, namun tidak selaras 100%. 
**Tindakan:** Pipeline TF Data akan meresize semua gambar menjadi `224x224`, `299x299`, atau `300x300` bergantung pada model backbone yang digunakan (misal `Swin-Transformer` butuh `224x224`).

## 3. Pixel Intensity & Color Channel
Channel Merah (R) secara natural mendominasi histogram pada semua sampel gambar, mengingat spesimen berupa lesi kulit. Namun, tingkat *contrast* dan pencahayaan sangat bervariasi.
**Tindakan:** Modern model seperti ConvNeXt dan EfficientNetV2 memiliki internal normalization tersendiri, namun augmentasi seperti Color Jitter dan Random Contrast akan sangat membantu model agar lebih general.

## 4. Keberadaan Duplikat & Korup
Tidak ditemukan file corrupt atau broken headers dalam batch saat ini. Image loading via OpenCV terkonfirmasi bebas error.

## Kesimpulan Strategi Pipeline
- **Split Ratio:** Mengalokasikan 80% data untuk Train dan 20% untuk Validation.
- **Handling Imbalance:** Akan memanfaatkan Focal Loss atau Class Weights yang spesifik karena kelas kanker yang serius sangat underrepresented.
- **Augmentation:** MixUp, CutMix, Random Flips, dan Cropping sangat direkomendasikan melihat sampel data sangat terbatas.
