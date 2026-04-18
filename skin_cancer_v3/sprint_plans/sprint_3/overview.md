# Sprint 3 — Multi-Model Exploration & Optimization

## Sprint Goal
*Membuktikan dan melatih 8 arsitektur deep learning berbeda (klasik & SOTA modern), mencari model dengan akurasi dan interpretability terbaik (khususnya untuk Melanoma), dan mengekspor model pemenang ke production.*

## Sprint Info

| Item | Detail |
|------|--------|
| **Sprint** | 3 |
| **Durasi** | 5-7 hari |
| **Total Story Points** | 62 SP |
| **Start Date** | TBD |
| **End Date** | TBD |
| **Status** | Belum Mulai |

## Mengapa 8 Model?
Berdasarkan riset terbaru (Kaggle ISIC 2024, paper NIH 2024-2025), tidak ada satu arsitektur "silver bullet" untuk lesi kulit. Model modern (Swin Transformer, ConvNeXt) memberikan global context yang superior, sementara CNN klasik (DenseNet, Xception) ahli di tekstur lokal. Ensemble dari keduanya sering kali menghasilkan state-of-the-art accuracy (>95%).

### The Model Lineup
1. **EfficientNetV2-S** (ISIC 2024 Winner Backbone, NAS-optimized)
2. **ConvNeXt-Tiny** (Modern CNN, Transformer-inspired design)
3. **Swin Transformer** (ICCV 2021 Best Paper, Global attention)
4. **DenseNet121** (Klasik, Feature reuse, Medical imaging favorite)
5. **ResNet50V2** (Klasik, Residual connections, Strong baseline)
6. **InceptionV3** (Klasik, Multi-scale features)
7. **MobileNetV2** (Klasik, Lightweight untuk deployment)
8. **Xception** (Klasik, Texture recognition specialist)

## Task List

| # | Task | Priority | SP | Status |
|---|------|----------|-----|--------|
| 1 | [Model Architecture Documentation (Updated SOTA)](./task_01_model_documentation.md) | High | 3 | |
| 2 | [EfficientNetV2-S: Training & Evaluation](./task_02_efficientnetv2s.md) | High | 4 | |
| 3 | [ConvNeXt-Tiny: Training & Evaluation](./task_03_convnext_tiny.md) | High | 4 | |
| 4 | [Swin Transformer: Training & Evaluation](./task_04_swin_transformer.md) | High | 5 | |
| 5 | [DenseNet121: Training & Evaluation](./task_05_densenet121.md) | High | 3 | |
| 6 | [ResNet50V2: Training & Evaluation](./task_06_resnet50v2.md) | High | 3 | |
| 7 | [InceptionV3: Training & Evaluation](./task_07_inceptionv3.md) | Medium | 3 | |
| 8 | [MobileNetV2: Training & Evaluation](./task_08_mobilenetv2.md) | Medium | 3 | |
| 9 | [Xception: Training & Evaluation](./task_09_xception.md) | Medium | 3 | |
| 10 | [Fine-Tuning Strategy (All Models)](./task_10_fine_tuning.md) | High | 3 | |
| 11 | [Hyperparameter Tuning (Top 3)](./task_11_hyperparameter_tuning.md) | High | 5 | |
| 12 | [Advanced Augmentation (MixUp/CutMix)](./task_12_advanced_augmentation.md) | Medium | 3 | |
| 13 | [Ensemble Model Strategy](./task_13_ensemble_model.md) | Medium | 5 | |
| 14 | [Per-Class Performance Analysis](./task_14_per_class_analysis.md) | High | 2 | |
| 15 | [ROC-AUC Curve per Model](./task_15_roc_auc.md) | Medium | 2 | |
| 16 | [Grad-CAM Visualization (Explainability)](./task_16_gradcam.md) | Medium | 3 | |
| 17 | [Final Model Comparison Report](./task_17_model_comparison.md) | High | 3 | |
| 18 | [Export Best Model to .h5 & ONNX](./task_18_export_model.md) | High | 2 | |

## Definition of Done
- [ ] 8 model arsitektur berhasil dilatih dan dievaluasi.
- [ ] Dokumentasi komprehensif metrik performa (Accuracy, Precision, Recall, F1-Score).
- [ ] "Melanoma Recall" dihitung dan dioptimasi.
- [ ] Grad-CAM diimplementasikan untuk validasi diagnosis model.
- [ ] 1 Model / Ensemble terbaik terpilih dan siap diekspor.
- [ ] Weights model disimpan ke `/models/` lengkap dengan `model_config.json`.

## Dependencies
- Sprint 2 harus selesai 100% (pipeline `tf.data.Dataset` siap).
- Akses ke GPU / Tensor Processing Unit (TPU) untuk mengurangi waktu training secara drastis!
