# 🩺 Skin Cancer Classification (ISIC 9-Class)

## 📌 Project Overview
This project is an end-to-end, multi-class skin cancer classification pipeline built using **PyTorch** and **EfficientNet-B3**. The model is trained on the International Skin Imaging Collaboration (ISIC) dataset and classifies dermoscopic images into 9 distinct lesion categories.

The pipeline is fully modularized and includes exploratory data analysis, stratified data splitting, advanced data augmentation (via Albumentations), Test Time Augmentation (TTA), a CLI inference tool, and a **Streamlit Web Application** for user-friendly deployment.

---

## 🎯 Classes
1. Actinic Keratosis
2. Basal Cell Carcinoma
3. Dermatofibroma
4. Melanoma
5. Nevus (Mole)
6. Pigmented Benign Keratosis
7. Seborrheic Keratosis
8. Squamous Cell Carcinoma
9. Vascular Lesion

---

## 🚀 Key Features
- **State-of-the-Art Backbone**: Utilizes `EfficientNet-B3` natively fine-tuned for high-resolution images ($300 \times 300$).
- **Advanced Augmentation**: Uses `Albumentations` for affine transformations, color jittering, and Gaussian noise to prevent overfitting.
- **Explainable AI (XAI)**: Implements **Grad-CAM** (Gradient-weighted Class Activation Mapping) to visualize which areas of the lesion the model focuses on.
- **Test Time Augmentation**: Employs TTA (Original, Horz-Flip, Vert-Flip, 90° Rotate) to stabilize testing and inference accuracy.
- **Medical Web App Deployment**: A responsive, clean interface built with `Streamlit` that allows users to upload images and instantly get clinical predictions along with interpretability maps.

---

## 🛠️ Project Structure
```
skin_cancer_classifier/
│
├── app.py                   # Streamlit Web Application
├── colab_training.ipynb     # Jupyter Notebook for Cloud GPU Training
├── config.py                # Hyperparameters, paths, and constants
├── dataset.py               # Custom PyTorch Dataset & Albumentations pipelines
├── eda.py                   # Exploratory Data Analysis & visual plots
├── evaluate.py              # Metrics, Confusion Matrix, and ROC AUC generation
├── gradcam.py               # Explainability visualization logic
├── inference.py             # CLI Inference script with TTA support
├── main.py                  # Primary orchestrator for the local execution
├── model.py                 # EfficientNet architecture builder
├── optimize_pipeline.py     # Main training script for the final optimized model
├── train.py                 # Core training loop, early stopping, and LR scheduler
│
└── outputs/                 # Directory for models, EDA plots, and Evaluation Reports
```

---

## 📊 Model Performance

*(The metrics below are from our final 50-Epoch EfficientNet-B3 run out of Google Colab)*

| Metric | Score |
|--------|-------|
| **Accuracy** | `77.38%` |
| **Precision (Macro)** | `68.49%` |
| **Recall (Macro)** | `67.73%` |
| **F1-Score (Macro)** | `67.14%` |

> 📁 *Detailed ROC curves, per-class metrics, and the Confusion Matrix can be found in `outputs/evaluation_report.md` after training.*

---

## 💻 Installation & Usage

### 1. Requirements
Ensure you have Python 3.10+ installed.

```bash
pip install -r requirements.txt
```

### 2. To Run the Web Application
Make sure you have downloaded or trained the `best_model_optimized.pt` and placed it inside `%PROJECT_ROOT%/skin_cancer_classifier/outputs/models/`.

Then, launch Streamlit:
```bash
streamlit run skin_cancer_classifier/app.py
```

### 3. To Run CLI Inference
You can test any single image from your terminal using TTA and Grad-CAM:
```bash
python skin_cancer_classifier/inference.py --image_path path/to/image.jpg
```

---

## 🧠 Training your own Model

### Option A: Cloud GPU (Recommended)
1. Open Google Colab.
2. Upload `colab_training.ipynb`.
3. Set Runtime Hardware Accelerator to **T4 GPU**.
4. Run all blocks to automatically download the dataset, train for 50 epochs, and export the `.pt` file.

### Option B: Local CPU / GPU
Execute the optimization pipeline natively:
```bash
python skin_cancer_classifier/optimize_pipeline.py
```

---

## 📝 Disclaimer
This tool is a deep-learning portfolio project and is **not** to be used for actual medical diagnosis. Always consult a certified dermatologist for professional medical advice.
