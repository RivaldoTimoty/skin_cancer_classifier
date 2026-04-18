# Task 02 — Install Dependencies

## Task Info

| Item | Detail |
|------|--------|
| **Sprint** | 0 |
| **Task ID** | S0-T02 |
| **Priority** | High |
| **Story Points** | 1 |
| **Status** | Belum Mulai |
| **Assignee** | - |

## Description
Install semua library yang dibutuhkan untuk project ML/DL skin cancer detection.

## Acceptance Criteria
- [ ] TensorFlow/Keras terinstall dan bisa di-import
- [ ] OpenCV terinstall
- [ ] Scikit-learn terinstall
- [ ] Matplotlib, Seaborn, Plotly terinstall
- [ ] Pandas, NumPy terinstall
- [ ] Streamlit terinstall
- [ ] Semua library bisa di-import tanpa error

## Implementation Steps
1. Pastikan virtual environment aktif
2. Install library utama:
   ```bash
   pip install tensorflow==2.15.0
   pip install opencv-python==4.9.0.80
   pip install scikit-learn==1.4.0
   pip install matplotlib==3.8.2
   pip install seaborn==0.13.2
   pip install plotly==5.18.0
   pip install pandas==2.2.0
   pip install numpy==1.26.3
   pip install Pillow==10.2.0
   pip install streamlit==1.30.0
   pip install albumentations==1.3.1
   pip install tensorboard==2.15.1
   pip install keras-cv==0.8.2
   pip install tensorflow-hub==0.16.1
   ```
3. Verifikasi instalasi:
   ```python
   import tensorflow as tf
   import cv2
   import sklearn
   print(f"TensorFlow: {tf.__version__}")
   print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
   ```

## Library Breakdown

| Library | Fungsi | Kategori |
|---------|--------|----------|
| `tensorflow` | Deep Learning framework | Core |
| `opencv-python` | Image processing | Core |
| `scikit-learn` | ML utilities, metrics | Core |
| `matplotlib` | Visualization | Viz |
| `seaborn` | Statistical visualization | Viz |
| `plotly` | Interactive charts | Viz |
| `pandas` | Data manipulation | Data |
| `numpy` | Numerical computing | Data |
| `Pillow` | Image handling | Data |
| `streamlit` | Web application | Deployment |
| `albumentations` | Advanced augmentation | Training |
| `tensorboard` | Training monitoring | Training |
| `keras-cv` | Computer Vision extensions (Swin Transformer) | Training |
| `tensorflow-hub` | Pre-trained models repository | Training |

## Estimated Time
~30 menit (tergantung kecepatan internet)

## Dependencies
- S0-T01: Virtual environment sudah setup

## Potential Issues
- TensorFlow bisa besar (~500MB+), pastikan disk space cukup
- Jika ada GPU NVIDIA, install `tensorflow[and-cuda]` untuk GPU support
- Jika error kompatibilitas, coba turunkan versi Python ke 3.10
