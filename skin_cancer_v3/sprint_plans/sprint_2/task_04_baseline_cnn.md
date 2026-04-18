# Task 04 — Build Baseline CNN Model

## Task Info

| Item | Detail |
|------|--------|
| **Sprint** | 2 |
| **Task ID** | S2-T04 |
| **Priority** | High |
| **Story Points** | 3 |
| **Status** | Belum Mulai |
| **Assignee** | - |

## Description
Membangun model CNN sederhana (custom architecture) sebagai baseline untuk dibandingkan dengan transfer learning models nanti.

## Acceptance Criteria
- [ ] Custom CNN model dibangun dengan Keras Sequential/Functional API
- [ ] Model summary didokumentasikan (layers, params)
- [ ] Model bisa compile dan fit tanpa error
- [ ] Module `src/models.py` dibuat
- [ ] Total parameters < 5M (tetap ringan)

## Architecture Design

```
Input (224, 224, 3)
    ↓
Conv2D(32, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
    ↓
Conv2D(64, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
    ↓
Conv2D(128, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
    ↓
Conv2D(256, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
    ↓
GlobalAveragePooling2D
    ↓
Dense(256) → Dropout(0.5) → ReLU
    ↓
Dense(128) → Dropout(0.3) → ReLU
    ↓
Dense(9, softmax)     ← Output (9 classes)
```

## Code Skeleton

```python
# src/models.py
import tensorflow as tf
from tensorflow.keras import layers, models

def build_baseline_cnn(input_shape=(224, 224, 3), num_classes=9):
    """Build a simple CNN baseline model."""
    
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 2
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 3
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 4
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Classifier
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ], name='baseline_cnn')
    
    return model

# Compile
model = build_baseline_cnn()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()
```

## Expected Model Summary
- **Total params:** ~1-3M
- **Input:** (224, 224, 3)
- **Output:** (9,) softmax probabilities

## Estimated Time
~1.5 jam

## Dependencies
- S2-T01 (data pipeline — untuk input shape verification)
