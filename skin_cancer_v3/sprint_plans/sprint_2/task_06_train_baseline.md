# Task 06 — Train Baseline Model

## Task Info

| Item | Detail |
|------|--------|
| **Sprint** | 2 |
| **Task ID** | S2-T06 |
| **Priority** | High |
| **Story Points** | 2 |
| **Status** | Belum Mulai |
| **Assignee** | - |

## Description
Melatih baseline CNN model dengan data pipeline dan callbacks yang sudah disiapkan.

## Acceptance Criteria
- [ ] Model dilatih hingga konvergen (atau EarlyStopping trigger)
- [ ] Training history (loss, accuracy) disimpan
- [ ] Training curves divisualisasikan (loss & accuracy vs epoch)
- [ ] Best model checkpoint tersimpan di `models/`
- [ ] Training time dicatat

## Implementation

```python
import time

# Training parameters
EPOCHS = 50
BATCH_SIZE = 32

# Get data
train_ds = create_dataset('data/Train', augment=True)
val_ds = create_dataset('data/Validation', augment=False)

# Build model
model = build_baseline_cnn()
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Get callbacks
callbacks = get_callbacks('baseline_cnn')

# Train
start = time.time()
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weight_dict
)
training_time = time.time() - start
print(f"Training time: {training_time/60:.1f} minutes")

# Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history['loss'], label='Train Loss')
ax1.plot(history.history['val_loss'], label='Val Loss')
ax1.set_title('Loss Curves')
ax1.legend()

ax2.plot(history.history['accuracy'], label='Train Accuracy')
ax2.plot(history.history['val_accuracy'], label='Val Accuracy')
ax2.set_title('Accuracy Curves')
ax2.legend()

plt.savefig('reports/figures/baseline_training_curves.png', dpi=150)
```

## Expected Training Config

| Parameter | Value |
|-----------|-------|
| Epochs | 50 (max, EarlyStopping) |
| Batch Size | 32 |
| Optimizer | Adam (lr=1e-3) |
| Loss | Categorical Crossentropy |
| Class Weights | Balanced |

## Estimated Time
~2-4 jam (tergantung GPU/CPU)

## Dependencies
- S2-T01 sampai S2-T05 selesai

## Notes
- Training tanpa GPU bisa sangat lambat (~4+ jam)
- Monitor GPU memory jika menggunakan CUDA
- Jika memory error, kurangi batch size ke 16
