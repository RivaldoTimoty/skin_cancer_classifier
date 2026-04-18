# Task 05 — Setup Training Callbacks

## Task Info

| Item | Detail |
|------|--------|
| **Sprint** | 2 |
| **Task ID** | S2-T05 |
| **Priority** | High |
| **Story Points** | 2 |
| **Status** | Belum Mulai |
| **Assignee** | - |

## Description
Menyiapkan Keras callbacks untuk monitoring, early stopping, dan checkpointing selama training.

## Acceptance Criteria
- [ ] EarlyStopping callback dikonfigurasi
- [ ] ReduceLROnPlateau callback dikonfigurasi
- [ ] ModelCheckpoint callback dikonfigurasi
- [ ] TensorBoard callback dikonfigurasi
- [ ] Semua callback terintegrasi dalam list

## Implementation

```python
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
)
import os
from datetime import datetime

def get_callbacks(model_name='baseline'):
    """Get training callbacks."""
    
    log_dir = os.path.join('logs', model_name, 
                           datetime.now().strftime('%Y%m%d-%H%M%S'))
    
    callbacks = [
        # Stop training when val_loss stops improving
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate when val_loss plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # Save best model checkpoint
        ModelCheckpoint(
            filepath=f'models/{model_name}_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        
        # TensorBoard logging
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        )
    ]
    
    return callbacks
```

## Callback Summary

| Callback | Monitor | Action |
|----------|---------|--------|
| EarlyStopping | val_loss | Stop after 10 epochs tanpa improvement |
| ReduceLROnPlateau | val_loss | Kurangi LR ×0.5 setelah 5 epochs plateau |
| ModelCheckpoint | val_accuracy | Simpan model dengan accuracy tertinggi |
| TensorBoard | all | Log metrics untuk visualisasi |

## Estimated Time
~30 menit

## Dependencies
- S2-T04 (model sudah ada)
