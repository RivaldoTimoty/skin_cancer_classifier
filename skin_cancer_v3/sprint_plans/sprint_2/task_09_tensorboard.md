# Task 09 — Setup TensorBoard Logging

## Task Info

| Item | Detail |
|------|--------|
| **Sprint** | 2 |
| **Task ID** | S2-T09 |
| **Priority** | Medium |
| **Story Points** | 1 |
| **Status** | Belum Mulai |
| **Assignee** | - |

## Description
Setup TensorBoard untuk monitoring training secara real-time, termasuk loss curves, accuracy, dan learning rate.

## Acceptance Criteria
- [ ] TensorBoard callback terintegrasi di training
- [ ] Log directory terorganisir per experiment
- [ ] TensorBoard bisa diakses via browser
- [ ] Scalar metrics (loss, accuracy) terlog

## Implementation

```python
from tensorflow.keras.callbacks import TensorBoard
from datetime import datetime
import os

def get_tensorboard_callback(experiment_name):
    log_dir = os.path.join(
        'logs', 'tensorboard', 
        experiment_name,
        datetime.now().strftime('%Y%m%d-%H%M%S')
    )
    return TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=False,
        update_freq='epoch'
    )
```

### Launch TensorBoard
```bash
tensorboard --logdir logs/tensorboard --port 6006
```
Lalu buka `http://localhost:6006` di browser.

## Estimated Time
~15 menit

## Dependencies
- S2-T05 (callback list)
- TensorBoard terinstall
