# Task 02 — Analisis Distribusi Kelas

## Task Info

| Item | Detail |
|------|--------|
| **Sprint** | 1 |
| **Task ID** | S1-T02 |
| **Priority** | High |
| **Story Points** | 2 |
| **Status** | Belum Mulai |
| **Assignee** | - |

## Description
Menganalisis dan memvisualisasikan distribusi jumlah gambar per kelas untuk mengidentifikasi class imbalance.

## Acceptance Criteria
- [ ] Bar chart distribusi kelas (train & test) dibuat
- [ ] Pie chart proporsi kelas dibuat
- [ ] Rasio imbalance dihitung dan dicatat
- [ ] Statistik deskriptif per kelas (min, max, mean, median)
- [ ] Plot disimpan ke `reports/figures/`

## Implementation Steps
1. Hitung jumlah gambar per kelas (Train & Test)
2. Buat horizontal bar chart yang sorted
3. Buat pie chart dengan persentase
4. Hitung imbalance ratio (max/min)
5. Buat tabel ringkasan statistik

## Expected Output

### Data Distribusi (Train)
| Kelas | Jumlah | Proporsi |
|-------|--------|----------|
| pigmented benign keratosis | 462 | 20.6% |
| melanoma | 438 | 19.6% |
| basal cell carcinoma | 376 | 16.8% |
| nevus | 357 | 15.9% |
| squamous cell carcinoma | 181 | 8.1% |
| vascular lesion | 139 | 6.2% |
| actinic keratosis | 114 | 5.1% |
| dermatofibroma | 95 | 4.2% |
| seborrheic keratosis | 77 | 3.4% |

### Key Metrics
- **Imbalance Ratio:** 6:1 (462 vs 77)
- **Total Train:** 2,239 images
- **Mean per class:** ~249 images
- **Std per class:** ~145 images

## Code Skeleton
```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Count images per class
train_dir = 'data/Train'
class_counts = {}
for cls in os.listdir(train_dir):
    cls_path = os.path.join(train_dir, cls)
    class_counts[cls] = len(os.listdir(cls_path))

df = pd.DataFrame(list(class_counts.items()), columns=['Class', 'Count'])
df = df.sort_values('Count', ascending=True)

# Bar Chart
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=df, x='Count', y='Class', palette='viridis')
plt.title('Distribution of Skin Lesion Classes')
plt.tight_layout()
plt.savefig('reports/figures/class_distribution.png', dpi=150)
```

## Estimated Time
~30 menit

## Dependencies
- Sprint 0 selesai
