# Task 09 — Model Comparison Dashboard

## Task Info

| Item | Detail |
|------|--------|
| **Sprint** | 4 |
| **Task ID** | S4-T09 |
| **Priority** | Medium |
| **Story Points** | 3 |
| **Status** | Belum Mulai |

## Description
Dashboard interaktif yang menampilkan perbandingan performa semua 8 model (baseline + 7 transfer learning). Menampilkan metrics, charts, dan Grad-CAM comparisons.

## Acceptance Criteria
- [ ] Tabel comparison interaktif (sortable)
- [ ] Bar chart: accuracy per model
- [ ] Radar chart: multi-metric per model
- [ ] Confusion matrix viewer (selectable per model)
- [ ] Grad-CAM comparison grid (same image, all models)
- [ ] "Active Model" indicator

## Layout

```
┌──────────────────────────────────────────┐
│  MODEL COMPARISON DASHBOARD              │
├──────────────────────────────────────────┤
│                                          │
│  [Bar Chart: Accuracy by Model]          │
│                                          │
├──────────────────────────────────────────┤
│                                          │
│  Model | Acc | F1 | Melanoma | Size | ⚡ │
│  ──────┼─────┼────┼──────────┼──────┼───│
│  EffNet| 87% |0.85| 91%      | 48MB | ★ │
│  Dense | 85% |0.83| 89%      | 33MB |   │
│  Xcpt  | 84% |0.82| 87%      | 88MB |   │
│  ...   |     |    |          |      |   │
│                                          │
├──────────────────────────────────────────┤
│  [Radar Chart: Selected Model]           │
│                                          │
│  Select Model: [EfficientNetB3 ▼]       │
└──────────────────────────────────────────┘
```

## Key Components

```tsx
// Recharts radar chart
<RadarChart data={modelMetrics}>
  <PolarGrid />
  <PolarAngleAxis dataKey="metric" />
  <Radar name="EfficientNetB3" dataKey="efficientnet" fill="#3B82F6" />
  <Radar name="DenseNet121" dataKey="densenet" fill="#14B8A6" />
</RadarChart>
```

## Estimated Time
~3 jam

## Dependencies
- S3-T16 (model comparison data)
- S4-T03 (design system)
