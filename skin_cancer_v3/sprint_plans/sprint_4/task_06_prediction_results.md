# Task 06 — Prediction Results Page

## Task Info

| Item | Detail |
|------|--------|
| **Sprint** | 4 |
| **Task ID** | S4-T06 |
| **Priority** | High |
| **Story Points** | 5 |
| **Status** | Belum Mulai |

## Description
Halaman hasil prediksi yang menampilkan: original image vs Grad-CAM, top predictions dengan confidence chart, risk alert, dan educational info — semuanya dengan animasi yang smooth.

## Acceptance Criteria
- [ ] Split layout: Image (left) | Results (right)
- [ ] Animated confidence bars (Framer Motion)
- [ ] Interactive donut/bar chart (Recharts)
- [ ] Risk level alert banner (color-coded)
- [ ] Disease detail expandable section
- [ ] "Upload Another" dan "Learn More" CTAs
- [ ] Share results button
- [ ] Medical disclaimer

## Layout

```
┌──────────────────────────────────────────┐
│  ← Back to Detection                    │
├──────────────────┬───────────────────────┤
│                  │  PREDICTION RESULTS   │
│  [Original]      │                       │
│  [Image]         │  HIGH RISK         │
│                  │  Melanoma             │
│  [Grad-CAM]      │  Confidence: 87.3%    │
│  [Heatmap]       │                       │
│                  │  ▓▓▓▓▓▓▓▓▓░ 87.3%    │
│  Toggle:         │  ▓▓░░░░░░░░  8.1%    │
│  ○ Original      │  ▓░░░░░░░░░  2.3%    │
│  ● Grad-CAM      │                       │
│  ○ Overlay       │  [Chart]           │
│                  │                       │
├──────────────────┴───────────────────────┤
│                                          │
│  ABOUT MELANOMA                          │
│  Description, symptoms, risk factors     │
│  [Learn More →]                          │
├──────────────────────────────────────────┤
│                                          │
│  [↻ Upload Another]  [📖 All Diseases]   │
├──────────────────────────────────────────┤
│  Disclaimer: AI tool, not diagnosis   │
└──────────────────────────────────────────┘
```

## Key Components

```tsx
// Animated confidence bar
<motion.div 
  initial={{ width: 0 }} 
  animate={{ width: `${confidence * 100}%` }}
  transition={{ duration: 1, ease: "easeOut" }}
  className="h-3 rounded-full bg-gradient-to-r from-brand-blue to-brand-teal"
/>

// Risk Level Banner
const RiskBanner = ({ level }) => {
  const styles = {
    critical: "bg-red-500/10 border-red-500 text-red-400",
    high: "bg-orange-500/10 border-orange-500 text-orange-400",
    low: "bg-green-500/10 border-green-500 text-green-400",
  }
  return (
    <div className={`border rounded-xl p-4 ${styles[level]}`}>
      {level === 'critical' && "HIGH RISK — Consult a dermatologist"}
    </div>
  )
}
```

## Estimated Time
~5 jam

## Dependencies
- S4-T02 (API response available)
- S4-T05 (upload component — passes image to this page)
