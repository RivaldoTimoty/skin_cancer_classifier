# Task 08 — Disease Information Pages

## Task Info

| Item | Detail |
|------|--------|
| **Sprint** | 4 |
| **Task ID** | S4-T08 |
| **Priority** | Medium |
| **Story Points** | 4 |
| **Status** | Belum Mulai |

## Description
Halaman edukatif untuk setiap 9 jenis penyakit kulit: deskripsi, gejala, faktor risiko, statistik, dan rekomendasi. Menggunakan dynamic routes (`/diseases/[slug]`).

## Acceptance Criteria
- [ ] `/diseases` — Grid overview semua 9 penyakit
- [ ] `/diseases/melanoma` etc — Detail page per penyakit
- [ ] Risk level badge (Critical/High/Medium/Low)
- [ ] Sample images dari dataset
- [ ] ABCDE rule explanation (untuk melanoma)
- [ ] "When to see a doctor" section
- [ ] References (medical sources)

## Disease Data Structure

```typescript
interface Disease {
  slug: string;
  name: string;
  riskLevel: 'critical' | 'high' | 'medium' | 'low';
  description: string;
  symptoms: string[];
  riskFactors: string[];
  prevalence: string;
  recommendation: string;
  whenToSeeDoctor: string;
  abcdeRule?: string;  // melanoma only
  references: string[];
}
```

## Disease Card Design

```
┌──────────────────────────┐
│  CRITICAL             │
│                          │
│  Melanoma                │
│                          │
│  Most dangerous form of  │
│  skin cancer. Can spread │
│  to other organs.        │
│                          │
│  [Learn More →]          │
└──────────────────────────┘
```

## Estimated Time
~4 jam

## Dependencies
- S4-T03 (design system)
