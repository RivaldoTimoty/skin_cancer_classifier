# Task 04 — Landing Page / Hero Section

## Task Info

| Item | Detail |
|------|--------|
| **Sprint** | 4 |
| **Task ID** | S4-T04 |
| **Priority** | High |
| **Story Points** | 4 |
| **Status** | Belum Mulai |

## Description
Membangun landing page yang stunning dan profesional. Harus memberikan kesan pertama yang premium, medical-grade, dan trustworthy.

## Acceptance Criteria
- [ ] Hero section dengan gradient background + animated particles
- [ ] Headline + subheadline yang compelling
- [ ] CTA "Start Detection" button dengan glow animation
- [ ] Stats counter section (accuracy, classes, images analyzed)
- [ ] Feature cards (AI Detection, Grad-CAM, 9 Classes)
- [ ] How It Works section (3-step process)
- [ ] Disease cards grid (9 skin lesion types overview)
- [ ] Testimonial/trust section
- [ ] Footer dengan links

## Layout Wireframe

```
┌──────────────────────────────────────────┐
│  Navbar (Logo | Detect | Diseases | Dark)│
├──────────────────────────────────────────┤
│                                          │
│  AI-Powered                           │
│  SKIN CANCER DETECTION                   │
│  Detect 9 types of skin lesions with     │
│  state-of-the-art deep learning          │
│                                          │
│  [Start Detection →]                  │
│                                          │
│  ✓ 87% Accuracy  ✓ 9 Classes  ✓ Grad-CAM│
├──────────────────────────────────────────┤
│                                          │
│  [AI Detection] [Explainable] [Fast]     │
│  Feature Cards with icons + descriptions │
├──────────────────────────────────────────┤
│                                          │
│  HOW IT WORKS                            │
│  ① Upload → ② Analyze → ③ Results       │
│  Step cards with arrows                  │
├──────────────────────────────────────────┤
│                                          │
│  DETECTABLE DISEASES                     │
│  Grid of 9 disease cards with risk level │
├──────────────────────────────────────────┤
│  Footer (Links | Disclaimer | Credits)   │
└──────────────────────────────────────────┘
```

## Key Components

```tsx
// Hero Section dengan Framer Motion
<motion.div
  initial={{ opacity: 0, y: 20 }}
  animate={{ opacity: 1, y: 0 }}
  transition={{ duration: 0.8 }}
>
  <h1 className="text-6xl font-heading font-extrabold 
    bg-gradient-to-r from-brand-blue to-brand-teal 
    bg-clip-text text-transparent">
    AI-Powered Skin Cancer Detection
  </h1>
</motion.div>

// Animated stats counter
<motion.div whileInView={{ opacity: 1 }} viewport={{ once: true }}>
  <CountUp end={87} suffix="%" /> Accuracy
  <CountUp end={9} /> Disease Types
  <CountUp end={2239} /> Training Images
</motion.div>
```

## Estimated Time
~4 jam

## Dependencies
- S4-T03 (design system defined)
