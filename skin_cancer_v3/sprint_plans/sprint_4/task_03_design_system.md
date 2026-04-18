# Task 03 — Design System & Theme

## Task Info

| Item | Detail |
|------|--------|
| **Sprint** | 4 |
| **Task ID** | S4-T03 |
| **Priority** | High |
| **Story Points** | 3 |
| **Status** | Belum Mulai |

## Description
Membangun design system yang konsisten: color palette, typography, spacing, component tokens, dark/light mode, dan animasi dasar dengan Framer Motion.

## Acceptance Criteria
- [ ] Tailwind config extended dengan custom colors, fonts, animations
- [ ] Dark/Light mode toggle (next-themes)
- [ ] Global CSS variables defined
- [ ] Glassmorphism & gradient utility classes
- [ ] Base animation variants (Framer Motion)
- [ ] shadcn/ui theme customized

## Color Palette

```
Dark Mode (Primary):
  --background:    #0A0F1C (Deep Navy)
  --card:          #111827 (Dark Surface)
  --card-glass:    rgba(17, 24, 39, 0.7) + backdrop-blur
  --primary:       #3B82F6 (Royal Blue)
  --secondary:     #14B8A6 (Teal)
  --accent:        #8B5CF6 (Purple)
  --danger:        #EF4444 (Red — for high-risk alerts)
  --warning:       #F59E0B (Amber)
  --success:       #10B981 (Green)
  --text-primary:  #F9FAFB
  --text-secondary:#9CA3AF

Light Mode:
  --background:    #F8FAFC
  --card:          #FFFFFF
  --text-primary:  #0F172A
  --text-secondary:#64748B

Gradients:
  --gradient-hero:   linear-gradient(135deg, #3B82F6, #14B8A6)
  --gradient-card:   linear-gradient(135deg, #1E293B, #0F172A)
  --gradient-danger:  linear-gradient(135deg, #EF4444, #DC2626)
```

## 🔤 Typography

```css
/* Google Fonts: Plus Jakarta Sans + Inter */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

h1-h3: Plus Jakarta Sans (800, 700)
body, p: Inter (400, 500)
code, mono: JetBrains Mono
```

## Tailwind Config Extension

```typescript
// tailwind.config.ts
const config = {
  theme: {
    extend: {
      colors: {
        brand: {
          blue: '#3B82F6',
          teal: '#14B8A6',
          purple: '#8B5CF6',
          navy: '#0A0F1C',
        }
      },
      fontFamily: {
        heading: ['Plus Jakarta Sans', 'sans-serif'],
        body: ['Inter', 'sans-serif'],
      },
      animation: {
        'float': 'float 6s ease-in-out infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
        'slide-up': 'slideUp 0.5s ease-out',
      },
      backdropBlur: {
        xs: '2px',
      }
    }
  }
}
```

## Estimated Time
~2 jam

## Dependencies
- S4-T01 (Next.js + Tailwind setup)
