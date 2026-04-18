# Task 10 — Responsive Design & Animations

## Task Info

| Item | Detail |
|------|--------|
| **Sprint** | 4 |
| **Task ID** | S4-T10 |
| **Priority** | High |
| **Story Points** | 3 |
| **Status** | Belum Mulai |

## Description
Polish responsive design untuk semua breakpoints dan implementasi micro-animations yang membuat app terasa premium dan alive.

## Acceptance Criteria
- [ ] Mobile (< 640px): Single column, touch-friendly
- [ ] Tablet (640-1024px): Adjusted grid, sidebar collapse
- [ ] Desktop (> 1024px): Full layout, split panes
- [ ] Page transition animations (Framer Motion)
- [ ] Scroll-triggered animations (stagger children)
- [ ] Hover effects pada cards dan buttons
- [ ] Loading skeleton animations
- [ ] Smooth image transition (upload → result)

## Animation Variants

```tsx
// Framer Motion variants
export const fadeInUp = {
  initial: { opacity: 0, y: 30 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.6, ease: "easeOut" }
}

export const staggerContainer = {
  animate: { transition: { staggerChildren: 0.1 } }
}

export const scaleOnHover = {
  whileHover: { scale: 1.03 },
  whileTap: { scale: 0.98 },
  transition: { type: "spring", stiffness: 300 }
}

// Page transitions
export const pageTransition = {
  initial: { opacity: 0, x: -20 },
  animate: { opacity: 1, x: 0 },
  exit: { opacity: 0, x: 20 },
  transition: { duration: 0.3 }
}
```

## Estimated Time
~3 jam

## Dependencies
- S4-T04 sampai S4-T09 (all pages built)
