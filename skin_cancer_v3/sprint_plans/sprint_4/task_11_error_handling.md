# Task 11 — Error Handling & Loading States

## Task Info

| Item | Detail |
|------|--------|
| **Sprint** | 4 |
| **Task ID** | S4-T11 |
| **Priority** | High |
| **Story Points** | 2 |
| **Status** | Belum Mulai |

## Description
Implementasi error boundaries, loading states, toast notifications, dan graceful degradation.

## Acceptance Criteria
- [ ] Loading spinner/skeleton saat API call
- [ ] AI "analyzing" animation (pulse, scan effect)
- [ ] Toast notifications (success, error, warning)
- [ ] File validation errors (format, size)
- [ ] API connection error handling
- [ ] 404 page styled
- [ ] Network timeout handling
- [ ] Low confidence warning

## Loading States

```tsx
// Analyzing animation
const AnalyzingOverlay = () => (
  <motion.div className="flex flex-col items-center gap-4">
    <motion.div
      animate={{ rotate: 360 }}
      transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
    >
      <Brain className="w-12 h-12 text-brand-blue" />
    </motion.div>
    <motion.p
      animate={{ opacity: [1, 0.5, 1] }}
      transition={{ duration: 1.5, repeat: Infinity }}
    >
      Analyzing skin lesion...
    </motion.p>
    <div className="w-64">
      <motion.div
        className="h-1 bg-gradient-to-r from-brand-blue to-brand-teal rounded-full"
        initial={{ width: "0%" }}
        animate={{ width: "100%" }}
        transition={{ duration: 3 }}
      />
    </div>
  </motion.div>
)
```

## Estimated Time
~2 jam

## Dependencies
- S4-T05, S4-T06 (upload + results pages)
