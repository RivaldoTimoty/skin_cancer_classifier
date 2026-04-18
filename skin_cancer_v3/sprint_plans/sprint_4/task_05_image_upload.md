# Task 05 — Image Upload Component

## Task Info

| Item | Detail |
|------|--------|
| **Sprint** | 4 |
| **Task ID** | S4-T05 |
| **Priority** | High |
| **Story Points** | 3 |
| **Status** | Belum Mulai |

## Description
Komponen upload gambar premium dengan drag & drop, file browser, preview, dan validasi — menggunakan react-dropzone atau custom implementation.

## Acceptance Criteria
- [ ] Drag & drop zone dengan visual feedback (border animation)
- [ ] File browser fallback (click to browse)
- [ ] Image preview setelah upload
- [ ] File validation (format: JPG/PNG, size: max 10MB)
- [ ] Upload progress indicator
- [ ] Remove/replace image button
- [ ] "Analyze" CTA button setelah upload
- [ ] Glassmorphism card styling

## States

```
State 1: EMPTY
┌─────────────────────────────────┐
│                                 │
│    Drag & Drop your image    │
│    or click to browse           │
│                                 │
│    Supports: JPG, PNG (≤10MB)   │
│                                 │
│    [Browse Files]            │
└─────────────────────────────────┘

State 2: DRAGGING (dashed border animation)
┌─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┐
│                                 │
│    Drop your image here!     │
│                                 │
└─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┘

State 3: PREVIEW
┌─────────────────────────────────┐
│  ┌───────────┐  filename.jpg    │
│  │  [Image   │  1.2 MB          │
│  │  Preview] │  1024 × 768      │
│  └───────────┘  [Remove]      │
│                                 │
│  [Analyze Image →]           │
└─────────────────────────────────┘
```

## Estimated Time
~3 jam

## Dependencies
- S4-T03 (design system)
