# Task 07 — Grad-CAM Heatmap Display

## Task Info

| Item | Detail |
|------|--------|
| **Sprint** | 4 |
| **Task ID** | S4-T07 |
| **Priority** | Medium |
| **Story Points** | 3 |
| **Status** | Belum Mulai |

## Description
Komponen interaktif untuk menampilkan dan toggle antara original image, Grad-CAM heatmap, dan overlay.

## Acceptance Criteria
- [ ] 3 view modes: Original, Heatmap, Overlay
- [ ] Smooth transition antar modes (crossfade)
- [ ] Opacity slider untuk overlay blend
- [ ] Explanation tooltip: "Red = high attention"
- [ ] Zoom capability (pinch/scroll)
- [ ] Responsive image container

## Component Sketch

```tsx
const GradCamViewer = ({ originalSrc, gradcamSrc }) => {
  const [mode, setMode] = useState<'original' | 'heatmap' | 'overlay'>('original')
  const [opacity, setOpacity] = useState(0.4)
  
  return (
    <div className="relative rounded-2xl overflow-hidden bg-black/20">
      {/* Image container with smooth transitions */}
      <AnimatePresence mode="wait">
        <motion.img
          key={mode}
          src={mode === 'original' ? originalSrc : gradcamSrc}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="w-full h-auto"
        />
      </AnimatePresence>
      
      {/* Mode toggle */}
      <div className="flex gap-2 mt-4">
        <Button variant={mode === 'original' ? 'default' : 'outline'}
                onClick={() => setMode('original')}>
          Original
        </Button>
        <Button variant={mode === 'heatmap' ? 'default' : 'outline'}
                onClick={() => setMode('heatmap')}>
          Grad-CAM
        </Button>
        <Button variant={mode === 'overlay' ? 'default' : 'outline'}
                onClick={() => setMode('overlay')}>
          Overlay
        </Button>
      </div>
      
      {/* Opacity slider for overlay */}
      {mode === 'overlay' && (
        <Slider value={opacity} onChange={setOpacity} min={0} max={1} step={0.1} />
      )}
      
      <p className="text-sm text-muted-foreground mt-2">
        Red areas = high model attention | 🔵 Blue = low attention
      </p>
    </div>
  )
}
```

## Estimated Time
~3 jam

## Dependencies
- S4-T02 (API returns gradcam_image as base64)
- S4-T06 (results page)
