# Low-Resolution CCTV Face Enhancement
**Sentio Mind · POC Assignment · Project 4**

GitHub: https://github.com/Sentiodirector/sentio-poc-face-enhancement.git
Branch: FirstName_LastName_RollNumber

---

## Why This Exists

School CCTV cameras are mounted high and use cheap lenses. When Sentio Mind crops a face from that footage it is often 12 to 80 pixels wide. At that size DeepFace emotion analysis gives garbage, face_recognition matching fails most of the time, and the profile photos shown to staff look like blurry blobs. No deep learning super-resolution models allowed — this has to run on CPU in under 30 seconds for 100 faces.

---

## What You Receive

```
p4_face_enhancement/
├── raw_faces/
│   ├── face_001.jpg        ← tiny CCTV face crops, typically 12–80px wide
│   └── ...                 ← download from dataset link
├── reference_identities/
│   ├── person_A.jpg        ← clear high-res photos for evaluation only
│   └── ...
├── face_enhancement.py     ← your template — copy to solution.py
├── face_enhancement.json   ← schema for evaluation_metrics.json
└── README.md
```

---

## What You Must Build

Run `python solution.py` → it must produce:

1. `enhanced_faces/` — all processed images at exactly 240×240 JPEG
2. `enhancement_report.html` — side-by-side A/B grid: original vs enhanced
3. `evaluation_metrics.json` — follows `face_enhancement.json` schema exactly

### The 4-Stage Pipeline (run in this exact order)

**Stage 1 — Denoise**
```python
cv2.fastNlMeansDenoisingColored(img, h=8, hColor=8, templateWindowSize=7, searchWindowSize=21)
```

**Stage 2 — CLAHE**
Convert to LAB. Apply CLAHE (clipLimit=3.5, tileGridSize=(4,4)) to L channel only. Convert back to BGR.

**Stage 3 — Multi-step upscale**
If short side < 64px: upscale 2× LANCZOS4 → unsharp mask (sigma=1.0, strength=1.6) → upscale 2× LANCZOS4 → resize to 240×240.
Else: direct resize to 240×240 LANCZOS4.

**Stage 4 — Zone sharpening**
Use MediaPipe Face Mesh to locate eye + nose region. Apply unsharp(sigma=0.8, strength=2.0) to that region. Apply unsharp(sigma=1.2, strength=1.3) to the rest. Fallback if no face found: apply unsharp(sigma=1.0, strength=1.5) uniformly.

### Metrics to Report

- Face recognition match accuracy before enhancement (%)
- Face recognition match accuracy after enhancement (%)
- Average Laplacian variance before and after (sharpness)
- Average SSIM improvement (scikit-image)

---

## Hard Rules

- No deep learning models (ESRGAN, GFPGAN, etc.)
- Output must be exactly 240×240 pixels
- 100 faces must process in under 30 seconds on CPU
- Do not rename functions in `face_enhancement.py`
- Do not change key names in `face_enhancement.json`
- Python 3.9+, no Jupyter notebooks

## Libraries

```
opencv-python==4.9.0   face_recognition==1.3.0   mediapipe==0.10.14
numpy==1.26.4          Pillow==10.3.0             scikit-image==0.22.0
```

---

## Submit

| # | File | What |
|---|------|------|
| 1 | `solution.py` | Working script |
| 2 | `enhanced_faces/` | Folder with all 240×240 crops |
| 3 | `enhancement_report.html` | A/B comparison grid |
| 4 | `evaluation_metrics.json` | Metrics matching schema |
| 5 | `demo.mp4` | Screen recording under 2 min |

Push to your branch only. Do not touch main.

---

## Bonus

Skip enhancement if Laplacian variance of the input is already above 80 — just resize. This saves time on inputs that are already sharp enough.

*Sentio Mind · 2026*
