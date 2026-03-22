# Sentio Mind Project 4: Low-Resolution CCTV Face Enhancement

A classical (non-deep-learning) image processing pipeline that enhances tiny, blurry CCTV face crops into clear 240x240 images suitable for face recognition and emotion analysis.

## Problem Statement

School CCTV cameras produce extremely small face crops (as low as 12x12 pixels) where deep learning models like DeepFace fail and face_recognition matching is unreliable. This project builds a **CPU-only classical CV pipeline** that:

- Enhances low-resolution face crops to **240x240** resolution
- Improves face recognition match accuracy against reference identities
- Increases sharpness measurably (Laplacian variance)
- Runs on CPU in under **30 seconds** for 100 faces

## Pipeline Architecture

The enhancement pipeline consists of 4 sequential stages executed in fixed order:

### Stage 1 — Adaptive Denoising
- Uses `cv2.fastNlMeansDenoisingColored` with strength adapted to face size
- Tiny faces (<40px): full denoising (h=8), medium faces (<80px): moderate (h=6), larger faces: light (h=4)

### Stage 2 — CLAHE with Adaptive Gamma
- Converts to LAB color space
- Applies adaptive gamma correction: brightens dark faces (mean L < 80, gamma=0.8), dims overexposed faces (mean L > 190, gamma=1.3)
- Applies CLAHE (clipLimit=3.5, tileGridSize=(4,4)) on the L channel only

### Stage 3 — Multi-Step Upscale
- **Small faces (<64px short side):** 2x LANCZOS4 upscale → bilateral filter (smooth blocky artifacts) → unsharp mask (sigma=1.0, strength=1.6) → 2x LANCZOS4 upscale → light bilateral filter → final resize to 240x240
- **Larger faces:** direct LANCZOS4 resize to 240x240 + light unsharp mask (sigma=0.8, strength=0.8)

### Stage 4 — Zone-Based Sharpening
- Uses MediaPipe Face Mesh to detect eye and nose landmarks
- Builds a feathered convex hull mask over the eye+nose region
- Applies stronger sharpening (sigma=0.8, strength=2.0) to the eye+nose zone
- Applies lighter sharpening (sigma=1.2, strength=1.3) to the rest of the face
- Blends using Gaussian-blurred mask for smooth transitions
- **Fallback:** two-pass uniform unsharp mask if no landmarks detected

### Bonus: Skip Optimization
Faces with Laplacian variance above 80 (already sharp at target size) skip the full pipeline and are simply resized — saving processing time.

## Auto Face Extraction from Video

If `raw_faces/` is empty, the pipeline automatically extracts faces from video files in `Video_1/`:
- Uses **Haar cascades** (frontal + profile) for face detection
- Samples every 15th frame with 15% padding around detected faces
- Extracts up to 100 face crops automatically

## Tech Stack

| Library | Version | Purpose |
|---|---|---|
| OpenCV | 4.9.0 | Core image processing (denoise, CLAHE, resize, sharpen, bilateral filter) |
| face_recognition | 1.3.0 | HOG face detection, 128-d face encoding for identity matching |
| MediaPipe | 0.10.14 | Face Mesh landmarks for zone-based sharpening |
| NumPy | 1.26.4 | Array operations |
| Pillow | 10.3.0 | Image I/O support |
| scikit-image | 0.22.0 | SSIM (Structural Similarity Index) computation |

## Project Structure

```
├── solution.py                              # Main pipeline source code
├── requirements.txt                         # Python dependencies
├── sentio-poc-face-enhancement-assignment.pdf  # Assignment specification
```

### Generated Output (after running)

```
├── enhanced_faces/             # 240x240 enhanced JPEG face images
├── enhancement_report.html     # Self-contained HTML A/B comparison report
└── evaluation_metrics.json     # Per-face + aggregate metrics (JSON)
```

## Installation & Usage

### Prerequisites
- Python 3.9+
- pip

### Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

### Prepare Input Data

**Option A — From video:**
Place CCTV video files (`.mp4`, `.avi`, `.mov`, etc.) in a `Video_1/` folder. Faces will be auto-extracted.

**Option B — From pre-extracted crops:**
Place face crop images (`.jpg`, `.jpeg`, `.png`) directly in `raw_faces/`.

**Reference identities:**
Place reference photos in one of these folders (auto-detected): `reference_identities/`, `profile_1/`, `Profiles_1/`, `profiles_1/`, `references/`, `ref_faces/`, or `ref/`.

### Run

```bash
python solution.py
```

### Output

| File | Description |
|---|---|
| `enhanced_faces/` | Enhanced 240x240 JPEG images |
| `enhancement_report.html` | Dark-themed visual before/after comparison (open in browser) |
| `evaluation_metrics.json` | Machine-readable metrics matching the integration schema |

## Evaluation Metrics

The pipeline evaluates enhancement quality using three metrics:

- **Recognition Accuracy** — percentage of faces matched against reference identities (before vs. after enhancement)
- **Sharpness** — Laplacian variance (higher = sharper)
- **SSIM** — Structural Similarity Index between original (resized) and enhanced face

## Configuration

All parameters are at the top of `solution.py`:

```python
VIDEO_DIR              = Path("Video_1")          # Input video directory
RAW_FACES_DIR          = Path("raw_faces")        # Extracted face crops
ENHANCED_DIR           = Path("enhanced_faces")   # Output directory
TARGET_SIZE            = (240, 240)               # Target output resolution
SHARPNESS_SKIP_THRESHOLD = 80.0                   # Skip enhancement if already sharp
```

## Key Design Decisions

1. **Bilateral filtering in upscale stage** — smooths blocky interpolation artifacts between upscale passes, producing cleaner final images
2. **Haar cascade face extraction** — uses both frontal and profile cascades for robust face detection from video frames
3. **Adaptive denoising by size** — stronger denoising on tiny noisy crops, lighter on larger crops to preserve existing detail
4. **Zone sharpening with feathered mask** — Gaussian-blurred convex hull mask ensures smooth transitions between strongly and lightly sharpened regions
5. **Smart face encoding** — for 240x240 crops, provides full image as face location to skip expensive HOG detection scan

## Author

**Rudra**
