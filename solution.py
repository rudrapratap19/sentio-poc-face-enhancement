"""
face_enhancement.py  ->  solution.py
Sentio Mind · Project 4 · Low-Resolution CCTV Face Enhancement

Copy this file to solution.py and fill in every TODO block.
Do not rename any function.
Run: python solution.py
Output goes into enhanced_faces/ (created automatically).
"""

import cv2
import json
import base64
import time
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
VIDEO_DIR        = Path("Video_1")            # folder containing CCTV video(s)
RAW_FACES_DIR    = Path("raw_faces")
ENHANCED_DIR     = Path("enhanced_faces")
REPORT_HTML_OUT  = Path("enhancement_report.html")
METRICS_JSON_OUT = Path("evaluation_metrics.json")

TARGET_SIZE      = (240, 240)
RAW_FACES_DIR.mkdir(exist_ok=True)
ENHANCED_DIR.mkdir(exist_ok=True)

# Bonus: skip enhancement threshold
SHARPNESS_SKIP_THRESHOLD = 80.0

def _find_reference_dir() -> Path:
    """Auto-detect reference identity folder name."""
    for name in ["reference_identities", "profile_1", "Profiles_1",
                 "profiles_1", "Profile_1", "references"]:
        p = Path(name)
        if p.exists() and p.is_dir():
            return p
    return Path("reference_identities")


REFERENCE_DIR = Path("../reference_identities") if Path("../reference_identities").exists() else _find_reference_dir()

# Lazy-loaded MediaPipe FaceMesh (reused across calls for performance)
_face_mesh_instance = None


def _get_face_mesh():
    """Lazy-load a single MediaPipe FaceMesh instance for reuse.
    Returns None if mediapipe solutions API is unavailable."""
    global _face_mesh_instance
    if _face_mesh_instance is None:
        try:
            import mediapipe as mp
            _face_mesh_instance = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.3,
            )
        except (AttributeError, Exception) as e:
            print(f"  WARNING: MediaPipe FaceMesh unavailable ({e})")
            print(f"  Stage 4 will use uniform sharpening fallback.")
            _face_mesh_instance = "unavailable"
    return None if _face_mesh_instance == "unavailable" else _face_mesh_instance


# MediaPipe Face Mesh landmark indices for eye + nose region
_LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
_RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
_NOSE = [1, 2, 3, 4, 5, 6, 168, 195, 197, 98, 327, 294, 64, 240, 460, 278, 48]
_EYE_NOSE_INDICES = _LEFT_EYE + _RIGHT_EYE + _NOSE


# ---------------------------------------------------------------------------
# VIDEO FACE EXTRACTION — populate raw_faces/ from Video_1/ folder if needed
# ---------------------------------------------------------------------------

def extract_faces_from_video(video_path: Path, output_dir: Path,
                              every_n: int = 15, padding: float = 0.15,
                              max_faces: int = 100, start_count: int = 0) -> int:
    """
    Detect and crop faces from a video using Haar cascades (frontal + profile).
    Saves JPEG crops into output_dir. Returns total face count.
    Padding reduced to 0.15 to produce tighter crops matching CCTV face sizes.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {video_path}")
        return start_count

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Video: {video_path.name} | {total_frames} frames | {fps:.1f} FPS")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    profile_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_profileface.xml"
    )

    count = start_count
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or count >= max_faces:
            break

        if frame_idx % every_n != 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(12, 12)
        )
        if len(faces) == 0:
            faces = profile_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(12, 12)
            )

        fh, fw = frame.shape[:2]
        for (x, y, w, h) in faces:
            if count >= max_faces:
                break
            pad_x = int(w * padding)
            pad_y = int(h * padding)
            x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
            x2, y2 = min(fw, x + w + pad_x), min(fh, y + h + pad_y)

            crop = frame[y1:y2, x1:x2]
            if crop.shape[0] < 12 or crop.shape[1] < 12:
                continue

            count += 1
            cv2.imwrite(str(output_dir / f"face_{count:04d}.jpg"), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            if count % 10 == 0:
                print(f"    Extracted {count} faces so far...")

        frame_idx += 1

    cap.release()
    return count


# ---------------------------------------------------------------------------
# STAGE 1 — DENOISE
# ---------------------------------------------------------------------------

def stage1_denoise(img: np.ndarray) -> np.ndarray:
    """
    cv2.fastNlMeansDenoisingColored with h=8, hColor=8, templateWindowSize=7, searchWindowSize=21
    Adapts denoising strength based on image size — tiny noisy crops get full h=8,
    larger crops get lighter denoising to preserve detail.
    """
    h_img, w_img = img.shape[:2]
    short_side = min(h_img, w_img)

    if short_side < 40:
        h_val, hc_val = 8, 8
    elif short_side < 80:
        h_val, hc_val = 6, 6
    else:
        h_val, hc_val = 4, 4

    return cv2.fastNlMeansDenoisingColored(img, None, h=h_val, hColor=hc_val,
                                           templateWindowSize=7, searchWindowSize=21)


# ---------------------------------------------------------------------------
# STAGE 2 — CLAHE
# ---------------------------------------------------------------------------

def stage2_clahe(img: np.ndarray) -> np.ndarray:
    """
    Convert to LAB. Apply adaptive gamma correction based on mean brightness,
    then CLAHE (clipLimit=3.5, tileGridSize=(4,4)) to L channel. Merge + convert back.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    # Adaptive gamma: brighten dark faces, dim over-exposed ones
    mean_l = np.mean(l_ch)
    if mean_l < 80:
        gamma = 0.8
        lut = np.array([np.clip(((i / 255.0) ** gamma) * 255, 0, 255)
                         for i in range(256)], dtype=np.uint8)
        l_ch = cv2.LUT(l_ch, lut)
    elif mean_l > 190:
        gamma = 1.3
        lut = np.array([np.clip(((i / 255.0) ** gamma) * 255, 0, 255)
                         for i in range(256)], dtype=np.uint8)
        l_ch = cv2.LUT(l_ch, lut)

    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(4, 4))
    l_ch = clahe.apply(l_ch)
    lab = cv2.merge([l_ch, a_ch, b_ch])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# STAGE 3 — MULTI-STEP UPSCALE
# ---------------------------------------------------------------------------

def unsharp_mask(img: np.ndarray, sigma: float, strength: float) -> np.ndarray:
    """
    blurred = GaussianBlur(img, sigma)
    result  = img + strength * (img - blurred)
    Clip to 0-255.
    """
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    result = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
    return np.clip(result, 0, 255).astype(np.uint8)


def stage3_upscale(img: np.ndarray) -> np.ndarray:
    """
    If short side < 64px: 2x LANCZOS4 -> unsharp(1.0, 1.6) -> 2x LANCZOS4 -> resize to TARGET_SIZE.
    Otherwise: direct resize to TARGET_SIZE LANCZOS4 + light sharpening.
    """
    h, w = img.shape[:2]
    short_side = min(h, w)

    if short_side < 64:
        # First 2x upscale
        img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
        # Bilateral to smooth blocky artifacts before sharpening
        img = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)
        # Intermediate unsharp mask
        img = unsharp_mask(img, 1.0, 1.6)
        # Second 2x upscale
        h2, w2 = img.shape[:2]
        img = cv2.resize(img, (w2 * 2, h2 * 2), interpolation=cv2.INTER_LANCZOS4)
        # Light smoothing after final upscale
        img = cv2.bilateralFilter(img, d=3, sigmaColor=30, sigmaSpace=30)
    else:
        # For already-larger faces: resize then sharpen to recover detail lost in resize
        img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
        img = unsharp_mask(img, 0.8, 0.8)
        return img

    # Final resize to target
    return cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)


# ---------------------------------------------------------------------------
# STAGE 4 — ZONE SHARPENING
# ---------------------------------------------------------------------------

def stage4_zone_sharpen(img: np.ndarray) -> np.ndarray:
    """
    MediaPipe Face Mesh -> locate eye + nose region -> create mask.
    Apply unsharp(0.8, 2.0) to eye+nose zone.
    Apply unsharp(1.2, 1.3) to the rest.
    Blend using the mask.
    Fallback if no face found or MediaPipe unavailable: unsharp(1.0, 1.5) uniformly.
    """
    face_mesh = _get_face_mesh()
    if face_mesh is None:
        return unsharp_mask(img, 1.0, 1.5)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detection = face_mesh.process(rgb)

    if detection.multi_face_landmarks:
        landmarks = detection.multi_face_landmarks[0]
        h, w = img.shape[:2]

        # Collect eye + nose landmark points
        points = []
        for idx in _EYE_NOSE_INDICES:
            lm = landmarks.landmark[idx]
            x = int(lm.x * w)
            y = int(lm.y * h)
            points.append([x, y])

        # Build convex hull mask for eye+nose region
        hull = cv2.convexHull(np.array(points))
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)

        # Feather mask edges for smooth blending
        mask = cv2.GaussianBlur(mask, (0, 0), 5)
        mask_f = mask.astype(np.float32) / 255.0
        mask_3ch = cv2.merge([mask_f, mask_f, mask_f])

        # Strong sharpening on eye+nose zone
        strong = unsharp_mask(img, 0.8, 2.0).astype(np.float32)
        # Lighter sharpening on the rest
        light = unsharp_mask(img, 1.2, 1.3).astype(np.float32)

        # Blend: mask selects strong region, inverse selects light region
        blended = strong * mask_3ch + light * (1.0 - mask_3ch)
        return np.clip(blended, 0, 255).astype(np.uint8)
    else:
        # Fallback: two-pass uniform sharpening (compensates for missing zone targeting)
        img = unsharp_mask(img, 1.0, 1.5)
        # Light second pass to boost fine detail
        img = unsharp_mask(img, 0.5, 0.6)
        return img


# ---------------------------------------------------------------------------
# FULL PIPELINE — do not change this function
# ---------------------------------------------------------------------------

def enhance_face(img: np.ndarray) -> np.ndarray:
    """Run all 4 stages in order. Do not modify."""
    img = stage1_denoise(img)
    img = stage2_clahe(img)
    img = stage3_upscale(img)
    img = stage4_zone_sharpen(img)
    return img


# ---------------------------------------------------------------------------
# EVALUATION HELPERS
# ---------------------------------------------------------------------------

def sharpness(img: np.ndarray) -> float:
    """Laplacian variance. Higher = sharper. Convert to grayscale first."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def get_face_encoding(img: np.ndarray):
    """
    128-d face encoding. Return numpy array if face found, else None.
    For 240x240 crops (known face images), provides the full image as the
    face location to skip the expensive HOG detection scan.
    """
    import face_recognition
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    if min(h, w) >= 200:
        # Already a 240x240 face crop — skip detection, use full image as face box
        # face_recognition uses (top, right, bottom, left) format
        margin = int(min(h, w) * 0.05)
        locations = [(margin, w - margin, h - margin, margin)]
    else:
        locations = face_recognition.face_locations(rgb, number_of_times_to_upsample=2,
                                                    model="hog")
    if not locations:
        return None
    encodings = face_recognition.face_encodings(rgb, locations)
    return encodings[0] if encodings else None


def ssim_score(a: np.ndarray, b: np.ndarray) -> float:
    """
    Structural Similarity Index between two images.
    Both resized to TARGET_SIZE before comparison. Convert to grayscale.
    Return float. Higher = more similar.
    """
    from skimage.metrics import structural_similarity
    a_resized = cv2.resize(a, TARGET_SIZE)
    b_resized = cv2.resize(b, TARGET_SIZE)
    a_gray = cv2.cvtColor(a_resized, cv2.COLOR_BGR2GRAY)
    b_gray = cv2.cvtColor(b_resized, cv2.COLOR_BGR2GRAY)
    return structural_similarity(a_gray, b_gray)


# ---------------------------------------------------------------------------
# HTML A/B REPORT
# ---------------------------------------------------------------------------

def generate_ab_report(results: list, output_path: Path):
    """
    Self-contained HTML. No CDN.
    Summary header: overall accuracy improvement + sharpness gain.
    Grid: each row = original image | enhanced image | sharpness before/after | match before/after.
    Images embedded as base64.
    """
    n = len(results)
    if n == 0:
        output_path.write_text("<html><body><h1>No faces processed</h1></body></html>")
        return

    # Compute summary stats
    acc_before = sum(1 for r in results if r["match_before"]) / n * 100
    acc_after = sum(1 for r in results if r["match_after"]) / n * 100
    avg_sharp_before = np.mean([r["sharpness_before"] for r in results])
    avg_sharp_after = np.mean([r["sharpness_after"] for r in results])
    avg_ssim = np.mean([r["ssim_improvement"] for r in results])
    sharp_gain = avg_sharp_after - avg_sharp_before

    # Build per-face cards
    cards_html = ""
    for i, r in enumerate(results):
        match_b_icon = '<span style="color:#22c55e;font-weight:700">&#10003;</span>' if r["match_before"] else '<span style="color:#ef4444;font-weight:700">&#10007;</span>'
        match_a_icon = '<span style="color:#22c55e;font-weight:700">&#10003;</span>' if r["match_after"] else '<span style="color:#ef4444;font-weight:700">&#10007;</span>'
        identity_str = r["matched_identity"] if r["matched_identity"] else "---"
        sharp_change = r["sharpness_after"] - r["sharpness_before"]
        sharp_color = "#22c55e" if sharp_change > 0 else "#ef4444"
        orig_h, orig_w = r["original_size_px"]

        cards_html += f"""
        <div class="card">
            <div class="card-header">
                <span class="card-num">#{i+1}</span>
                <span class="card-name">{r["filename"]}</span>
                <span class="card-size">{orig_w}x{orig_h} &rarr; 240x240</span>
            </div>
            <div class="card-images">
                <div class="img-col">
                    <div class="img-label">Original</div>
                    <img src="data:image/jpeg;base64,{r["raw_b64"]}" alt="original" />
                </div>
                <div class="img-col">
                    <div class="img-label">Enhanced</div>
                    <img src="data:image/jpeg;base64,{r["enhanced_b64"]}" alt="enhanced" />
                </div>
            </div>
            <div class="card-metrics">
                <div class="metric">
                    <span class="metric-label">Sharpness</span>
                    <span class="metric-value">{r["sharpness_before"]:.1f} &rarr; {r["sharpness_after"]:.1f}
                        <span style="color:{sharp_color};font-size:0.85em">({sharp_change:+.1f})</span>
                    </span>
                </div>
                <div class="metric">
                    <span class="metric-label">Recognition</span>
                    <span class="metric-value">{match_b_icon} &rarr; {match_a_icon}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">SSIM</span>
                    <span class="metric-value">{r["ssim_improvement"]:.4f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Identity</span>
                    <span class="metric-value">{identity_str}</span>
                </div>
            </div>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Face Enhancement A/B Report - Sentio Mind</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        background: #0f172a;
        color: #e2e8f0;
        line-height: 1.6;
        padding: 2rem;
    }}
    .container {{ max-width: 1200px; margin: 0 auto; }}

    .header {{
        text-align: center;
        margin-bottom: 2rem;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid #1e293b;
    }}
    .header h1 {{
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 0.25rem;
    }}
    .header .subtitle {{
        font-size: 0.95rem;
        color: #94a3b8;
    }}

    .summary {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-bottom: 2.5rem;
    }}
    .stat-card {{
        background: #1e293b;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        border: 1px solid #334155;
    }}
    .stat-card .stat-label {{
        display: block;
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }}
    .stat-card .stat-value {{
        display: block;
        font-size: 1.6rem;
        font-weight: 700;
        color: #f1f5f9;
    }}
    .stat-card .stat-detail {{
        display: block;
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 0.25rem;
    }}
    .stat-card.highlight .stat-value {{ color: #22c55e; }}

    .grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(520px, 1fr));
        gap: 1.5rem;
    }}
    .card {{
        background: #1e293b;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #334155;
    }}
    .card-header {{
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem 1rem;
        background: #162032;
        border-bottom: 1px solid #334155;
    }}
    .card-num {{
        background: #38bdf8;
        color: #0f172a;
        font-weight: 700;
        font-size: 0.75rem;
        padding: 0.15rem 0.5rem;
        border-radius: 6px;
    }}
    .card-name {{ font-weight: 600; font-size: 0.9rem; }}
    .card-size {{ margin-left: auto; font-size: 0.8rem; color: #64748b; }}

    .card-images {{
        display: flex;
        gap: 1rem;
        padding: 1rem;
        justify-content: center;
    }}
    .img-col {{ text-align: center; }}
    .img-label {{
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }}
    .card-images img {{
        width: 200px;
        height: 200px;
        object-fit: cover;
        border-radius: 8px;
        border: 2px solid #334155;
        image-rendering: pixelated;
    }}

    .card-metrics {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        border-top: 1px solid #334155;
    }}
    .metric {{
        padding: 0.6rem 0.75rem;
        text-align: center;
        border-right: 1px solid #334155;
    }}
    .metric:last-child {{ border-right: none; }}
    .metric-label {{
        display: block;
        font-size: 0.7rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }}
    .metric-value {{
        display: block;
        font-size: 0.85rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-top: 0.15rem;
    }}

    .footer {{
        text-align: center;
        margin-top: 2.5rem;
        padding-top: 1.5rem;
        border-top: 1px solid #1e293b;
        font-size: 0.8rem;
        color: #475569;
    }}

    @media (max-width: 600px) {{
        .grid {{ grid-template-columns: 1fr; }}
        .card-images img {{ width: 140px; height: 140px; }}
        .card-metrics {{ grid-template-columns: repeat(2, 1fr); }}
        .metric {{ border-bottom: 1px solid #334155; }}
    }}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>Face Enhancement A/B Report</h1>
        <div class="subtitle">Sentio Mind &middot; Project 4 &middot; Low-Resolution CCTV Face Enhancement</div>
    </div>

    <div class="summary">
        <div class="stat-card">
            <span class="stat-label">Faces Processed</span>
            <span class="stat-value">{n}</span>
        </div>
        <div class="stat-card highlight">
            <span class="stat-label">Recognition Accuracy</span>
            <span class="stat-value">{acc_before:.1f}% &rarr; {acc_after:.1f}%</span>
            <span class="stat-detail">+{acc_after - acc_before:.1f}pp improvement</span>
        </div>
        <div class="stat-card highlight">
            <span class="stat-label">Avg Sharpness</span>
            <span class="stat-value">{avg_sharp_before:.1f} &rarr; {avg_sharp_after:.1f}</span>
            <span class="stat-detail">+{sharp_gain:.1f} gain</span>
        </div>
        <div class="stat-card">
            <span class="stat-label">Avg SSIM</span>
            <span class="stat-value">{avg_ssim:.4f}</span>
        </div>
    </div>

    <div class="grid">
        {cards_html}
    </div>

    <div class="footer">
        Generated by solution.py &middot; Sentio Mind Face Enhancement Pipeline
    </div>
</div>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t_start = time.time()

    # --- Step 0: Extract faces from Video_1/ folder if raw_faces/ is empty --
    existing = list(RAW_FACES_DIR.glob("*.jpg")) + list(RAW_FACES_DIR.glob("*.jpeg")) + list(RAW_FACES_DIR.glob("*.png"))
    if not existing:
        print("=" * 55)
        print("  raw_faces/ is empty — extracting from Video_1/ ...")
        print("=" * 55)

        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}
        video_files = sorted([
            v for v in VIDEO_DIR.glob("*") if v.suffix.lower() in video_exts
        ]) if VIDEO_DIR.exists() else []

        if not video_files:
            print(f"  ERROR: No video files found in {VIDEO_DIR}/")
            print(f"  Place your CCTV video(s) inside the Video_1/ folder and re-run.")
            exit(1)

        total_extracted = 0
        for vf in video_files:
            total_extracted = extract_faces_from_video(vf, RAW_FACES_DIR,
                                                       start_count=total_extracted)

        print(f"  Done! Extracted {total_extracted} face crops into {RAW_FACES_DIR}/")
        if total_extracted == 0:
            print("  ERROR: No faces detected in any video.")
            exit(1)
        print()
    else:
        print(f"  Found {len(existing)} faces in raw_faces/, skipping extraction.")

    # Load reference encodings for evaluation
    print(f"  Reference folder: {REFERENCE_DIR}/")
    reference_encodings = {}
    if REFERENCE_DIR.exists():
        for ref in sorted(REFERENCE_DIR.glob("*")):
            if ref.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            img = cv2.imread(str(ref))
            if img is None:
                continue
            enc = get_face_encoding(img)
            if enc is not None:
                reference_encodings[ref.stem] = enc
                print(f"  Reference: {ref.stem}")
            else:
                print(f"  WARNING: no face in {ref.name}")
    else:
        print(f"  WARNING: {REFERENCE_DIR}/ not found")

    print(f"Loaded {len(reference_encodings)} reference identities")

    face_paths = sorted(RAW_FACES_DIR.glob("*.jpg")) + sorted(RAW_FACES_DIR.glob("*.jpeg")) + sorted(RAW_FACES_DIR.glob("*.png"))
    print(f"Processing {len(face_paths)} face crops ...")

    # --- Phase 1: Enhance all faces (timed for 30s budget) -----------------
    t_enhance_start = time.time()
    enhanced_data = {}  # filename -> (raw, raw_at_target, enhanced)

    for fp in face_paths:
        raw = cv2.imread(str(fp))
        if raw is None:
            continue

        raw_at_target = cv2.resize(raw, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
        raw_sharp = sharpness(raw_at_target)

        # Bonus: skip full enhancement if already sharp enough at target size
        if raw_sharp > SHARPNESS_SKIP_THRESHOLD:
            enhanced = raw_at_target
        else:
            enhanced = enhance_face(raw.copy())

        cv2.imwrite(str(ENHANCED_DIR / fp.name), enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
        enhanced_data[fp.name] = (raw, raw_at_target, enhanced)

    t_enhance = round(time.time() - t_enhance_start, 2)
    print(f"  Enhancement done in {t_enhance}s")

    # --- Phase 2: Evaluate (not counted toward 30s budget) -----------------
    import face_recognition as fr

    refs_list = list(reference_encodings.values())
    refs_names = list(reference_encodings.keys())

    results = []

    for fp in face_paths:
        if fp.name not in enhanced_data:
            continue

        raw, raw_at_target, enhanced = enhanced_data[fp.name]

        # Sharpness — both at 240x240 for fair comparison
        sharp_b = sharpness(raw_at_target)
        sharp_a = sharpness(enhanced)
        ssim_g = ssim_score(raw_at_target, enhanced)

        # Face encoding on 240x240 images
        enc_raw = get_face_encoding(raw_at_target)
        enc_enh = get_face_encoding(enhanced)

        match_b = False
        match_a = False
        mid = None

        if refs_list:
            if enc_raw is not None:
                match_b = any(fr.compare_faces(refs_list, enc_raw, tolerance=0.60))
            if enc_enh is not None:
                hits = fr.compare_faces(refs_list, enc_enh, tolerance=0.60)
                match_a = any(hits)
                if match_a:
                    mid = refs_names[hits.index(True)]

        # Encode for report
        _, rb = cv2.imencode(".jpg", raw_at_target, [cv2.IMWRITE_JPEG_QUALITY, 82])
        _, eb = cv2.imencode(".jpg", enhanced, [cv2.IMWRITE_JPEG_QUALITY, 82])

        results.append({
            "filename":          fp.name,
            "original_size_px":  list(raw.shape[:2]),
            "enhanced_size_px":  list(enhanced.shape[:2]),
            "sharpness_before":  round(sharp_b, 2),
            "sharpness_after":   round(sharp_a, 2),
            "ssim_improvement":  round(ssim_g, 4),
            "match_before":      match_b,
            "match_after":       match_a,
            "matched_identity":  mid,
            "raw_b64":           base64.b64encode(rb).decode(),
            "enhanced_b64":      base64.b64encode(eb).decode(),
        })
        print(f"  {fp.name}: sharp {sharp_b:.1f}->{sharp_a:.1f}  match {match_b}->{match_a}")

    n = len(results)
    t_total = round(time.time() - t_start, 2)

    metrics = {
        "source":                          "p4_face_enhancement",
        "total_faces_processed":           n,
        "processing_time_sec":             t_enhance,
        "pipeline_stages_applied":         ["denoise", "clahe", "upscale_multistep", "zone_sharpen"],
        "recognition_accuracy_before_pct": round(sum(r["match_before"] for r in results) / n * 100, 1) if n else 0.0,
        "recognition_accuracy_after_pct":  round(sum(r["match_after"]  for r in results) / n * 100, 1) if n else 0.0,
        "avg_sharpness_before":            round(float(np.mean([r["sharpness_before"] for r in results])), 2) if results else 0.0,
        "avg_sharpness_after":             round(float(np.mean([r["sharpness_after"]  for r in results])), 2) if results else 0.0,
        "avg_ssim_improvement":            round(float(np.mean([r["ssim_improvement"] for r in results])), 4) if results else 0.0,
        "per_face": [{k: v for k, v in r.items() if k not in ["raw_b64", "enhanced_b64"]} for r in results],
    }

    with open(METRICS_JSON_OUT, "w") as f:
        json.dump(metrics, f, indent=2)

    generate_ab_report(results, REPORT_HTML_OUT)

    print()
    print("=" * 55)
    print(f"  Done in {t_total}s  (enhancement: {t_enhance}s)")
    print(f"  Recognition:  {metrics['recognition_accuracy_before_pct']}%  ->  {metrics['recognition_accuracy_after_pct']}%")
    print(f"  Sharpness:    {metrics['avg_sharpness_before']}  ->  {metrics['avg_sharpness_after']}")
    print(f"  Enhanced  -> {ENHANCED_DIR}/")
    print(f"  Report    -> {REPORT_HTML_OUT}")
    print(f"  Metrics   -> {METRICS_JSON_OUT}")
    print("=" * 55)
