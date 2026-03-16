"""
face_enhancement.py
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
RAW_FACES_DIR    = Path("raw_faces")
REFERENCE_DIR    = Path("reference_identities")
ENHANCED_DIR     = Path("enhanced_faces")
REPORT_HTML_OUT  = Path("enhancement_report.html")
METRICS_JSON_OUT = Path("evaluation_metrics.json")

TARGET_SIZE      = (240, 240)
ENHANCED_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# STAGE 1 — DENOISE
# ---------------------------------------------------------------------------

def stage1_denoise(img: np.ndarray) -> np.ndarray:
    """
    cv2.fastNlMeansDenoisingColored with h=8, hColor=8, templateWindowSize=7, searchWindowSize=21
    TODO: one line
    """
    # TODO
    return img


# ---------------------------------------------------------------------------
# STAGE 2 — CLAHE
# ---------------------------------------------------------------------------

def stage2_clahe(img: np.ndarray) -> np.ndarray:
    """
    Convert to LAB. Apply CLAHE (clipLimit=3.5, tileGridSize=(4,4)) to L channel. Merge + convert back.
    TODO: implement
    """
    # TODO
    return img


# ---------------------------------------------------------------------------
# STAGE 3 — MULTI-STEP UPSCALE
# ---------------------------------------------------------------------------

def unsharp_mask(img: np.ndarray, sigma: float, strength: float) -> np.ndarray:
    """
    blurred = GaussianBlur(img, sigma)
    result  = img + strength * (img - blurred)
    Clip to 0–255.
    TODO: implement with cv2.GaussianBlur + cv2.addWeighted
    """
    # TODO
    return img


def stage3_upscale(img: np.ndarray) -> np.ndarray:
    """
    If short side < 64px: 2× LANCZOS4 → unsharp(1.0, 1.6) → 2× LANCZOS4 → resize to TARGET_SIZE.
    Otherwise: direct resize to TARGET_SIZE LANCZOS4.
    TODO: implement
    """
    # TODO
    return cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)


# ---------------------------------------------------------------------------
# STAGE 4 — ZONE SHARPENING
# ---------------------------------------------------------------------------

def stage4_zone_sharpen(img: np.ndarray) -> np.ndarray:
    """
    MediaPipe Face Mesh → locate eye + nose region → create mask.
    Apply unsharp(0.8, 2.0) to eye+nose zone.
    Apply unsharp(1.2, 1.3) to the rest.
    Blend using the mask.
    Fallback if no face found: unsharp(1.0, 1.5) uniformly.
    TODO: implement
    """
    # TODO
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
    """Laplacian variance. Higher = sharper. Convert to grayscale first.
    TODO: cv2.Laplacian(gray, cv2.CV_64F).var()
    """
    # TODO
    return 0.0


def get_face_encoding(img: np.ndarray):
    """
    128-d face encoding. Return numpy array if face found, else None.
    Use number_of_times_to_upsample=2 for small faces.
    TODO: implement with face_recognition
    """
    # TODO
    return None


def ssim_score(a: np.ndarray, b: np.ndarray) -> float:
    """
    Structural Similarity Index between two images.
    Both resized to TARGET_SIZE before comparison. Convert to grayscale.
    Return float. Higher = more similar.
    TODO: from skimage.metrics import structural_similarity
    """
    # TODO
    return 0.0


# ---------------------------------------------------------------------------
# HTML A/B REPORT
# ---------------------------------------------------------------------------

def generate_ab_report(results: list, output_path: Path):
    """
    Self-contained HTML. No CDN.

    Summary header: overall accuracy improvement + sharpness gain.
    Grid: each row = original image | enhanced image | sharpness before/after | match before/after.
    Images embedded as base64.

    TODO: implement
    """
    # TODO
    pass


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t_start = time.time()

    # Load reference encodings for evaluation
    reference_encodings = {}
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

    print(f"Loaded {len(reference_encodings)} reference identities")

    face_paths = sorted(RAW_FACES_DIR.glob("*.jpg")) + sorted(RAW_FACES_DIR.glob("*.png"))
    print(f"Processing {len(face_paths)} face crops ...")

    results = []

    for fp in face_paths:
        raw = cv2.imread(str(fp))
        if raw is None:
            continue

        enhanced = enhance_face(raw.copy())
        cv2.imwrite(str(ENHANCED_DIR / fp.name), enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])

        sharp_b  = sharpness(raw)
        sharp_a  = sharpness(enhanced)
        ssim_g   = ssim_score(cv2.resize(raw, TARGET_SIZE), enhanced)

        enc_raw = get_face_encoding(raw)
        enc_enh = get_face_encoding(enhanced)

        match_b = False
        match_a = False
        mid     = None

        if reference_encodings:
            import face_recognition as fr
            refs  = list(reference_encodings.values())
            names = list(reference_encodings.keys())
            if enc_raw is not None:
                match_b = any(fr.compare_faces(refs, enc_raw, tolerance=0.60))
            if enc_enh is not None:
                hits = fr.compare_faces(refs, enc_enh, tolerance=0.60)
                match_a = any(hits)
                if match_a:
                    mid = names[hits.index(True)]

        # Encode for report
        _, rb = cv2.imencode(".jpg", cv2.resize(raw, TARGET_SIZE), [cv2.IMWRITE_JPEG_QUALITY, 82])
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
        print(f"  {fp.name}: sharp {sharp_b:.1f}→{sharp_a:.1f}  match {match_b}→{match_a}")

    n   = len(results)
    t_s = round(time.time() - t_start, 2)

    metrics = {
        "source":                          "p4_face_enhancement",
        "total_faces_processed":           n,
        "processing_time_sec":             t_s,
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
    print(f"  Done in {t_s}s  for {n} faces")
    print(f"  Recognition:  {metrics['recognition_accuracy_before_pct']}%  →  {metrics['recognition_accuracy_after_pct']}%")
    print(f"  Sharpness:    {metrics['avg_sharpness_before']}  →  {metrics['avg_sharpness_after']}")
    print(f"  Enhanced  → {ENHANCED_DIR}/")
    print(f"  Report    → {REPORT_HTML_OUT}")
    print(f"  Metrics   → {METRICS_JSON_OUT}")
    print("=" * 55)
