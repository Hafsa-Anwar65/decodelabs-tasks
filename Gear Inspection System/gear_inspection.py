"""
gear_inspection.py
--------------------
DecodeLabs Project 2 — Automated Quality Inspection (Computer Vision)

Implements the exact Input -> Process -> Output pipeline from the training
deck:

    INPUT    : load raw image
    PROCESS  : grayscale -> Gaussian blur -> threshold (binarize)
               -> findContours (RETR_EXTERNAL) -> convexHull
               -> convexityDefects -> compare distance to tolerance
    OUTPUT   : draw a red bounding box around any defect + print PASS/FAIL,
               save an annotated image, and (in this simulation) print the
               binary signal that would be sent to a PLC.

Run:
    python3 gear_inspection.py
"""

import glob
import os

import cv2
import numpy as np

# ----------------------------------------------------------------------
# CONFIG — the "Tolerance Gate" from Phase 3 of the slides
# ----------------------------------------------------------------------
INPUT_DIR = os.path.join(os.path.dirname(__file__), "test_images")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

GAUSSIAN_KERNEL = (5, 5)
THRESHOLD_VALUE = 127
THRESHOLD_MAX = 255

# Convexity-defect distance tolerance, in pixels. Anything deeper than
# this is treated as a real structural defect (broken tooth / crack)
# rather than normal manufacturing tolerance on tooth gaps.
DEFECT_DISTANCE_THRESHOLD_PX = 30.0


def preprocess(img):
    """Phase 1: Isolating the Signal from the Noise."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL, 0)
    # THRESH_BINARY_INV because our part is dark-on-light background;
    # invert so the part itself becomes the white foreground blob.
    _, thresh = cv2.threshold(
        blurred, THRESHOLD_VALUE, THRESHOLD_MAX, cv2.THRESH_BINARY_INV
    )
    return gray, blurred, thresh


def find_largest_contour(thresh):
    """Phase 2, Step 1: Tracing the Boundary."""
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    # The part is the largest external contour in frame.
    return max(contours, key=cv2.contourArea)


def analyze_convexity_defects(contour):
    """
    Phase 2, Steps 2-3: The Rubber Band Concept + Measuring the Gaps.

    Returns a list of (start, end, farthest_point, actual_distance_px)
    for every convexity defect deeper than the tolerance gate.
    """
    hull_indices = cv2.convexHull(contour, returnPoints=False)

    # convexityDefects needs at least 3 hull points and a monotonic index
    # order; guard against degenerate contours.
    if hull_indices is None or len(hull_indices) < 3:
        return []

    hull_indices = np.sort(hull_indices, axis=0)
    defects = cv2.convexityDefects(contour, hull_indices)

    if defects is None:
        return []

    significant_defects = []
    for i in range(defects.shape[0]):
        s, e, f, d_raw = defects[i][0]

        # CRITICAL TRAP from the slides: OpenCV returns distance scaled
        # by a fixed-point integer of 256. Must divide to get real pixels.
        actual_distance = d_raw / 256.0

        if actual_distance > DEFECT_DISTANCE_THRESHOLD_PX:
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            significant_defects.append((start, end, far, actual_distance))

    return significant_defects


def draw_verdict(img, defects, contour):
    """Phase 3, Step 3: Actuate — draw the bounding box + PASS/FAIL label."""
    annotated = img.copy()

    if not defects:
        cv2.putText(
            annotated, "PASS", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
            1.1, (0, 150, 0), 3,
        )
        return annotated, "PASS"

    # Bounding box around the whole part (simple + robust for this scale);
    # draw a small red marker at each defect's farthest_point too.
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)

    for (_, _, far, dist) in defects:
        cv2.circle(annotated, far, 6, (0, 0, 255), -1)

    cv2.putText(
        annotated, "FAIL: STRUCTURAL DEFECT DETECTED", (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
    )
    return annotated, "FAIL"


def inspect_image(path):
    img = cv2.imread(path)
    if img is None:
        return {"file": os.path.basename(path), "verdict": "ERROR", "defects": []}

    gray, blurred, thresh = preprocess(img)
    contour = find_largest_contour(thresh)

    if contour is None:
        return {"file": os.path.basename(path), "verdict": "ERROR (no part found)", "defects": []}

    defects = analyze_convexity_defects(contour)
    annotated, verdict = draw_verdict(img, defects, contour)

    out_path = os.path.join(OUTPUT_DIR, f"result_{os.path.basename(path)}")
    cv2.imwrite(out_path, annotated)

    return {
        "file": os.path.basename(path),
        "verdict": verdict,
        "defects": defects,
        "output_image": out_path,
    }


def send_to_plc(verdict, filename):
    """Simulates Phase 3, Step 3: transmitting the binary signal to a PLC."""
    signal = 0 if verdict == "PASS" else 1
    print(f"    [PLC SIGNAL] part='{filename}' FAIL_BIT={signal}")


def main():
    image_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.jpg")))
    if not image_paths:
        print(f"No images found in {INPUT_DIR}. Run generate_test_images.py first.")
        return

    print(f"Running inspection pipeline on {len(image_paths)} parts...\n")

    correct = 0
    total = 0
    for path in image_paths:
        result = inspect_image(path)
        ground_truth = "FAIL" if "DEFECT" in result["file"] else "PASS"
        is_correct = result["verdict"] == ground_truth
        correct += int(is_correct)
        total += 1

        mark = "✅" if is_correct else "❌"
        print(f"{mark} {result['file']:25s} -> {result['verdict']:6s} (expected {ground_truth})")

        if result["defects"]:
            for (_, _, far, dist) in result["defects"]:
                print(f"      defect at {far}, depth={dist:.1f}px")

        send_to_plc(result["verdict"], result["file"])

    print(f"\nSorting accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"Annotated images saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
