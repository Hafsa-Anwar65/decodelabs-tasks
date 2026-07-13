"""
generate_test_images.py
------------------------
DecodeLabs Project 2 does not ship a real camera dataset, so this script
SYNTHESIZES a small validation set of gear images using OpenCV drawing
primitives:

    - 5 "PASS" gears  -> perfect, symmetric teeth
    - 5 "FAIL" gears  -> one broken/missing tooth (simulates a real
                         manufacturing defect: chipped or sheared tooth)

This gives gear_inspection.py something real to run against, and mirrors
the slide deck's "10 perfect + 10 defective validation dataset" concept
(scaled down to 5+5 for speed; change N_PASS / N_FAIL below to expand).

Run:
    python3 generate_test_images.py
"""

import math
import os
import random

import cv2
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "test_images")
os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE = 500
CENTER = (IMG_SIZE // 2, IMG_SIZE // 2)
OUTER_R = 180
INNER_R = 150
N_TEETH = 20

N_PASS = 5
N_FAIL = 5


def make_gear_points(outer_r, inner_r, n_teeth, broken_tooth_idx=None, crack=False):
    """Generate polygon points for a gear shape. Optionally break one tooth
    to simulate a real defect."""
    points = []
    n_points = n_teeth * 2
    for i in range(n_points):
        angle = 2 * math.pi * i / n_points
        tooth_idx = i // 2

        r = outer_r if i % 2 == 0 else inner_r

        # Simulate a broken/missing tooth: collapse that tooth's radius
        # down toward the inner radius (like it sheared off).
        if broken_tooth_idx is not None and tooth_idx == broken_tooth_idx:
            r = inner_r * 0.92

        x = CENTER[0] + r * math.cos(angle)
        y = CENTER[1] + r * math.sin(angle)
        points.append([x, y])

    pts = np.array(points, dtype=np.int32)

    return pts


def add_noise(img, amount=6):
    """Add mild Gaussian sensor noise so the pipeline's blur/threshold
    stages have something real to do (mirrors 'raw frames capture
    high-frequency noise' from the slides)."""
    noise = np.random.randint(-amount, amount, img.shape, dtype=np.int16)
    noisy = img.astype(np.int16) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def draw_gear(defective=False):
    img = np.full((IMG_SIZE, IMG_SIZE, 3), 235, dtype=np.uint8)  # light background

    broken_idx = random.randint(0, N_TEETH - 1) if defective else None
    pts = make_gear_points(OUTER_R, INNER_R, N_TEETH, broken_tooth_idx=broken_idx)

    # Draw the gear body (dark metallic gray)
    cv2.fillPoly(img, [pts], color=(90, 90, 90))
    # Center hole
    cv2.circle(img, CENTER, 40, (235, 235, 235), -1)

    # Optional crack defect for extra variety
    if defective and random.random() > 0.5:
        angle = random.uniform(0, 2 * math.pi)
        x1 = int(CENTER[0] + INNER_R * 0.5 * math.cos(angle))
        y1 = int(CENTER[1] + INNER_R * 0.5 * math.sin(angle))
        x2 = int(CENTER[0] + OUTER_R * 1.0 * math.cos(angle))
        y2 = int(CENTER[1] + OUTER_R * 1.0 * math.sin(angle))
        cv2.line(img, (x1, y1), (x2, y2), (235, 235, 235), 4)

    img = add_noise(img)
    return img


def main():
    random.seed(42)
    np.random.seed(42)

    for i in range(N_PASS):
        img = draw_gear(defective=False)
        path = os.path.join(OUT_DIR, f"part_{i+1:02d}_OK.jpg")
        cv2.imwrite(path, img)
        print(f"Created {path}")

    for i in range(N_FAIL):
        img = draw_gear(defective=True)
        path = os.path.join(OUT_DIR, f"part_{i+1+N_PASS:02d}_DEFECT.jpg")
        cv2.imwrite(path, img)
        print(f"Created {path}")

    print(f"\nDone. {N_PASS + N_FAIL} images written to {OUT_DIR}")


if __name__ == "__main__":
    main()
