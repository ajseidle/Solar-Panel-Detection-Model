
"""
Purpose:
Converts each detected object into a structured set of brightness, texture, and shape features, enabling quantitative analysis of model behavior and detection quality.
"""
import cv2
import numpy as np

def compute_object_metrics(patch):
    if patch is None or patch.size == 0:
        return None

    h, w = patch.shape[:2]
    area = h * w

    # Brightness via HSV-V
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2].astype(np.float32)

    mean_v = float(V.mean())
    std_v = float(V.std())
    shadow_frac = float((V < 40).mean())

    # Edge density
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(edges.mean() / 255.0)

    # Box proxy
    aspect_ratio = float(w / (h + 1e-6))

    # Pseudo-mask compactness
    _, mask = cv2.threshold(V.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fill_ratio = 0.0
    solidity = 0.0
    compactness = 0.0

    if cnts:
        c = max(cnts, key=cv2.contourArea)
        a = float(cv2.contourArea(c))
        p = float(cv2.arcLength(c, True))

        fill_ratio = a / (area + 1e-6)

        hull = cv2.convexHull(c)
        hull_a = float(cv2.contourArea(hull)) + 1e-6
        solidity = a / hull_a

        compactness = (p * p) / (4.0 * np.pi * a + 1e-6)

    return {
        "mean_v": mean_v,
        "std_v": std_v,
        "shadow_frac": shadow_frac,
        "edge_density": edge_density,
        "aspect_ratio": aspect_ratio,
        "fill_ratio": fill_ratio,
        "solidity": solidity,
        "compactness": compactness,
    }
