"""F1. Preprocesamiento: mediana + expansión del histograma (Ec. 2.25, 3.1)."""
import cv2
import numpy as np


def stretch_histogram(img: np.ndarray) -> np.ndarray:
    lo, hi = int(img.min()), int(img.max())
    if lo == 0 and hi == 255 or hi == lo:  # Ec. 3.1: G = I
        return img
    out = (img.astype(np.float32) - lo) * 255.0 / (hi - lo)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def preprocess(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    return stretch_histogram(cv2.medianBlur(gray, ksize))
