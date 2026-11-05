"""Orquestación F1..F6."""
from dataclasses import dataclass, field

import cv2
import numpy as np

from .arch_split import split_arches
from .config import Params
from .diastema_curves import diastema_curves
from .interdental import detect_both
from .isolate import Tooth, isolate_arch
from .preprocess import preprocess
from .roi import Rect, detect_roi


@dataclass
class SegmentationResult:
    rect: Rect
    theta: int
    center_coef: np.ndarray
    upper_coef: np.ndarray
    lower_coef: np.ndarray
    upper_diastemas: list
    lower_diastemas: list
    teeth: list = field(default_factory=list)
    overlay: np.ndarray = None


def load_gray(path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {path}")
    return img


def draw_overlay(gray, r: SegmentationResult) -> np.ndarray:
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(out, (r.rect.c0, r.rect.r0), (r.rect.c1, r.rect.r1), (0, 255, 255), 2)
    xs = np.arange(r.rect.c0, r.rect.c1)
    for coef, col in ((r.center_coef, (0, 255, 0)), (r.upper_coef, (255, 128, 0)),
                      (r.lower_coef, (255, 128, 0))):
        pts = np.stack([xs, np.polyval(coef, xs)], 1).astype(np.int32)
        cv2.polylines(out, [pts], False, col, 2)
    for x, y in r.upper_diastemas + r.lower_diastemas:
        cv2.circle(out, (int(x), int(y)), 5, (0, 0, 255), -1)
    for t in r.teeth:
        w, l = t.width, t.length
        box = cv2.boxPoints(((t.center[0], t.center[1]), (w, l), t.angle_deg))
        cv2.polylines(out, [box.astype(np.int32)], True, (255, 0, 255), 1)
    return out


def segment(gray: np.ndarray, params: Params = None) -> SegmentationResult:
    p = params or Params()
    img = preprocess(gray, p.median_ksize)                                 # F1
    rect, binary, theta = detect_roi(img, p)                               # F2
    center = split_arches(binary, rect, p)                                 # F3
    up_c, low_c = diastema_curves(img, center, rect, p)                    # F4
    up_d, low_d = detect_both(img, up_c, low_c, rect.c0, rect.c1, p)      # F5
    teeth = isolate_arch(img, up_d, up_c, "upper", p) + \
        isolate_arch(img, low_d, low_c, "lower", p)                        # F6
    res = SegmentationResult(rect, theta, center, up_c, low_c, up_d, low_d, teeth)
    res.overlay = draw_overlay(gray, res)
    return res
