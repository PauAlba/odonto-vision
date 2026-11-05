"""F6. Aislamiento de piezas: ancho (Ec. 3.9), largo (Ec. 3.12), orientación y recorte."""
from dataclasses import dataclass

import cv2
import numpy as np

from .config import Params


@dataclass
class Tooth:
    arch: str            # "upper" | "lower"
    index: int           # posición izquierda->derecha dentro de la arcada
    center: tuple        # (x, y)
    width: float
    length: float
    angle_deg: float     # rotación aplicada para enderezar la pieza
    image: np.ndarray    # recorte


def isolate_arch(img, diastemas, coef, arch: str, p: Params):
    if len(diastemas) < 2:
        return []
    pts = np.array(diastemas)
    widths = np.hypot(*np.diff(pts, axis=0).T)                 # Ec. 3.9
    length = float(np.mean(p.length_factor * widths))          # Ec. 3.12, promedio por arcada
    deriv = np.polyder(coef)
    teeth = []
    for i, w in enumerate(widths):
        cx, cy = (pts[i] + pts[i + 1]) / 2.0
        alpha = float(np.degrees(np.arctan(np.polyval(deriv, cx))))
        wi, li = max(int(round(w)), 1), max(int(round(length)), 1)
        M = cv2.getRotationMatrix2D((cx, cy), alpha, 1.0)
        M[0, 2] += wi / 2.0 - cx
        M[1, 2] += li / 2.0 - cy
        crop = cv2.warpAffine(img, M, (wi, li), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
        teeth.append(Tooth(arch, i, (float(cx), float(cy)), float(w), length, alpha, crop))
    return teeth
