"""F4. Aproximación polinomial de los diastemas (Ecs. 3.6 y 3.7)."""
import numpy as np

from .config import Params
from .roi import Rect


def _shifted_mean(img, coef, x0, x1, dy):
    xs = np.arange(x0, x1)
    ys = np.polyval(coef, xs) + dy
    yi = np.rint(ys).astype(int)
    if yi.min() < 0 or yi.max() >= img.shape[0]:
        return np.inf
    return float(img[yi, xs].mean())


def diastema_curves(img: np.ndarray, coef, rect: Rect, p: Params):
    """Devuelve (coef_superior, coef_inferior): C(x) desplazada al mínimo de intensidad."""
    H = rect.h
    lo, hi = int(p.diastema_lo * H), int(p.diastema_hi * H)
    offs = range(lo, hi + 1)
    d_s = min(offs, key=lambda d: _shifted_mean(img, coef, rect.c0, rect.c1, -d))
    d_i = min(offs, key=lambda d: _shifted_mean(img, coef, rect.c0, rect.c1, +d))
    up = np.array(coef, dtype=float)
    low = np.array(coef, dtype=float)
    up[-1] -= d_s
    low[-1] += d_i
    return up, low
