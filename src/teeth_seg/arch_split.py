"""F3. Separación de arcadas: puntos característicos + regresión polinomial (mínimos cuadrados)."""
import numpy as np

from .config import Params
from .roi import Rect, largest_component


def median_filter_1d(v: np.ndarray, k: int) -> np.ndarray:
    k = k | 1  # impar
    pad = k // 2
    vp = np.pad(v, pad, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(vp, k)
    return np.median(win, axis=1)


def characteristic_points(binary: np.ndarray, rect: Rect):
    """Máscara de la región entre arcadas -> puntos medios por columna (x, y) en coords. de imagen."""
    sub, _ = largest_component(binary[rect.r0:rect.r1, rect.c0:rect.c1])
    cols = np.where(sub.any(axis=0))[0]
    if cols.size < 3:
        raise ValueError("No se detectó la región entre arcadas")
    first = np.array([np.argmax(sub[:, c]) for c in cols])
    last = np.array([sub.shape[0] - 1 - np.argmax(sub[::-1, c]) for c in cols])
    mean_h = (last - first + 1).mean()
    # Suprimir pixeles sobresalientes: conservar sólo mean_h pixeles desde abajo
    first = np.maximum(first, np.ceil(last - mean_h + 1)).astype(int)
    mid = (first + last) / 2.0
    return (cols + rect.c0).astype(float), mid + rect.r0


def fit_polynomial(xs, ys, degree: int, k: int):
    ys_f = median_filter_1d(ys, k)
    return np.polyfit(xs, ys_f, degree)  # mínimos cuadrados


def split_arches(binary: np.ndarray, rect: Rect, p: Params):
    """Devuelve coeficientes de la curva C(x) que separa maxilar y mandíbula."""
    xs, ys = characteristic_points(binary, rect)
    return fit_polynomial(xs, ys, p.poly_degree, p.median1d_k)
