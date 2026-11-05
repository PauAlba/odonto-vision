"""F2. Región de interés inicial: binarización, componentes conexas y proyección parabólica."""
from dataclasses import dataclass

import cv2
import numpy as np

from .config import Params


@dataclass
class Rect:
    r0: int
    r1: int  # exclusivo
    c0: int
    c1: int  # exclusivo

    @property
    def h(self):
        return self.r1 - self.r0

    @property
    def w(self):
        return self.c1 - self.c0


def initial_rect(shape, p: Params) -> Rect:
    M, N = shape
    h, w = int(p.search_h_frac * M), int(p.search_w_frac * N)
    r0, c0 = (M - h) // 2, (N - w) // 2
    return Rect(r0, r0 + h, c0, c0 + w)


def threshold_from_region(img: np.ndarray, rect: Rect) -> int:
    """Ecs. 3.2 y 3.3 + acumulación de histograma -> umbral global θ."""
    mean = img[rect.r0:rect.r1, rect.c0:rect.c1].mean()
    P = 100.0 * mean / 255.0
    hist = np.bincount(img.ravel(), minlength=256)
    cum = np.cumsum(hist) * 100.0 / img.size
    theta = 0
    while theta < 255 and cum[theta] < P:
        theta += 1
    return theta


def binarize(img: np.ndarray, theta: int) -> np.ndarray:
    """Región oscura = 1 (foreground)."""
    return (img <= theta).astype(np.uint8)


def largest_component(mask: np.ndarray):
    """Devuelve (máscara de la mayor componente, área)."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    if n <= 1:
        return np.zeros_like(mask), 0
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == k).astype(np.uint8), int(stats[k, cv2.CC_STAT_AREA])


def parabola_through(p0, p1, p2):
    """Ec. 3.5: coeficientes (a, b, c) de la parábola por tres puntos (x, y)."""
    A = np.array([[x * x, x, 1.0] for x, _ in (p0, p1, p2)])
    y = np.array([p0[1], p1[1], p2[1]], dtype=float)
    return np.linalg.solve(A, y)


def curve_sum(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> float:
    """Proyección integral a lo largo de una curva (media de intensidades)."""
    yi = np.clip(np.rint(ys).astype(int), 0, img.shape[0] - 1)
    return float(img[yi, xs].mean())


def best_parabola(img, rect: Rect, row_c: float, p: Params, min_sum_only=True):
    """Familia de parábolas (punto medio y desplazamiento vertical) -> la de mínima intensidad."""
    xs = np.arange(rect.c0, rect.c1)
    xm = (rect.c0 + rect.c1 - 1) / 2.0
    best = (np.inf, None)
    for shift in range(-p.parabola_shift_max, p.parabola_shift_max + 1, p.parabola_step):
        for sag in range(-p.parabola_sag_max, p.parabola_sag_max + 1, p.parabola_step):
            if sag == 0:  # puntos alineados -> sistema singular
                continue
            r = row_c + shift
            a, b, c = parabola_through((xs[0], r), (xm, r + sag), (xs[-1], r))
            ys = a * xs * xs + b * xs + c
            s = curve_sum(img, xs, ys)
            if s < best[0]:
                best = (s, (a, b, c))
    return best[1]


def detect_roi(img: np.ndarray, p: Params):
    """Devuelve (rect_rectificado, binaria, umbral)."""
    M, N = img.shape
    rect0 = initial_rect(img.shape, p)
    theta = threshold_from_region(img, rect0)
    binary = binarize(img, theta)

    # Mejora iterativa: desplazar el rectángulo y conservar el de mayor región conexa
    dmax = int(p.alpha * M)
    best = (-1, 0)
    for dy in range(-dmax, dmax + 1):
        r0, r1 = rect0.r0 + dy, rect0.r1 + dy
        if r0 < 0 or r1 > M:
            continue
        _, area = largest_component(binary[r0:r1, rect0.c0:rect0.c1])
        if area > best[0]:
            best = (area, dy)
    rect1 = Rect(rect0.r0 + best[1], rect0.r1 + best[1], rect0.c0, rect0.c1)

    # Rectificación con proyecciones integrales parabólicas
    coef = best_parabola(img, rect1, (rect1.r0 + rect1.r1) / 2.0, p)
    if coef is not None:
        xs = np.arange(rect1.c0, rect1.c1)
        row = float(np.mean(coef[0] * xs * xs + coef[1] * xs + coef[2]))
        h = rect1.h
        r0 = int(np.clip(round(row - h / 2), 0, M - h))
        rect1 = Rect(r0, r0 + h, rect1.c0, rect1.c1)
    return rect1, binary, theta
