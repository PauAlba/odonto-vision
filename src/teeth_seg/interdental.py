"""F5. Detección de espacios interdentales (Sobel + negativo, ventanas, filtro y centrado)."""
import cv2
import numpy as np

from .config import Params, arch_widths


def auxiliary_image(img: np.ndarray) -> np.ndarray:
    """Sec. 3.5.1: negativo + magnitud del gradiente de Sobel (Ec. 2.32)."""
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag = cv2.normalize(cv2.magnitude(gx, gy), None, 0, 255, cv2.NORM_MINMAX)
    neg = 255.0 - img.astype(np.float32)
    return (neg + mag) / 2.0


class Arch:
    """Curva de soporte muestreada con longitud de arco."""

    def __init__(self, coef, x0, x1):
        self.coef = coef
        self.xs = np.arange(x0, x1, dtype=float)
        self.ys = np.polyval(coef, self.xs)
        seg = np.hypot(np.diff(self.xs), np.diff(self.ys))
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(self.s[-1])

    def point_at(self, s):
        return float(np.interp(s, self.s, self.xs)), float(np.interp(s, self.s, self.ys))

    def idx_at(self, s):
        return int(np.clip(np.searchsorted(self.s, s), 0, len(self.s) - 1))


def _window_scores(aux, arch: Arch, p: Params):
    """Suma de intensidades de la ventana (win_h x win_w) centrada en cada punto de la curva."""
    integ = cv2.integral(aux)
    H, W = aux.shape
    hh, hw = p.win_h // 2, p.win_w // 2
    sc = np.full(len(arch.xs), -np.inf)
    for i, (x, y) in enumerate(zip(arch.xs, arch.ys)):
        xi, yi = int(round(x)), int(round(y))
        r0, r1, c0, c1 = yi - hh, yi + hh + 1, xi - hw, xi + hw + 1
        if r0 < 0 or c0 < 0 or r1 > H or c1 > W:
            continue
        sc[i] = integ[r1, c1] - integ[r0, c1] - integ[r1, c0] + integ[r0, c0]
    return sc


def detect_diastemas(aux, coef, x0, x1, widths_rel, p: Params):
    """Devuelve lista de (x, y) con los diastemas ordenados de izquierda a derecha."""
    arch = Arch(coef, x0, x1)
    sc = _window_scores(aux, arch, p)
    px = np.array(widths_rel) * arch.length          # Ec. 3.8 (ancho de cada pieza en px)
    bounds = np.concatenate([[0.0], np.cumsum(px)])  # 17 límites esperados
    n = len(bounds)

    # Ventana de búsqueda por límite (intervalo de arco alrededor de la posición esperada)
    idx = []
    for k, b in enumerate(bounds):
        adj = [px[j] for j in (k - 1, k) if 0 <= j < len(px)]
        half = p.interval_frac * min(adj)
        a, c = arch.idx_at(b - half), arch.idx_at(b + half)
        seg = sc[a:c + 1]
        idx.append(a + int(np.argmax(seg)) if np.isfinite(seg).any() else arch.idx_at(b))

    # Filtro de distancia euclidiana: < 70% del ancho promedio -> conservar el más brillante
    mean_w = float(px.mean())
    pts = [(i, sc[i]) for i in idx]
    keep = []
    for i, s in pts:
        if keep:
            j, sj = keep[-1]
            d = np.hypot(arch.xs[i] - arch.xs[j], arch.ys[i] - arch.ys[j])
            if d < p.min_dist_frac * mean_w:
                if s > sj:
                    keep[-1] = (i, s)
                continue
        keep.append((i, s))

    # Centrado: búsqueda local de la región más intensa (sin cruzar vecinos)
    idxs = [i for i, _ in keep]
    out = []
    for n_i, i in enumerate(idxs):
        left = idxs[n_i - 1] if n_i > 0 else 0
        right = idxs[n_i + 1] if n_i < len(idxs) - 1 else len(arch.xs) - 1
        rng = p.center_frac * mean_w
        a = max(arch.idx_at(arch.s[i] - rng), (left + i) // 2 if n_i > 0 else 0)
        c = min(arch.idx_at(arch.s[i] + rng), (right + i) // 2 if n_i < len(idxs) - 1 else len(arch.xs) - 1)
        seg = sc[a:c + 1]
        j = a + int(np.argmax(seg)) if np.isfinite(seg).any() else i
        out.append((float(arch.xs[j]), float(arch.ys[j])))
    return out


def detect_both(img, up_coef, low_coef, x0, x1, p: Params):
    from .config import WIDTH_PCT_LOWER, WIDTH_PCT_UPPER
    aux = auxiliary_image(img)
    up = detect_diastemas(aux, up_coef, x0, x1, arch_widths(WIDTH_PCT_UPPER), p)
    low = detect_diastemas(aux, low_coef, x0, x1, arch_widths(WIDTH_PCT_LOWER), p)
    return up, low
