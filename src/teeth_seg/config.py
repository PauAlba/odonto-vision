"""Parámetros y constantes de la tesis (Cap. 3)."""
from dataclasses import dataclass

# Tabla 3.1: % del ancho por pieza, orden hemiarcada (tercer molar -> incisivo central)
PIECES = ["M3", "M2", "M1", "PM2", "PM1", "C", "IL", "IC"]
WIDTH_PCT_UPPER = [6.7376, 7.1631, 6.0283, 4.8581, 5.2127, 5.0354, 5.3900, 6.0993]
WIDTH_PCT_LOWER = [7.7762, 7.8114, 6.5095, 4.9613, 4.8557, 4.3983, 3.8705, 3.7297]

# Tabla 3.2: relación longitud-ancho (informativa; la tesis usa l = 3w, Ec. 3.12)
LW_RATIO_UPPER = [2.5000, 2.6973, 2.7200, 2.8181, 2.5581, 2.5566, 2.3673, 2.1607]
LW_RATIO_LOWER = [2.5733, 2.8052, 2.8181, 2.7926, 2.6363, 2.4454, 2.4362, 2.4318]


def arch_widths(pcts):
    """Anchos relativos izquierda->derecha (16 piezas), normalizados a suma 1."""
    full = list(pcts) + list(reversed(pcts))
    total = sum(full)
    return [p / total for p in full]


@dataclass
class Params:
    # F1
    median_ksize: int = 3
    # F2
    search_h_frac: float = 0.2      # 0.2*M
    search_w_frac: float = 0.6      # 0.6*N
    alpha: float = 0.05             # Ec. 3.4
    parabola_step: int = 10         # separación entre parábolas (px)
    parabola_sag_max: int = 60      # desplazamiento máx. del punto medio (px)
    parabola_shift_max: int = 40    # desplazamiento vertical máx. (px)
    # F3
    median1d_k: int = 25
    poly_degree: int = 2
    # F4  (Ec. 3.6, 3.7)
    diastema_lo: float = 0.35
    diastema_hi: float = 0.45
    # F5
    win_h: int = 31                 # ventana 31x9 (alto x ancho)
    win_w: int = 9
    min_dist_frac: float = 0.70     # filtro euclidiano
    interval_frac: float = 0.5      # semiancho de búsqueda (x ancho mínimo vecino)
    center_frac: float = 0.3        # rango de centrado (x ancho de pieza contigua)
    # F6
    length_factor: float = 3.0      # Ec. 3.12
