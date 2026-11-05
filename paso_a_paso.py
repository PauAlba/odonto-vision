"""Muestra con matplotlib CADA paso del algoritmo, con el título de la operación.

Uso: python paso_a_paso.py img/rad2.jpg            (una ventana por paso; ciérrala para avanzar)
     python paso_a_paso.py img/rad2.jpg --save pasos   (guarda PNG en la carpeta 'pasos')
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
import cv2  # noqa: E402

from teeth_seg.arch_split import characteristic_points, fit_polynomial, median_filter_1d  # noqa: E402
from teeth_seg.config import Params  # noqa: E402
from teeth_seg.diastema_curves import diastema_curves  # noqa: E402
from teeth_seg.interdental import auxiliary_image, detect_both  # noqa: E402
from teeth_seg.isolate import isolate_arch  # noqa: E402
from teeth_seg.pipeline import load_gray  # noqa: E402
from teeth_seg.preprocess import stretch_histogram  # noqa: E402
from teeth_seg.roi import (binarize, best_parabola, initial_rect, largest_component,  # noqa: E402
                           threshold_from_region, Rect)

_n = 0


def show(title, img=None, draw=None, cmap="gray", save=None, fig_size=(13, 6)):
    """Dibuja un paso. `draw(ax)` agrega líneas/puntos encima de la imagen."""
    global _n
    _n += 1
    fig, ax = plt.subplots(figsize=fig_size)
    if img is not None:
        ax.imshow(img, cmap=cmap)
    if draw:
        draw(ax)
    ax.set_title(f"Paso {_n}: {title}")
    ax.axis("off")
    fig.tight_layout()
    if save:
        Path(save).mkdir(exist_ok=True)
        fig.savefig(Path(save) / f"{_n:02d}.png", dpi=100)
        plt.close(fig)
    else:
        plt.show()


def rect_patch(ax, r, color="yellow"):
    ax.add_patch(plt.Rectangle((r.c0, r.r0), r.w, r.h, fill=False, ec=color, lw=2))


def run(path, save=None):
    p = Params()
    g = load_gray(path)
    M, N = g.shape
    S = dict(save=save)

    show("Imagen original (escala de grises)", g, **S)

    # F1 ------------------------------------------------------------
    sm = cv2.medianBlur(g, p.median_ksize)
    show("F1 · Suavizado: filtro de la mediana 3x3 (quita ruido)", sm, **S)

    img = stretch_histogram(sm)
    show("F1 · Mejora de contraste: expansión del histograma", img, **S)

    # F2 ------------------------------------------------------------
    r0 = initial_rect(img.shape, p)
    show("F2 · Espacio de búsqueda inicial (0.2M x 0.6N, centrado)", img,
         lambda ax: rect_patch(ax, r0), **S)

    theta = threshold_from_region(img, r0)
    binary = binarize(img, theta)
    show(f"F2 · Binarización con umbral global θ = {theta} (oscuro = blanco)", binary * 255,
         lambda ax: rect_patch(ax, r0, "red"), **S)

    dmax, best = int(p.alpha * M), (-1, 0)
    for dy in range(-dmax, dmax + 1):
        a, b = r0.r0 + dy, r0.r1 + dy
        if a < 0 or b > M:
            continue
        _, area = largest_component(binary[a:b, r0.c0:r0.c1])
        if area > best[0]:
            best = (area, dy)
    r1 = Rect(r0.r0 + best[1], r0.r1 + best[1], r0.c0, r0.c1)
    show(f"F2 · Mejora: rectángulo desplazado {best[1]} px donde hay la mayor región conexa", img,
         lambda ax: (rect_patch(ax, r0, "gray"), rect_patch(ax, r1, "lime")), **S)

    coef = best_parabola(img, r1, (r1.r0 + r1.r1) / 2.0, p)
    xs = np.arange(r1.c0, r1.c1)
    show("F2 · Rectificación: parábola de mínima intensidad (proyección integral parabólica)", img,
         lambda ax: (rect_patch(ax, r1, "lime"), ax.plot(xs, np.polyval(coef, xs), "r", lw=2)), **S)

    from teeth_seg.roi import detect_roi
    rect, _, _ = detect_roi(img, p)
    show("F2 · Región de interés rectificada (resultado final de F2)", img,
         lambda ax: rect_patch(ax, rect, "cyan"), **S)

    # F3 ------------------------------------------------------------
    sub, _ = largest_component(binary[rect.r0:rect.r1, rect.c0:rect.c1])
    show("F3 · Mayor componente conexa = región entre las arcadas", sub * 255, **S)

    px, py = characteristic_points(binary, rect)
    show("F3 · Quitar partes salientes y tomar el punto medio de cada columna", img,
         lambda ax: ax.plot(px, py, "r.", ms=2), **S)

    pyf = median_filter_1d(py, p.median1d_k)
    show(f"F3 · Filtro de la mediana 1D (K={p.median1d_k}) sobre los puntos", img,
         lambda ax: (ax.plot(px, py, "r.", ms=2, label="crudos"),
                     ax.plot(px, pyf, "c.", ms=2, label="suavizados"), ax.legend()), **S)

    center = fit_polynomial(px, py, p.poly_degree, p.median1d_k)
    show(f"F3 · Regresión polinomial de grado {p.poly_degree} (curva que separa maxilar y mandíbula)", img,
         lambda ax: ax.plot(px, np.polyval(center, px), "lime", lw=2), **S)

    # F4 ------------------------------------------------------------
    up_c, low_c = diastema_curves(img, center, rect, p)
    show("F4 · Curvas de diastemas: la curva central se desplaza (35%-45% de H) hasta la mínima intensidad", img,
         lambda ax: [ax.plot(px, np.polyval(c, px), col, lw=2)
                     for c, col in ((center, "lime"), (up_c, "orange"), (low_c, "orange"))], **S)

    # F5 ------------------------------------------------------------
    sob = cv2.magnitude(cv2.Sobel(img, cv2.CV_32F, 1, 0), cv2.Sobel(img, cv2.CV_32F, 0, 1))
    show("F5 · Bordes: magnitud del gradiente de Sobel", sob, **S)
    show("F5 · Negativo de la imagen (los diastemas quedan brillantes)", 255 - img, **S)
    aux = auxiliary_image(img)
    show("F5 · Imagen auxiliar = negativo + Sobel", aux, **S)

    up_d, low_d = detect_both(img, up_c, low_c, rect.c0, rect.c1, p)

    def dots(ax):
        for d, c in ((up_d, up_c), (low_d, low_c)):
            ax.plot(px, np.polyval(c, px), "orange", lw=1)
            ax.plot([x for x, _ in d], [y for _, y in d], "ro", ms=6)
    show(f"F5 · Diastemas: ventanas {p.win_h}x{p.win_w}, filtro de distancia y centrado "
         f"({len(up_d)} arriba / {len(low_d)} abajo)", aux, dots, **S)

    # F6 ------------------------------------------------------------
    teeth = isolate_arch(img, up_d, up_c, "upper", p) + isolate_arch(img, low_d, low_c, "lower", p)

    def boxes(ax):
        for t in teeth:
            b = cv2.boxPoints((t.center, (t.width, t.length), t.angle_deg))
            ax.add_patch(plt.Polygon(b, fill=False, ec="magenta", lw=1))
    show("F6 · Rectángulos de piezas: ancho = distancia entre diastemas, largo = 3·ancho, girados con la curva",
         img, boxes, **S)

    for arch, name in (("upper", "superior"), ("lower", "inferior")):
        lst = [t for t in teeth if t.arch == arch]
        if not lst:
            continue
        global _n
        _n += 1
        fig, axs = plt.subplots(1, len(lst), figsize=(1.2 * len(lst), 3), squeeze=False)
        for a, t in zip(axs[0], lst):
            a.imshow(t.image, cmap="gray", vmin=0, vmax=255)
            a.set_title(str(t.index + 1), fontsize=8)
            a.axis("off")
        fig.suptitle(f"Paso {_n}: F6 · Piezas extraídas, arcada {name} (izquierda → derecha)")
        if save:
            fig.savefig(Path(save) / f"{_n:02d}.png", dpi=100)
            plt.close(fig)
        else:
            plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--save")
    a = ap.parse_args()
    run(a.image, a.save)
