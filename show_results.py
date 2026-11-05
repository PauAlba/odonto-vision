"""Muestra con matplotlib el overlay y las piezas ordenadas desde output/.

Uso: python show_results.py [nombre ...]   (sin args: recorre todas las carpetas)
     python show_results.py rad2 --save     (guarda montaje en output/<n>/montaje.png)
"""
import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "output"
FDI_UPPER = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
FDI_LOWER = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]


def read(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_UNCHANGED)


def teeth(folder, arch):
    files = sorted(folder.glob(f"{arch}_*.png"))  # orden izquierda->derecha
    return [read(f) for f in files]


def show(folder: Path, save=False):
    up, low = teeth(folder, "upper"), teeth(folder, "lower")
    n = max(len(up), len(low), 1)
    fig = plt.figure(figsize=(max(14, n * 1.1), 11))
    gs = fig.add_gridspec(3, n, height_ratios=[4, 1.6, 1.6])
    ax = fig.add_subplot(gs[0, :])
    ax.imshow(cv2.cvtColor(read(folder / "overlay.png"), cv2.COLOR_BGR2RGB))
    ax.set_title(folder.name)
    ax.axis("off")
    for row, (name, lst, fdi) in enumerate((("Superior", up, FDI_UPPER), ("Inferior", low, FDI_LOWER)), 1):
        for i in range(n):
            a = fig.add_subplot(gs[row, i])
            a.axis("off")
            if i < len(lst):
                a.imshow(lst[i], cmap="gray", vmin=0, vmax=255)
                a.set_title(str(fdi[i]) if len(lst) == 16 else str(i + 1), fontsize=8)
            if i == 0:
                a.text(-0.1, 0.5, name, transform=a.transAxes, rotation=90, va="center", ha="right")
    fig.tight_layout()
    if save:
        fig.savefig(folder / "montaje.png", dpi=110)
    plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    dirs = [OUT / n for n in a.names] if a.names else sorted(d for d in OUT.iterdir() if d.is_dir())
    for d in dirs:
        show(d, a.save)
