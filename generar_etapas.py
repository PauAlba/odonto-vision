"""Guarda 10 imágenes por etapa (mediana, estiramiento_histograma, binarizacion, region_inicial) en output/<etapa>/.

Nomenclatura: NOMBREDELARCHIVO_NOMBREDELAFUNCION.jpg. Usa sólo imágenes de img/ con nombre de persona.
"""
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from teeth_seg.config import Params  # noqa: E402
from teeth_seg.pipeline import load_gray  # noqa: E402
from teeth_seg.preprocess import stretch_histogram  # noqa: E402
from teeth_seg.roi import binarize, detect_roi, initial_rect, threshold_from_region  # noqa: E402

N_IMAGES = 10
p = Params()


def save(path: Path, im):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", im)
    assert ok
    buf.tofile(str(path))


# Con nombre de persona: <id>_<NOMBRE APELLIDO>_Panorama_<fecha>.jpg
files = sorted(f for f in (ROOT / "img").glob("*.jpg") if re.match(r"^\d+_[^_]+_Panorama", f.name))[:N_IMAGES]

for f in files:
    g = load_gray(f)
    med = cv2.medianBlur(g, p.median_ksize)
    est = stretch_histogram(med)
    r0 = initial_rect(est.shape, p)
    rect, _, _ = detect_roi(est, p)
    theta = threshold_from_region(est, r0)
    bina = binarize(est, theta) * 255
    vis = cv2.cvtColor(est, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(vis, (r0.c0, r0.r0), (r0.c1, r0.r1), (0, 255, 255), 3)   # inicial (amarillo)
    cv2.rectangle(vis, (rect.c0, rect.r0), (rect.c1, rect.r1), (255, 255, 0), 3)  # rectificada (cian)
    save(ROOT / "output/mediana" / f"{f.stem}_mediana.jpg", med)
    save(ROOT / "output/estiramiento_histograma" / f"{f.stem}_estiramiento_histograma.jpg", est)
    save(ROOT / "output/binarizacion" / f"{f.stem}_binarizacion.jpg", bina)
    save(ROOT / "output/region_inicial" / f"{f.stem}_region_inicial.jpg", vis)
    print("OK", f.name)
