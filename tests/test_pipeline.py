import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from teeth_seg import segment  # noqa: E402
from teeth_seg.config import WIDTH_PCT_LOWER, WIDTH_PCT_UPPER, arch_widths  # noqa: E402
from teeth_seg.pipeline import load_gray  # noqa: E402
from teeth_seg.preprocess import stretch_histogram  # noqa: E402
from teeth_seg.roi import parabola_through  # noqa: E402


def test_widths_normalized():
    for t in (WIDTH_PCT_UPPER, WIDTH_PCT_LOWER):
        w = arch_widths(t)
        assert len(w) == 16 and abs(sum(w) - 1) < 1e-9


def test_parabola_through():
    a, b, c = parabola_through((0, 1), (1, 2), (2, 5))  # y = x^2 + 1
    assert np.allclose([a, b, c], [1, 0, 1])


def test_stretch():
    im = np.array([[50, 100], [150, 200]], np.uint8)
    out = stretch_histogram(im)
    assert out.min() == 0 and out.max() == 255


def test_real_image():
    imgs = sorted((ROOT / "img").glob("*.jpg"))
    if not imgs:
        return
    r = segment(load_gray(imgs[0]))
    assert len(r.upper_diastemas) >= 2 and len(r.teeth) > 0
