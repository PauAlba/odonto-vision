import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent / "src"))
from teeth_seg import segment  # noqa: E402
from teeth_seg.pipeline import load_gray  # noqa: E402


def save(path: Path, im):
    ok, buf = cv2.imencode(path.suffix or ".png", im)
    if ok:
        buf.tofile(str(path))


def main():
    ap = argparse.ArgumentParser(description="Segmentación de panorámicas dentales")
    ap.add_argument("input", help="imagen o carpeta")
    ap.add_argument("-o", "--out", default="output")
    a = ap.parse_args()
    src = Path(a.input)
    files = sorted(src.glob("*.jpg")) if src.is_dir() else [src]
    for f in files:
        try:
            r = segment(load_gray(f))
        except Exception as e:  # imagen fallida no detiene el lote
            print(f"[ERROR] {f.name}: {e}")
            continue
        d = Path(a.out) / f.stem
        d.mkdir(parents=True, exist_ok=True)
        save(d / "overlay.png", r.overlay)
        for t in r.teeth:
            save(d / f"{t.arch}_{t.index:02d}.png", t.image)
        print(f"[OK] {f.name}: {len(r.teeth)} piezas")


if __name__ == "__main__":
    main()
