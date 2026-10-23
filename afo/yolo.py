"""Uklad katalogow w formacie Ultralytics YOLO, budowany z cache AFO.

Obrazy sa podlinkowane twardo (os.link) — zero kosztu na dysku i zero kopiowania
4,5 GB. Snapshot z HF nie jest modyfikowany; wszystko powstaje w <cache>/yolo.
Etykiety sa kopiowane z przycieciem wspolrzednych do [0,1], zgodnie z protokolem.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import yaml

from .dataset import SPLITS, AFODataset
from .download import dataset_root

#: Ultralytics wymaga pary katalogow images/<split> i labels/<split>
YOLO_DIRNAME = "yolo"


def _link_or_copy(src: Path, dst: Path) -> str:
    """Twarde dowiazanie; gdy sie nie da (inny wolumin, FS bez hardlinkow) — kopia."""
    if dst.exists():
        return "exists"
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def _resize_to(src: Path, dst: Path, max_side: int, quality: int = 95) -> str:
    """Zapisuje kopie o dluzszym boku <= max_side. Obrazy mniejsze — dowiazuje.

    Powod: oryginaly maja 2,5K-4K, a dekodowanie JPEG-a jest waskim gardlem
    treningu (domyslna augmentacja mozaikowa czyta 4 obrazy na probke).
    Przeskalowanie raz, z wyprzedzeniem, skraca dekodowanie ~2,8x i zbija
    zbior z 4,5 GB do ~0,9 GB, czyli miesci go w cache'u systemu plikow.
    Etykiety sa znormalizowane, wiec skalowanie ich NIE dotyczy.
    """
    if dst.exists():
        return "exists"
    import cv2

    im = cv2.imread(str(src))
    if im is None:
        raise RuntimeError(f"Nie udalo sie wczytac {src}")
    h, w = im.shape[:2]
    if max(h, w) <= max_side:
        return _link_or_copy(src, dst)
    r = max_side / max(h, w)
    out = cv2.resize(im, (round(w * r), round(h * r)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst), out, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "resize"


def prepare_yolo(
    root: Path | str | None = None,
    out_dir: Path | str | None = None,
    clip: bool = True,
    force: bool = False,
    max_side: int | None = None,
) -> Path:
    """Buduje uklad YOLO i zwraca sciezke do pliku ``afo.yaml``.

    ``max_side`` (np. 1280) zapisuje przeskalowane kopie obrazow zamiast
    dowiazan — patrz ``_resize_to``. None zostawia oryginaly.

    Nic nie znika po cichu: liczba obrazow i etykiet jest porownywana z loaderem,
    a rozbieznosc konczy sie wyjatkiem, nie cichym pominieciem.
    """
    root = dataset_root(root)
    if out_dir:
        out = Path(out_dir)
    else:
        out = root / (f"{YOLO_DIRNAME}{max_side}" if max_side else YOLO_DIRNAME)
    out.mkdir(parents=True, exist_ok=True)

    class_names: list[str] = []
    stats: dict[str, dict[str, int]] = {}

    for split in SPLITS:
        ds = AFODataset(split, root=root, clip=clip)
        class_names = list(ds.class_names)

        img_dir = out / "images" / split
        lbl_dir = out / "labels" / split
        if force:
            shutil.rmtree(img_dir, ignore_errors=True)
            shutil.rmtree(lbl_dir, ignore_errors=True)
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        modes = {"link": 0, "copy": 0, "exists": 0, "resize": 0}
        n_obj = 0
        records = ds.records
        try:
            from tqdm import tqdm

            records = tqdm(records, desc=f"yolo/{split}", unit="img", leave=False)
        except ImportError:
            pass
        for rec in records:
            dst = img_dir / rec.filename
            if max_side:
                modes[_resize_to(rec.image_path, dst, max_side)] += 1
            else:
                modes[_link_or_copy(rec.image_path, dst)] += 1
            lines = [
                f"{a.class_id} {a.cx:.6f} {a.cy:.6f} {a.w:.6f} {a.h:.6f}"
                for a in rec.annotations
            ]
            n_obj += len(lines)
            # pusty plik dla obrazu tla — Ultralytics traktuje go jako negatyw
            (lbl_dir / (Path(rec.filename).stem + ".txt")).write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )

        n_img = len(list(img_dir.iterdir()))
        n_lbl = len(list(lbl_dir.iterdir()))
        if n_img != len(ds) or n_lbl != len(ds):
            raise RuntimeError(
                f"{split}: loader ma {len(ds)} rekordow, a na dysku jest "
                f"{n_img} obrazow i {n_lbl} etykiet — przerwane, zeby nie uczyc "
                f"na niepelnym zbiorze."
            )
        stats[split] = {"images": n_img, "objects": n_obj, **modes}
        print(
            f"[yolo/{split}] {n_img} obrazow ("
            + ", ".join(f"{v} {k}" for k, v in modes.items() if v)
            + f"), {n_obj} obiektow"
        )

    cfg = {
        "path": str(out).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: n for i, n in enumerate(class_names)},
    }
    if max_side:
        cfg["_note"] = (
            f"Obrazy przeskalowane do dluzszego boku {max_side} px (INTER_AREA, JPEG q95). "
            f"Etykiety znormalizowane, wiec niezmienione."
        )
    yaml_path = out / "afo.yaml"
    yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"[yolo] {yaml_path}")
    return yaml_path


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-clip", action="store_true")
    ap.add_argument("--force", action="store_true", help="przebuduj od zera")
    ap.add_argument(
        "--max-side", type=int, default=None,
        help="przeskaluj obrazy do tego dluzszego boku (np. 1280) zamiast dowiazywac",
    )
    args = ap.parse_args()
    prepare_yolo(
        args.root, args.out, clip=not args.no_clip, force=args.force, max_side=args.max_side
    )


if __name__ == "__main__":
    main()
