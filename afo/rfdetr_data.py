"""Uklad zbioru dla pakietu ``rfdetr`` (konwencja Roboflow / COCO).

    <out>/train/_annotations.coco.json + obrazy
    <out>/valid/_annotations.coco.json + obrazy
    <out>/test/_annotations.coco.json  + obrazy

UWAGA: jesli obrazy sa przeskalowane (patrz ``afo.yolo``), JSON musi zawierac
ICH wymiary, a nie wymiary oryginalu — inaczej ramki w pikselach rozjada sie o
wspolczynnik skali. Dlatego rozmiary czytane sa z plikow, ktore faktycznie leza
w katalogu wyjsciowym, a nie ze snapshotu HF.
"""

from __future__ import annotations

import json
from pathlib import Path

from .coco import CATEGORY_ID_OFFSET, _categories
from .dataset import SPLITS, AFODataset
from .download import dataset_root
from .yolo import _link_or_copy

#: rfdetr oczekuje katalogu "valid", nie "val"
SPLIT_DIRNAME = {"train": "train", "val": "valid", "test": "test"}


def prepare_rfdetr(
    root: Path | str | None = None,
    images_from: Path | str | None = None,
    out_dir: Path | str | None = None,
    clip: bool = True,
) -> Path:
    """Buduje uklad dla rfdetr i zwraca katalog glowny.

    ``images_from`` wskazuje gotowy katalog ``images/<split>`` (np. z
    ``afo.yolo`` po przeskalowaniu). Gdy None — uzywane sa oryginaly.
    """
    root = dataset_root(root)
    src_base = Path(images_from) if images_from else None
    out = Path(out_dir) if out_dir else root / (
        f"rfdetr_{src_base.parent.name}" if src_base else "rfdetr"
    )
    out.mkdir(parents=True, exist_ok=True)

    from PIL import Image

    for split in SPLITS:
        ds = AFODataset(split, root=root, clip=clip)
        dst_dir = out / SPLIT_DIRNAME[split]
        dst_dir.mkdir(parents=True, exist_ok=True)

        images, annotations = [], []
        ann_id = 1
        n_rescaled = 0
        for img_id, rec in enumerate(ds.records, 1):
            src = (src_base / split / rec.filename) if src_base else rec.image_path
            if not src.is_file():
                raise FileNotFoundError(f"Brak obrazu zrodlowego {src}")
            _link_or_copy(src, dst_dir / rec.filename)

            with Image.open(dst_dir / rec.filename) as im:
                w, h = im.size
            if (w, h) != rec.size():
                n_rescaled += 1

            images.append({"id": img_id, "file_name": rec.filename, "width": w, "height": h})
            for a in rec.annotations:
                x, y, bw, bh = a.to_xywh(w, h)
                x0, y0 = max(0.0, x), max(0.0, y)
                x1, y1 = min(x + bw, w), min(y + bh, h)
                bw, bh = x1 - x0, y1 - y0
                if bw <= 0 or bh <= 0:
                    continue
                annotations.append({
                    "id": ann_id, "image_id": img_id,
                    "category_id": a.class_id + CATEGORY_ID_OFFSET,
                    "bbox": [round(x0, 2), round(y0, 2), round(bw, 2), round(bh, 2)],
                    "area": round(bw * bh, 2), "iscrowd": 0, "segmentation": [],
                })
                ann_id += 1

        coco = {
            "info": {"description": f"AFO {split}", "doi": "10.3233/ICA-210649"},
            "licenses": [{"id": 1, "name": "CC BY-NC-SA 3.0"}],
            "images": images,
            "annotations": annotations,
            "categories": _categories(ds.class_names),
        }
        (dst_dir / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")
        note = f", {n_rescaled} obrazow o zmienionym rozmiarze wzgledem oryginalu" if n_rescaled else ""
        print(
            f"[rfdetr/{SPLIT_DIRNAME[split]}] {len(images)} obrazow, "
            f"{len(annotations)} obiektow{note}"
        )
        if len(annotations) != ds.num_objects():
            print(
                f"  UWAGA: loader ma {ds.num_objects()} obiektow, zapisano "
                f"{len(annotations)} — roznica {ds.num_objects() - len(annotations)} "
                f"(ramki o zerowej powierzchni)"
            )

    print(f"[rfdetr] {out}")
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None)
    ap.add_argument("--images-from", default=None, help="katalog <...>/images z afo.yolo")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    prepare_rfdetr(args.root, args.images_from, args.out)


if __name__ == "__main__":
    main()
