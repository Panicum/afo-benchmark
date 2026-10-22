"""Eksport AFO do formatu COCO detection JSON.

Ramki z Darknet (znormalizowany srodek + rozmiar) na bezwzgledne piksele
(x, y, w, h) w ukladzie COCO. Rozmiary obrazow czytane z naglowkow plikow.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .dataset import AFODataset

#: identyfikatory kategorii COCO licza sie od 1; klasa 0 z AFO to kategoria 1
CATEGORY_ID_OFFSET = 1


def _categories(class_names: Iterable[str]) -> list[dict]:
    return [
        {"id": i + CATEGORY_ID_OFFSET, "name": name, "supercategory": "afo"}
        for i, name in enumerate(class_names)
    ]


def build_coco(
    ds: AFODataset,
    *,
    drop_degenerate: bool = True,
    progress: bool = True,
) -> dict:
    """Buduje slownik COCO z gotowego ``AFODataset``.

    ``drop_degenerate`` usuwa ramki o zerowej powierzchni po przeliczeniu na
    piksele (pycocotools i tak ich nie oceni). Ile i ktorych — raportowane w
    ``info.notes`` oraz na stdout, nigdy po cichu.
    """
    images: list[dict] = []
    annotations: list[dict] = []
    dropped: list[str] = []
    ann_id = 1

    iterator = enumerate(ds.records, 1)
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, total=len(ds.records), desc=f"COCO {ds.split}", unit="img")
        except ImportError:
            pass

    for img_id, rec in iterator:
        width, height = rec.size()
        images.append(
            {
                "id": img_id,
                "file_name": rec.filename,
                "width": width,
                "height": height,
            }
        )
        for a in rec.annotations:
            x, y, w, h = a.to_xywh(width, height)
            # przyciecie do ramy obrazu (dla wspolrzednych juz przycietych w [0,1] to no-op)
            x0 = max(0.0, min(x, width))
            y0 = max(0.0, min(y, height))
            x1 = max(0.0, min(x + w, width))
            y1 = max(0.0, min(y + h, height))
            w, h = x1 - x0, y1 - y0
            if drop_degenerate and (w <= 0 or h <= 0):
                dropped.append(f"{rec.filename}:cls{a.class_id}")
                continue
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": a.class_id + CATEGORY_ID_OFFSET,
                    "bbox": [round(x0, 2), round(y0, 2), round(w, 2), round(h, 2)],
                    "area": round(w * h, 2),
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )
            ann_id += 1

    if dropped:
        print(
            f"[AFO/{ds.split}] UWAGA: pominieto {len(dropped)} ramek o zerowej powierzchni: "
            f"{dropped[:10]}{' ...' if len(dropped) > 10 else ''}"
        )

    notes = [
        f"split={ds.split}",
        f"categories_dir={ds.categories}",
        f"clip={ds.clip}",
        f"clipped_coords={ds.n_clipped}",
        f"dropped_degenerate={len(dropped)}",
    ]
    if ds.missing_images:
        notes.append(f"missing_images={ds.missing_images}")
    if ds.missing_labels:
        notes.append(f"missing_labels={ds.missing_labels}")

    return {
        "info": {
            "description": "AFO — Aerial dataset of Floating Objects",
            "url": "https://huggingface.co/datasets/jgasienica/afo-dataset",
            "version": "1.0",
            "year": 2021,
            "contributor": "Gasienica-Jozkowy, Knapik, Cyganek",
            "doi": "10.3233/ICA-210649",
            "notes": "; ".join(notes),
        },
        "licenses": [
            {
                "id": 1,
                "name": "CC BY-NC-SA 3.0",
                "url": "https://creativecommons.org/licenses/by-nc-sa/3.0/",
            }
        ],
        "images": images,
        "annotations": annotations,
        "categories": _categories(ds.class_names),
    }


def export_coco(
    split: str,
    out_path: Path | str | None = None,
    root: Path | str | None = None,
    *,
    clip: bool = True,
    categories: str = "6categories",
    progress: bool = True,
) -> Path:
    """Eksportuje jeden podzial do pliku COCO JSON i zwraca jego sciezke.

    Domyslne miejsce zapisu: ``<cache>/coco/<split>.json`` — w cache, nie w repo.
    """
    ds = AFODataset(split=split, root=root, clip=clip, categories=categories)
    coco = build_coco(ds, progress=progress)

    if out_path is None:
        out_path = ds.root / "coco" / f"{ds.split}.json"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(coco), encoding="utf-8")

    print(
        f"[AFO/{ds.split}] zapisano {out_path} — "
        f"{len(coco['images'])} obrazow, {len(coco['annotations'])} obiektow"
    )
    return out_path
