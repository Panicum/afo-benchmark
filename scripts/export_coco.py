#!/usr/bin/env python
"""Eksport oficjalnych podzialow AFO do COCO JSON.

Pliki trafiaja do cache (``<AFO_ROOT>/coco/{train,val,test}.json``), nie do repo.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from afo.coco import export_coco
from afo.download import dataset_root, ensure_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--root", default=None, help="katalog zbioru (domyslnie cache AFO)")
    ap.add_argument("--out-dir", default=None, help="katalog wyjsciowy (domyslnie <root>/coco)")
    ap.add_argument("--no-clip", action="store_true", help="nie przycinaj wspolrzednych do [0,1]")
    ap.add_argument("--categories", default="6categories")
    ap.add_argument("--download", action="store_true", help="pobierz zbior, jesli go nie ma")
    args = ap.parse_args()

    if args.download:
        ensure_dataset(args.root)

    root = dataset_root(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "coco"

    for split in args.splits:
        export_coco(
            split,
            out_path=out_dir / f"{split}.json",
            root=root,
            clip=not args.no_clip,
            categories=args.categories,
        )


if __name__ == "__main__":
    main()
