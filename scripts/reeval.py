#!/usr/bin/env python
"""Ponowna ewaluacja gotowych biegow z zapisanych ``predictions_test.json``.

Po zmianie w ewaluatorze (np. definicji kubelkow rozmiaru) wyniki trzeba
przeliczyc — ale NIE trzeba uczyc modeli od nowa, bo predykcje sa zapisane.
Pola spoza ewaluacji (``speed``, ``params_M``, ``train``, ``weights_hf``) sa
zachowane bez zmian.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from afo.download import dataset_root
from afo.evaluate import evaluate, evaluate_two_category

#: pola, ktore nalezą do biegu, a nie do ewaluacji — przenoszone bez zmian
KEEP = ("speed", "params_M", "gflops", "model", "framework", "train", "weights_hf")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--only", nargs="*", default=None, help="tylko te katalogi biegow")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = dataset_root()
    results = Path(args.results_dir)
    runs = sorted(p.parent for p in results.glob("*/metrics.json"))
    if args.only:
        runs = [r for r in runs if r.name in args.only]
    if not runs:
        raise SystemExit("Brak biegow do przeliczenia.")

    for run in runs:
        old = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        pred = run / "predictions_test.json"
        if not pred.is_file():
            print(f"[reeval] {run.name}: brak predictions_test.json — POMIJAM")
            continue

        cats = old.get("categories", "6categories")
        two_cat = cats == "2categories"
        gt = root / ("coco2cat" if two_cat else "coco") / "test.json"
        imgsz = old.get("imgsz")

        print(f"\n[reeval] {run.name} ({cats}, imgsz={imgsz})")
        new = evaluate(gt, pred, imgsz=imgsz, verbose=False)
        new["categories"] = cats
        if not two_cat:
            new["two_category"] = evaluate_two_category(gt, pred, imgsz=imgsz, verbose=False)
        for k in KEEP:
            if k in old:
                new[k] = old[k]

        before = old.get("afo_buckets", {})
        after = new.get("afo_buckets", {})
        for b in after:
            o = before.get(b, {})
            print(
                f"    {b:7s} AP50-95 {o.get('AP50_95', float('nan')):.4f} -> "
                f"{after[b]['AP50_95']:.4f}   n {o.get('n_gt', 0):5d} -> {after[b]['n_gt']:5d}"
            )
        op = new.get("operating_point", {})
        if op.get("threshold") == op.get("threshold"):  # nie-NaN
            print(
                f"    punkt pracy: prog {op['threshold']:.2f}, F1 {op['f1']:.3f}, "
                f"FP/obraz {op['fp_per_image']:.2f}"
            )
        assert abs(new["mAP50_95"] - old["mAP50_95"]) < 1e-9, (
            f"{run.name}: mAP50-95 sie zmienilo ({old['mAP50_95']} -> {new['mAP50_95']}), "
            f"a zmiana dotyczyla tylko kubelkow — przerwane"
        )

        if not args.dry_run:
            (run / "metrics.json").write_text(json.dumps(new, indent=2), encoding="utf-8")

    print(f"\n[reeval] {'(dry-run) ' if args.dry_run else ''}przeliczono {len(runs)} biegow")


if __name__ == "__main__":
    main()
