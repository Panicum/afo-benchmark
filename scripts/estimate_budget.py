#!/usr/bin/env python
"""Oszacowanie wall-clock calej macierzy fazy B z pomiarow kalibracyjnych.

Czyta ``calibration.json`` (z ``calibrate_budget.py``) i ekstrapoluje na modele,
ktorych nie mierzono. Kazdy wiersz jest oznaczony jako ZMIERZONY albo SZACOWANY —
bez tego tabela wygladalaby na jednolicie wiarygodna, a nie jest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Mnozniki wzgledem modelu odniesienia (= 1,00) w tej samej rozdzielczosci.
# Przy 640 trening YOLO jest ograniczony dekodowaniem JPEG-ow, nie GPU, wiec
# rozmiar modelu prawie nie zmienia czasu epoki — zmierzone 11n 77,7 s vs
# 11m 87,7 s przy ~7x roznicy FLOP-ow, czyli +13% na siedmiokrotny model.
# RF-DETR jest ograniczony GPU i skaluje sie normalnie z rozmiarem.
FAMILY_FACTOR = {
    "yolov8n": 1.00, "yolov8s": 1.05, "yolov8m": 1.13,
    "yolo11n": 1.00, "yolo11s": 1.05, "yolo11m": 1.13,
    "rtdetr-l": 1.90,
    # Wyprowadzone z pomiarow przy odniesieniu 77,6 s/epoke @640:
    #   rfdetr-nano  154,5 s (batch 8,  peak  8,1 GB) -> 1,99
    #   rfdetr-base  237,7 s (batch 8,  peak 11,8 GB) -> 3,06
    #     ten sam model przy batchu 4 to 437,3 s, a przy 16 — 2812 s (thrashing).
    #     Batch 8 jest najwiekszy mieszczacy sie w limicie i zarazem najszybszy.
    #   rfdetr-small — brak pomiaru, srednia geometryczna nano i base
    "rfdetr-nano": 1.99, "rfdetr-small": 2.47, "rfdetr-base": 3.06,
}

#: ktore rodziny maja mnozniki zgadywane, a nie wyprowadzone z pomiaru
GUESSED_FAMILIES = ("rtdetr",)

#: macierz z BENCHMARK.md: (model, imgsz)
MATRIX: list[tuple[str, int]] = [
    ("yolov8n", 640), ("yolov8s", 640), ("yolov8m", 640),
    ("yolo11n", 640), ("yolo11s", 640), ("yolo11m", 640),
    ("rfdetr-nano", 640), ("rfdetr-small", 640), ("rfdetr-base", 640),
    ("rtdetr-l", 640),
    ("yolo11m", 1280), ("rfdetr-base", 1280),
]


def baseline(cal: list[dict], imgsz: int) -> tuple[float, str]:
    """Czas epoki modelu odniesienia (mnoznik 1,0) dla danej rozdzielczosci.

    Liczony z modelu YOLO, zeby nie mieszac dwoch rodzin w jednej skali.
    """
    rows = [r for r in cal if r["imgsz"] == imgsz and r["model"].startswith("yolo")]
    if not rows:
        rows = [r for r in cal if r["imgsz"] == imgsz]
    if not rows:
        return (0.0, "brak")
    best = min(rows, key=lambda r: FAMILY_FACTOR.get(r["model"], 1.0))
    f = FAMILY_FACTOR.get(best["model"], 1.0)
    return best["s_per_epoch"] / f, best["model"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calibration", required=True, nargs="+", help="jeden lub wiecej JSON-ow")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument(
        "--early-stop-frac", type=float, default=0.75,
        help="jaki ulamek budzetu epok realnie sie wykona przy patience 20",
    )
    ap.add_argument("--eval-min", type=float, default=12.0, help="minut na ewaluacje+FPS na model")
    args = ap.parse_args()

    cal = []
    for p in args.calibration:
        cal.extend(json.loads(Path(p).read_text(encoding="utf-8")))

    # Odrzuc pomiary zrobione w thrashingu. Na Windowsie sterownik NVIDIA zamiast
    # rzucic OOM przenosi nadmiar do RAM-u, wiec bieg "udaje sie", tylko trwa
    # wielokrotnie dluzej. Taki wiersz wpuszczony do budzetu zawyzylby go kilkakrotnie.
    try:
        import torch

        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    except Exception:
        vram_gb = float("inf")

    good, rejected = [], []
    for r in cal:
        peak = r.get("peak_vram_GB")
        if r.get("invalid") or (peak is not None and peak > vram_gb):
            rejected.append(r)
        else:
            good.append(r)
    if rejected:
        print("ODRZUCONE pomiary (peak powyzej fizycznego VRAM = thrashing po PCIe):")
        for r in rejected:
            print(
                f"  {r['model']}@{r['imgsz']} batch {r.get('batch')}: "
                f"{r['s_per_epoch']} s/epoke, peak {r.get('peak_vram_GB')} GB"
            )
        print()
    cal = good

    # przy kilku pomiarach tego samego modelu bierz najszybszy poprawny
    measured: dict[tuple[str, int], dict] = {}
    for r in cal:
        k = (r["model"], r["imgsz"])
        if k not in measured or r["s_per_epoch"] < measured[k]["s_per_epoch"]:
            measured[k] = r

    base640, src640 = baseline(cal, 640)
    base1280, src1280 = baseline(cal, 1280)
    print(f"Odniesienie @640:  {base640:.1f} s/epoke (z {src640})")
    print(f"Odniesienie @1280: {base1280:.1f} s/epoke (z {src1280})\n")

    rows = []
    tot_full = tot_es = 0.0
    for model, imgsz in MATRIX:
        key = (model, imgsz)
        if key in measured:
            spe = measured[key]["s_per_epoch"]
            tag = "ZMIERZONY"
        else:
            base = base640 if imgsz == 640 else base1280
            if not base:
                continue
            spe = base * FAMILY_FACTOR.get(model, 1.0)
            tag = "szacowany"
        h_full = spe * args.epochs / 3600
        h_es = h_full * args.early_stop_frac
        tot_full += h_full
        tot_es += h_es
        rows.append([model, str(imgsz), tag, f"{spe:.0f}", f"{h_full:.1f}", f"{h_es:.1f}"])

    hdr = ["model", "imgsz", "zrodlo", "s/epoke", f"h @{args.epochs}ep", "h z early stop"]
    w = [max(len(hdr[i]), *(len(r[i]) for r in rows)) for i in range(len(hdr))]
    print("| " + " | ".join(h.ljust(w[i]) for i, h in enumerate(hdr)) + " |")
    print("|" + "|".join("-" * (x + 2) for x in w) + "|")
    for r in rows:
        print("| " + " | ".join(r[i].ljust(w[i]) for i in range(len(r))) + " |")

    eval_h = len(rows) * args.eval_min / 60
    print(f"\nTrening, pelne {args.epochs} epok:      {tot_full:5.1f} h")
    print(f"Trening, realnie z early stop:   {tot_es:5.1f} h  (zalozenie: "
          f"{args.early_stop_frac:.0%} budzetu epok)")
    print(f"Ewaluacja + pomiar FPS:          {eval_h:5.1f} h  ({len(rows)} modeli x "
          f"{args.eval_min:.0f} min)")
    print(f"RAZEM:                           {tot_es + eval_h:5.1f} h")
    n_meas = sum(1 for r in rows if r[2] == "ZMIERZONY")
    guessed = [r[0] for r in rows if any(g in r[0] for g in GUESSED_FAMILIES)]
    print(f"\nZMIERZONYCH wierszy: {n_meas}/{len(rows)}.")
    print(
        "Reszta to ekstrapolacja z pomiaru tej samej rodziny — poza "
        + (", ".join(sorted(set(guessed))) if guessed else "(brak)")
        + ", gdzie mnoznik jest zgadywany i moze sie mylic nawet 2x."
    )


if __name__ == "__main__":
    main()
