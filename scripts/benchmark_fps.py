#!/usr/bin/env python
"""Pomiar FPS jako OSOBNY przebieg, po zakonczeniu wszystkich treningow.

Dlaczego osobno, a nie na koncu treningu: GPU laptopa po godzinach pracy jest
gorace i obniza zegary. Te same wagi dawaly 43,8 FPS tuz po treningu i 76,2 FPS
na zimnej karcie — czyli liczba mierzona zaraz po treningu opisuje stan cieplny
maszyny, nie model. Ten skrypt czeka na wychlodzenie przed kazdym modelem i
zapisuje temperature oraz zegar obok wyniku.

Przed uruchomieniem: zasilacz w gniazdku, profil wysokiej wydajnosci, zamkniete
inne obciazenia GPU.

    python scripts/benchmark_fps.py --all          # uzupelnij wszystkie biegi
    python scripts/benchmark_fps.py --all --only yolo11n_640
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from afo.download import dataset_root
from afo.fps import COOLDOWN_TEMP_C, DEFAULT_REPEATS, gpu_state, measure_fps


def weights_for(run: Path) -> Path | None:
    for cand in (run / "best.pt", run.parent / "_runs" / run.name / "weights" / "best.pt"):
        if cand.is_file():
            return cand
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("weights", nargs="?", default=None, help="pojedyncze wagi .pt")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--all", action="store_true", help="wszystkie biegi w results/")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    ap.add_argument("--cooldown-temp", type=float, default=COOLDOWN_TEMP_C)
    ap.add_argument("--no-cooldown", action="store_true")
    ap.add_argument("--out", default=None, help="dla pojedynczych wag: zapisz JSON")
    args = ap.parse_args()

    st = gpu_state()
    print(f"GPU na starcie: {st['temp_c']} °C, zegar {st['clock_sm_mhz']} MHz\n")

    if not args.all:
        if not args.weights or not args.imgsz:
            raise SystemExit("Podaj wagi i --imgsz, albo uzyj --all")
        r = measure_fps(args.weights, args.imgsz, repeats=args.repeats,
                        cooldown=not args.no_cooldown, cooldown_temp_c=args.cooldown_temp)
        if args.out:
            Path(args.out).write_text(json.dumps(r, indent=2), encoding="utf-8")
            print(f"Zapisano {Path(args.out).resolve()}")
        return

    results = Path(args.results_dir)
    runs = sorted(p.parent for p in results.glob("*/metrics.json"))
    if args.only:
        runs = [r for r in runs if r.name in args.only]
    if not runs:
        raise SystemExit("Brak biegow.")

    root = dataset_root()
    updated, skipped = [], []
    for run in runs:
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        if metrics.get("framework") != "ultralytics":
            print(f"[fps] {run.name}: framework {metrics.get('framework')} — "
                  f"pomijam (mierzony wlasnym torem)")
            skipped.append(run.name)
            continue
        w = weights_for(run)
        if w is None:
            print(f"[fps] {run.name}: brak best.pt — POMIJAM")
            skipped.append(run.name)
            continue

        suffix = "_2cat" if metrics.get("categories") == "2categories" else ""
        images = root / f"yolo1280{suffix}" / "images" / "test"
        print(f"\n=== {run.name} ===")
        metrics["speed"] = measure_fps(
            str(w), metrics["imgsz"], images_dir=images, repeats=args.repeats,
            cooldown=not args.no_cooldown, cooldown_temp_c=args.cooldown_temp,
        )
        metrics["speed"]["measured_in"] = "osobny przebieg po treningach, po cool-downie"
        (run / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        updated.append(run.name)

    print(f"\nzaktualizowane ({len(updated)}): {', '.join(updated) or '—'}")
    if skipped:
        print(f"pominiete ({len(skipped)}): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
