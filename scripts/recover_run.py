#!/usr/bin/env python
"""Domyka przerwane biegi z zachowanych wag — zamiast uczyc je od nowa.

Po co to istnieje: bieg przerwany po N epokach ma na dysku ``best.pt``
(Ultralytics) albo ``checkpoint_best_ema.pth`` (RF-DETR) i ten plik w zupelnosci
wystarczy, zeby policzyc wszystkie metryki — trening nie musi sie powtarzac.
Raz skasowalem takie wagi z 88. epoki "zeby nie mylily kolejki" i wyrzucilem
4,7 h liczenia; ten skrypt sprawia, ze wlasciwa reakcja jest jedna komenda,
a nie przypomnienie sobie o fladze ``--post-only``.

    python scripts/recover_run.py            # pokaz, co da sie odzyskac
    python scripts/recover_run.py --run-all  # domknij wszystkie
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

#: nazwa pliku wag -> framework, po ktorym poznajemy czym domykac bieg
CHECKPOINTS = {
    "weights/best.pt": "ultralytics",
    "checkpoint_best_ema.pth": "rfdetr",
    "checkpoint_best_total.pth": "rfdetr",
    "checkpoint_best_regular.pth": "rfdetr",
}


def find_recoverable(results_dir: Path) -> list[tuple[str, str, Path]]:
    """Biegi, ktore maja wagi, ale nie maja metrics.json."""
    out = []
    runs_dir = results_dir / "_runs"
    if not runs_dir.is_dir():
        return out
    for run in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        if (results_dir / run.name / "metrics.json").is_file():
            continue
        for rel, fw in CHECKPOINTS.items():
            ckpt = run / rel
            if ckpt.is_file():
                out.append((run.name, fw, ckpt))
                break
    return out


def parse_run_name(name: str) -> tuple[str, int, str]:
    """``yolov8m_1280_2cat`` -> (model, imgsz, categories)."""
    parts = name.split("_")
    cats = "2categories" if parts[-1] == "2cat" else "6categories"
    if cats == "2categories":
        parts = parts[:-1]
    imgsz = int(parts[-1])
    return "_".join(parts[:-1]), imgsz, cats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--run-all", action="store_true", help="domknij, nie tylko pokaz")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    results = Path(args.results_dir)
    found = find_recoverable(results)
    if args.only:
        found = [f for f in found if f[0] in args.only]
    if not found:
        print("Brak biegow do odzyskania — kazdy katalog w results/_runs ma juz metrics.json.")
        return

    print(f"Do odzyskania ({len(found)}):")
    for name, fw, ckpt in found:
        mb = ckpt.stat().st_size / 1e6
        print(f"  {name:28s} {fw:12s} {ckpt.name} ({mb:.0f} MB)")
    if not args.run_all:
        print("\nNic nie zrobione. Dodaj --run-all, zeby je domknac.")
        print("NIGDY nie kasuj tych katalogow — wagi to jedyny kosztowny artefakt biegu.")
        return

    failed = []
    for name, fw, _ in found:
        model, imgsz, cats = parse_run_name(name)
        if fw == "rfdetr":
            cmd = [sys.executable, "-u", "-W", "ignore", "scripts/train_rfdetr.py",
                   "--size", model.split("-", 1)[1]]
        else:
            cmd = [sys.executable, "-u", "-W", "ignore", "scripts/train_ultralytics.py",
                   "--model", model]
        cmd += ["--imgsz", str(imgsz), "--categories", cats, "--post-only"]
        print(f"\n{'='*70}\n[recover] {name}\n{'='*70}", flush=True)
        if subprocess.run(cmd).returncode != 0:
            print(f"[recover] !!! {name} nie domkniety")
            failed.append(name)

    print(f"\n[recover] gotowe: {len(found) - len(failed)}/{len(found)}")
    if failed:
        print(f"[recover] nieudane: {', '.join(failed)}")


if __name__ == "__main__":
    main()
