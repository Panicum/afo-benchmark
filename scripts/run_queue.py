#!/usr/bin/env python
"""Sekwencyjny runner macierzy benchmarku — jeden bieg naraz.

Pomija to, co juz ma ``metrics.json``, wiec po przerwaniu wystarczy uruchomic
ponownie. Po kazdym biegu odswieza leaderboard, zeby czesciowy postep byl
widoczny bez czekania na koniec. Blad jednego modelu nie zatrzymuje reszty —
jest raportowany i kolejka idzie dalej.

    python scripts/run_queue.py --plan               # pokaz, co by zrobil
    python scripts/run_queue.py                      # uruchom
    python scripts/run_queue.py --max-hours 14       # zatrzymaj sie przed limitem
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PY = sys.executable

#: Konsola Windows bywa cp1252 i wywraca sie na polskich znakach w logach
#: podprocesow. Wymuszamy UTF-8 dla wszystkiego, co uruchamiamy.
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


@dataclass
class Job:
    model: str
    imgsz: int
    categories: str
    framework: str = "ultralytics"
    #: szacowany czas w godzinach — tylko do planowania, nie do raportowania
    est_h: float = 1.0

    @property
    def run_name(self) -> str:
        suffix = {"6categories": "", "2categories": "_2cat"}[self.categories]
        return f"{self.model}_{self.imgsz}{suffix}"

    def cmd(self) -> list[str]:
        if self.framework == "rfdetr":
            size = self.model.split("-", 1)[1]
            return [PY, "-u", "-W", "ignore", "scripts/train_rfdetr.py",
                    "--size", size, "--imgsz", str(self.imgsz),
                    "--categories", self.categories]
        return [PY, "-u", "-W", "ignore", "scripts/train_ultralytics.py",
                "--model", self.model, "--imgsz", str(self.imgsz),
                "--categories", self.categories]


# Kolejnosc = priorytet. Najpierw domkniecie obu gallezi YOLO przy 640 (rdzen
# benchmarku), potem 1280, na koncu RF-DETR. Czasy z ukonczonych biegow:
# n ~0,6 h, s ~1,3 h, m ~2,2 h przy 640.
YOLO_640 = ["yolov8n", "yolov8s", "yolov8m", "yolo11n", "yolo11s", "yolo11m"]
EST_640 = {"yolov8n": 0.6, "yolo11n": 0.7, "yolov8s": 1.3, "yolo11s": 1.3,
           "yolov8m": 2.2, "yolo11m": 2.2}


def default_plan() -> list[Job]:
    jobs: list[Job] = []
    for cats in ("6categories", "2categories"):
        for m in YOLO_640:
            jobs.append(Job(m, 640, cats, est_h=EST_640[m]))
    # 1280 dla najlepszego YOLO — model wybierany dynamicznie przed startem
    jobs.append(Job("__best_yolo__", 1280, "6categories", est_h=3.5))
    jobs.append(Job("__best_yolo__", 1280, "2categories", est_h=3.5))
    for size, est in (("nano", 2.5), ("small", 3.5), ("base", 5.0)):
        jobs.append(Job(f"rfdetr-{size}", 640, "6categories", "rfdetr", est))
    return jobs


def best_yolo(results_dir: Path, categories: str) -> str | None:
    """Najlepszy ukonczony model YOLO @640 w danej galezi, wg test mAP50-95."""
    best, best_v = None, -1.0
    for p in results_dir.glob("*/metrics.json"):
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if m.get("imgsz") != 640 or m.get("categories", "6categories") != categories:
            continue
        if m.get("framework") != "ultralytics":
            continue
        v = m.get("mAP50_95")
        if v is not None and v > best_v:
            best, best_v = m.get("model"), v
    return best


def done(results_dir: Path, run_name: str) -> bool:
    return (results_dir / run_name / "metrics.json").is_file()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--plan", action="store_true", help="tylko pokaz plan")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="nie zaczynaj nowego biegu po przekroczeniu tego czasu")
    ap.add_argument("--only", nargs="*", default=None,
                    help="uruchom tylko te run_name (po pominieciu ukonczonych)")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    jobs = default_plan()
    t_start = time.perf_counter()

    print(f"{'bieg':28s} {'framework':12s} {'szac. h':>8s}  status")
    print("-" * 70)
    todo: list[Job] = []
    for j in jobs:
        name = j.run_name
        if "__best_yolo__" in name:
            status = "zaleznie od wynikow @640"
        elif done(results_dir, name):
            status = "GOTOWE — pomijam"
        elif args.only and name not in args.only:
            status = "poza --only"
        else:
            status = "do zrobienia"
            todo.append(j)
        print(f"{name:28s} {j.framework:12s} {j.est_h:8.1f}  {status}")
    print("-" * 70)
    print(f"do zrobienia: {len(todo)} biegow, szacunkowo "
          f"{sum(j.est_h for j in todo):.1f} h (+ 2 biegi @1280 zalezne od wynikow)")
    if args.plan:
        return

    failed: list[str] = []
    for i, j in enumerate(jobs, 1):
        elapsed = (time.perf_counter() - t_start) / 3600
        if args.max_hours and elapsed + j.est_h > args.max_hours:
            # Nie zaczynamy biegu, ktory wedlug szacunku przekroczylby limit.
            # Szacunek moze byc nietrafiony, wiec to zabezpieczenie miekkie —
            # ale lepsze niz obudzenie sie z biegiem urwanym w polowie.
            print(
                f"\n[queue] {j.run_name} ({j.est_h:.1f} h szac.) nie zmiesci sie w "
                f"limicie {args.max_hours} h — minelo {elapsed:.1f} h. Szukam krotszego."
            )
            continue

        if j.model == "__best_yolo__":
            pick = best_yolo(results_dir, j.categories)
            if not pick:
                print(f"\n[queue] brak ukonczonych YOLO @640 dla {j.categories} — "
                      f"pomijam bieg @1280")
                continue
            j = Job(pick, j.imgsz, j.categories, j.framework, j.est_h)
            print(f"\n[queue] najlepszy YOLO @640 ({j.categories}) = {pick}")

        name = j.run_name
        if done(results_dir, name):
            continue
        if args.only and name not in args.only:
            continue

        print(f"\n{'='*70}\n[queue {i}/{len(jobs)}] {name}  (po {elapsed:.1f} h)\n{'='*70}",
              flush=True)
        t0 = time.perf_counter()
        r = subprocess.run(j.cmd(), env=ENV)
        dt = (time.perf_counter() - t0) / 3600
        if r.returncode != 0 or not done(results_dir, name):
            print(f"[queue] !!! {name} nie zakonczyl sie poprawnie (kod {r.returncode}, "
                  f"{dt:.2f} h) — ide dalej")
            failed.append(name)
            continue
        print(f"[queue] {name} gotowe w {dt:.2f} h")
        subprocess.run([PY, "scripts/make_leaderboard.py"], env=ENV)

    total = (time.perf_counter() - t_start) / 3600
    print(f"\n{'='*70}\n[queue] koniec po {total:.1f} h")
    ok = sorted(p.parent.name for p in results_dir.glob("*/metrics.json"))
    print(f"[queue] ukonczone ({len(ok)}): {', '.join(ok)}")
    if failed:
        print(f"[queue] NIEUDANE ({len(failed)}): {', '.join(failed)}")


if __name__ == "__main__":
    main()
