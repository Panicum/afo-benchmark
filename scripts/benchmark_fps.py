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
from afo.fps import (COOLDOWN_TEMP_C, DEFAULT_REPEATS, WARMUP_IMAGES, gpu_state,
                     measure_fps, time_predict_fn, wait_for_cooldown)


def weights_for(run: Path) -> Path | None:
    for cand in (run / "best.pt", run.parent / "_runs" / run.name / "weights" / "best.pt"):
        if cand.is_file():
            return cand
    return None


RFDETR_CLASS = {"nano": "RFDETRNano", "small": "RFDETRSmall", "base": "RFDETRBase"}


def measure_rfdetr(run: Path, metrics: dict, files: list[Path], repeats: int,
                   cooldown: bool, cooldown_temp_c: float) -> dict:
    """FPS dla RF-DETR — inny framework, ten sam protokol pomiaru."""
    import re
    import statistics

    import rfdetr

    ckpt = next(
        (run.parent / "_runs" / run.name / n
         for n in ("checkpoint_best_ema.pth", "checkpoint_best_total.pth")
         if (run.parent / "_runs" / run.name / n).is_file()), None)
    if ckpt is None:
        raise FileNotFoundError(f"brak checkpointu dla {run.name}")

    size = metrics["model"].split("-", 1)[1]
    cls = getattr(rfdetr, RFDETR_CLASS[size])
    imgsz = metrics["imgsz"]
    model = cls(pretrain_weights=str(ckpt), resolution=imgsz,
                num_classes=len(metrics["class_names"]))

    # ta sama niespojnosc rozdzielczosci co przy predykcji — patrz train_rfdetr
    shape = None
    override = metrics.get("inference_shape_override")
    if override:
        shape = (override["inferred_at"], override["inferred_at"])

    def predict(path: str):
        try:
            return model.predict(path, threshold=0.25,
                                 **({"shape": shape} if shape else {}))
        except ValueError as exc:
            m = re.search(r"block_size=(\d+)", str(exc))
            if not m:
                raise
            b = int(m.group(1))
            nonlocal_shape = max(b, round(imgsz / b) * b)
            raise RuntimeError(
                f"rozdzielczosc {imgsz} niepodzielna przez {b}; uzyj "
                f"inference_shape_override={nonlocal_shape}") from exc

    thermal = wait_for_cooldown(cooldown_temp_c) if cooldown else {"skipped": True}
    runs = [time_predict_fn(predict, files, WARMUP_IMAGES) for _ in range(max(1, repeats))]
    e2e = sorted(r["fps"] for r in runs)
    spread = 100 * (max(e2e) - min(e2e)) / max(statistics.median(e2e), 1e-9)
    print(f"[fps] mediana {statistics.median(e2e):.2f} FPS end-to-end "
          f"(zakres {min(e2e)}-{max(e2e)}, rozrzut {spread:.0f}%, {len(runs)} przebiegi)")
    return {
        "gpu": metrics["train"]["gpu"],
        "imgsz": imgsz, "precision": "fp16", "batch": 1,
        "warmup_images": WARMUP_IMAGES, "timed_images": runs[-1]["timed_images"],
        "repeats": len(runs),
        "end_to_end": {**{k: v for k, v in runs[-1].items() if k.endswith("_ms")},
                       "fps": round(statistics.median(e2e), 2),
                       "fps_min": min(e2e), "fps_max": max(e2e), "fps_all": e2e},
        "fps_spread_pct": round(spread, 1),
        "thermal_before": thermal, "thermal_after": gpu_state(),
        "measured_in": "osobny przebieg po treningach, po cool-downie",
    }


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
        suffix = "_2cat" if metrics.get("categories") == "2categories" else ""
        images = root / f"yolo1280{suffix}" / "images" / "test"

        if metrics.get("framework") == "rfdetr":
            files = sorted(p for p in images.iterdir()
                           if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
            print(f"\n=== {run.name} (rfdetr) ===")
            try:
                metrics["speed"] = measure_rfdetr(
                    run, metrics, files, args.repeats,
                    not args.no_cooldown, args.cooldown_temp)
            except Exception as exc:
                print(f"[fps] {run.name}: BLAD {type(exc).__name__}: {str(exc)[:160]}")
                skipped.append(run.name)
                continue
            (run / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            updated.append(run.name)
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
