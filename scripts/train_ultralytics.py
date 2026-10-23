#!/usr/bin/env python
"""Trening modelu Ultralytics (YOLOv8 / YOLO11 / RT-DETR) wg protokolu z BENCHMARK.md.

Uczy na train, wybiera po val mAP50-95, raportuje na tescie przez wspolny
``afo/evaluate.py``. Zapisuje komplet artefaktow do ``results/<model>_<imgsz>/``.

Batch dobierany sonda przeciw FIZYCZNEMU VRAM, nie przeciw wyjatkowi OOM —
patrz ``afo/vram.py``.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

from afo.download import dataset_root
from afo.vram import DEFAULT_LIMIT_FRAC, check_peak, reset_peak, sysmem_fallback_active

BATCH_CANDIDATES = {640: [8, 16, 32, 64], 1280: [2, 4, 8, 16]}


def probe_batch(model_name: str, imgsz: int, data: str, limit_frac: float) -> tuple[int, float]:
    """Najwiekszy batch mieszczacy sie w limicie VRAM. Zwraca (batch, peak_GB)."""
    from ultralytics import YOLO

    best, best_peak = BATCH_CANDIDATES[imgsz][0], 0.0
    for b in BATCH_CANDIDATES[imgsz]:
        reset_peak()
        try:
            YOLO(f"{model_name}.pt").train(
                data=data, epochs=1, imgsz=imgsz, batch=b, device=0, workers=4,
                fraction=0.03, val=False, plots=False, save=False, verbose=False,
                project=str((Path.home() / ".cache" / "afo" / "_probe").resolve()),
                name=f"{model_name}_{imgsz}_{b}", exist_ok=True, seed=0,
            )
        except (RuntimeError, MemoryError) as e:
            if "out of memory" not in str(e).lower():
                raise
            print(f"  batch {b:4d}: OOM")
            break
        c = check_peak(limit_frac)
        print(f"  batch {b:4d}: {c}")
        if not c.fits:
            break
        best, best_peak = b, c.peak_GB
    return best, best_peak


def _plain(obj):
    """Sprowadza strukture do typow, ktore zapisze ``yaml.safe_dump``.

    Powod: ``torch.__version__`` to ``TorchVersion`` (podklasa str), a args.yaml
    Ultralytics potrafi zawierac obiekty Path. safe_dump wysypuje sie na nich
    PO zakonczonym treningu, kasujac godzine pracy na zapisie konfiguracji.
    """
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if obj is None or type(obj) in (bool, int, float, str):
        return obj
    if isinstance(obj, bool):
        return bool(obj)
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return float(obj)
    return str(obj)


def git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="np. yolo11n, yolov8s, rtdetr-l")
    ap.add_argument("--imgsz", type=int, default=640, choices=[640, 1280])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=None, help="pomin sonde i wymus batch")
    ap.add_argument("--limit-frac", type=float, default=DEFAULT_LIMIT_FRAC)
    ap.add_argument("--data", default=None, help="afo.yaml (domyslnie <cache>/yolo1280/afo.yaml)")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--post-only", action="store_true",
        help="pomin trening, policz predykcje/metryki/FPS z istniejacego best.pt",
    )
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise SystemExit("Brak CUDA.")
    gpu = torch.cuda.get_device_name(0)

    data = args.data or str(dataset_root() / "yolo1280" / "afo.yaml")
    run_name = f"{args.model}_{args.imgsz}"
    # Sciezki bezwzgledne: Ultralytics traktuje wzgledny `project` jako podkatalog
    # swojego runs_dir, wiec "results/_runs" ladowalo w runs/detect/results/_runs.
    results_root = Path(args.results_dir).resolve()
    runs_root = results_root / "_runs"
    out_dir = results_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.txt"

    class Tee:
        def __init__(self, path):
            self.f = open(path, "w", encoding="utf-8")
        def write(self, s):
            sys.__stdout__.write(s); self.f.write(s); self.f.flush()
        def flush(self):
            sys.__stdout__.flush(); self.f.flush()

    sys.stdout = Tee(log_path)

    print(f"=== {run_name} ===")
    print(f"GPU: {gpu}")
    fb = sysmem_fallback_active()
    print(f"sysmem fallback sterownika: {fb} "
          f"({'batch MUSI byc ograniczany pomiarem' if fb else 'OOM zadziala normalnie'})")
    print(f"dane: {data}\n")

    if args.batch:
        batch, peak = args.batch, float("nan")
        print(f"batch wymuszony: {batch}")
    else:
        print("sonda batcha:")
        batch, peak = probe_batch(args.model, args.imgsz, data, args.limit_frac)
        print(f"wybrany batch: {batch} (peak {peak:.1f} GB)\n")

    if args.post_only:
        # Nie zmyslamy czasu treningu: czytamy go z results.csv tego biegu.
        # Zerowe "hours" w metrics.json wygladaloby jak darmowy trening.
        train_s = None
        csv = runs_root / run_name / "results.csv"
        if csv.is_file():
            rows = [r for r in csv.read_text(encoding="utf-8").splitlines() if r.strip()]
            if len(rows) > 1:
                train_s = float(rows[-1].split(",")[1])
        c = None
        print(f"--post-only: trening pominiety; czas z results.csv: "
              f"{'nieznany' if train_s is None else f'{train_s/3600:.2f} h'}")
    else:
        reset_peak()
        t0 = time.perf_counter()
        model = YOLO(f"{args.model}.pt")
        model.train(
            data=data, epochs=args.epochs, imgsz=args.imgsz, batch=batch, device=0,
            workers=args.workers, seed=args.seed, patience=args.patience,
            val=True, plots=True, save=True, exist_ok=True, deterministic=True,
            project=str(runs_root), name=run_name,
        )
        train_s = time.perf_counter() - t0
        c = check_peak(args.limit_frac)
        print(f"\ntrening: {train_s/3600:.2f} h | {c}")
        if not c.fits:
            print("UWAGA: szczyt przekroczyl limit — czesc biegu mogla thrashowac po PCIe.")

    run_dir = runs_root / run_name
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise SystemExit(f"Brak {best}")

    # pelna, rozwiazana konfiguracja treningu — zeby "domyslna augmentacja" byla sprawdzalna
    args_yaml = run_dir / "args.yaml"
    cfg = yaml.safe_load(args_yaml.read_text(encoding="utf-8")) if args_yaml.is_file() else {}
    cfg.update({
        "_afo_protocol": "BENCHMARK.md",
        "_gpu": gpu,
        "_git_rev": git_rev(),
        "_python": platform.python_version(),
        "_torch": str(torch.__version__),
        "_train_hours": None if train_s is None else round(train_s / 3600, 3),
        "_peak_vram_GB": None if c is None else round(c.peak_GB, 2),
        "_command": " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
    })
    (out_dir / "train_config.yaml").write_text(
        yaml.safe_dump(_plain(cfg), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    # --- predykcje na tescie -------------------------------------------------
    print("\npredykcje na tescie...")
    from afo.predict import predict_ultralytics

    pred_path = out_dir / "predictions_test.json"
    predict_ultralytics(str(best), args.imgsz, pred_path)

    # --- ewaluacja wspolnym ewaluatorem --------------------------------------
    from afo.evaluate import evaluate

    gt = dataset_root() / "coco" / "test.json"
    metrics = evaluate(gt, pred_path, imgsz=args.imgsz)

    # --- FPS -----------------------------------------------------------------
    print("\npomiar FPS...")
    from afo.fps import measure_fps

    fps = measure_fps(str(best), args.imgsz)
    metrics["speed"] = fps

    # model.info() zwraca None w ultralytics 8.4 — liczymy sami, zeby kolumna
    # "params" w leaderboardzie nie byla pusta.
    bm = YOLO(str(best)).model
    n_params = sum(p.numel() for p in bm.parameters())
    gflops = None
    try:
        from ultralytics.utils.torch_utils import get_flops

        gflops = round(float(get_flops(bm, args.imgsz)), 1) or None
    except Exception as e:
        print(f"UWAGA: nie udalo sie policzyc GFLOPs ({type(e).__name__})")

    metrics["model"] = args.model
    metrics["framework"] = "ultralytics"
    metrics["params_M"] = round(n_params / 1e6, 2)
    metrics["gflops"] = gflops
    metrics["train"] = {
        "epochs_budget": args.epochs, "patience": args.patience, "batch": batch,
        "seed": args.seed,
        "hours": None if train_s is None else round(train_s / 3600, 3),
        "peak_vram_GB": None if c is None else round(c.peak_GB, 2),
        "gpu": str(gpu),
    }
    metrics["weights_hf"] = f"jgasienica/afo-{args.model}-{args.imgsz}"

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    shutil.copy2(best, out_dir / "best.pt")  # .gitignore trzyma to poza repo
    print(f"\nZapisano {out_dir.resolve()}")


if __name__ == "__main__":
    main()
