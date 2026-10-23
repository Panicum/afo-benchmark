#!/usr/bin/env python
"""Pomiar czasu epoki na tym GPU — podstawa oszacowania budzetu fazy B.

Nie jest to bieg benchmarkowy: uczy przez kilka epok wylacznie po to, zeby
zmierzyc sekundy/epoke i najwiekszy batch mieszczacy sie w VRAM. Wyniki ida do
JSON, z ktorego `estimate_budget.py` liczy wall-clock calej macierzy.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


#: Udzial fizycznego VRAM-u, powyzej ktorego batch uznajemy za niemieszczacy sie.
#: UWAGA: na Windowsie sterownik NVIDIA domyslnie robi fallback do pamieci
#: systemowej zamiast rzucic OutOfMemoryError — alokacja 20 GB na karcie 16 GB
#: "udaje sie" i po cichu zamienia trening w thrashing po PCIe. Dlatego progiem
#: jest zmierzone zuzycie, a nie wyjatek.
VRAM_LIMIT_FRAC = 0.85


def largest_batch(model_name: str, imgsz: int, data: str, candidates: list[int]) -> int:
    """Najwiekszy batch, ktory miesci sie w FIZYCZNYM VRAM z zapasem."""
    from ultralytics import YOLO

    total = torch.cuda.get_device_properties(0).total_memory
    budget = VRAM_LIMIT_FRAC * total
    ok = candidates[0]
    for b in candidates:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        m = None
        try:
            m = YOLO(f"{model_name}.pt")
            m.train(
                data=data, epochs=1, imgsz=imgsz, batch=b, device=0, workers=4,
                fraction=0.03, val=False, plots=False, save=False, verbose=False,
                project=str(Path.home() / ".cache" / "afo" / "_calib"),
                name=f"probe_{model_name}_{imgsz}_{b}", exist_ok=True, seed=0,
            )
            peak = torch.cuda.max_memory_allocated()
            if peak > budget:
                print(
                    f"    batch {b:4d} ZA DUZY — peak {peak/1e9:.1f} GB > "
                    f"{budget/1e9:.1f} GB ({VRAM_LIMIT_FRAC:.0%} z {total/1e9:.1f} GB)"
                )
                break
            ok = b
            print(f"    batch {b:4d} OK   (peak {peak/1e9:.1f} GB)")
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower():
                raise
            print(f"    batch {b:4d} OOM")
            break
        finally:
            del m
            torch.cuda.empty_cache()
    return ok


def time_epochs(model_name: str, imgsz: int, batch: int, data: str, epochs: int) -> dict:
    """Czas na epoke przy pelnym zbiorze treningowym."""
    from ultralytics import YOLO

    torch.cuda.empty_cache()
    m = YOLO(f"{model_name}.pt")
    t0 = time.perf_counter()
    m.train(
        data=data, epochs=epochs, imgsz=imgsz, batch=batch, device=0, workers=8,
        val=True, plots=False, save=False, verbose=False, seed=0,
        project=str(Path.home() / ".cache" / "afo" / "_calib"),
        name=f"time_{model_name}_{imgsz}", exist_ok=True,
    )
    total = time.perf_counter() - t0
    info = m.model.info(verbose=False)
    params = info[1] if isinstance(info, (tuple, list)) and len(info) > 1 else None
    gflops = info[3] if isinstance(info, (tuple, list)) and len(info) > 3 else None
    return {
        "model": model_name,
        "imgsz": imgsz,
        "batch": batch,
        "epochs_timed": epochs,
        "total_s": round(total, 1),
        "s_per_epoch": round(total / epochs, 1),
        "params_M": round(params / 1e6, 2) if params else None,
        "gflops": round(gflops, 1) if gflops else None,
        "peak_vram_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(Path.home() / ".cache" / "afo" / "yolo" / "afo.yaml"))
    ap.add_argument("--epochs", type=int, default=2, help="ile epok mierzyc")
    ap.add_argument("--out", default="calibration.json")
    ap.add_argument(
        "--configs", nargs="+", default=["yolo11n:640", "yolo11m:640", "yolo11s:1280"],
        help="model:imgsz",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("Brak CUDA — kalibracja wymaga GPU.")
    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu}\nDane: {args.data}\n")

    results = []
    for cfg in args.configs:
        name, imgsz = cfg.split(":")
        imgsz = int(imgsz)
        print(f"== {name} @ {imgsz} ==")
        cands = [8, 16, 32, 64] if imgsz >= 1280 else [16, 32, 64, 128]
        b = largest_batch(name, imgsz, args.data, cands)
        print(f"  batch = {b}; mierze {args.epochs} epok...")
        r = time_epochs(name, imgsz, b, args.data, args.epochs)
        r["gpu"] = gpu
        print(f"  -> {r['s_per_epoch']} s/epoke, {r['params_M']} M par, {r['gflops']} GFLOPs\n")
        results.append(r)
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Zapisano {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
