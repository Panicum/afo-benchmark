"""Pomiar predkosci wnioskowania — wspolny dla wszystkich modeli benchmarku.

Batch 1, FP16, srednia po zbiorze testowym, po 50 obrazach rozgrzewkowych.
Czas mierzony po ``torch.cuda.synchronize()``, inaczej mierzy sie kolejkowanie,
a nie liczenie.

Raportowane osobno: czysty forward i pelny tor z preprocessingiem i NMS.
Roznica bywa dwukrotna, a tylko ten drugi odpowiada temu, co dziala na lodzi.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

from .download import dataset_root

WARMUP_IMAGES = 50


def _percentiles(xs: list[float]) -> dict[str, float]:
    xs = sorted(xs)
    n = len(xs)
    return {
        "mean_ms": round(1000 * statistics.fmean(xs), 3),
        "median_ms": round(1000 * xs[n // 2], 3),
        "p95_ms": round(1000 * xs[min(n - 1, int(0.95 * n))], 3),
    }


def measure_fps(
    weights: str,
    imgsz: int,
    images_dir: Path | str | None = None,
    warmup: int = WARMUP_IMAGES,
    half: bool = True,
    max_images: int | None = None,
) -> dict:
    """Mierzy FPS modelu Ultralytics na zbiorze testowym."""
    import torch
    from ultralytics import YOLO

    images_dir = Path(images_dir) if images_dir else dataset_root() / "yolo1280" / "images" / "test"
    files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if max_images:
        files = files[:max_images]
    if len(files) <= warmup:
        raise ValueError(f"Za malo obrazow ({len(files)}) na rozgrzewke {warmup}")

    model = YOLO(weights)
    gpu = torch.cuda.get_device_name(0)

    for p in files[:warmup]:
        model.predict(str(p), imgsz=imgsz, device=0, half=half, verbose=False)
    torch.cuda.synchronize()

    total, inference = [], []
    for p in files[warmup:]:
        t0 = time.perf_counter()
        r = model.predict(str(p), imgsz=imgsz, device=0, half=half, verbose=False)[0]
        torch.cuda.synchronize()
        total.append(time.perf_counter() - t0)
        s = r.speed  # ms: preprocess / inference / postprocess
        inference.append(s["inference"] / 1000.0)

    out = {
        "gpu": gpu,
        "imgsz": imgsz,
        "precision": "fp16" if half else "fp32",
        "batch": 1,
        "warmup_images": warmup,
        "timed_images": len(total),
        "end_to_end": {**_percentiles(total), "fps": round(1.0 / statistics.fmean(total), 2)},
        "forward_only": {
            **_percentiles(inference),
            "fps": round(1.0 / statistics.fmean(inference), 2),
        },
        "note": (
            f"{gpu} — czesc LAPTOPOWA (76 SM wobec 128 w wersji desktopowej), "
            f"podatna na throttling termiczny. FPS porownywac miedzy wierszami tej "
            f"tabeli, nie z wynikami z desktopowych kart."
        ),
    }
    print(
        f"[fps] {out['end_to_end']['fps']} FPS end-to-end, "
        f"{out['forward_only']['fps']} FPS sam forward "
        f"({out['timed_images']} obrazow, {imgsz}px, fp16)"
    )
    return out


def main() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("weights")
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--images-dir", default=None)
    ap.add_argument("--warmup", type=int, default=WARMUP_IMAGES)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    r = measure_fps(a.weights, a.imgsz, a.images_dir, a.warmup, half=not a.fp32)
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=2), encoding="utf-8")
        print(f"Zapisano {Path(a.out).resolve()}")


if __name__ == "__main__":
    main()
