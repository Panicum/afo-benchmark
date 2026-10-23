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


#: Ile razy powtorzyc caly przebieg. Pojedynczy pomiar na laptopie zalezy od
#: stanu cieplnego: te same wagi dawaly 43,8 FPS tuz po 45-minutowym treningu
#: i 76,2 FPS na zimno (rozrzut 74%). Powtorzenia pod rzad trzymaja sie w 3%,
#: wiec raportujemy MEDIANE z kilku przebiegow plus zakres.
DEFAULT_REPEATS = 4


def measure_fps(
    weights: str,
    imgsz: int,
    images_dir: Path | str | None = None,
    warmup: int = WARMUP_IMAGES,
    half: bool = True,
    max_images: int | None = None,
    repeats: int = DEFAULT_REPEATS,
) -> dict:
    """Mierzy FPS modelu Ultralytics na zbiorze testowym (mediana z ``repeats``)."""
    n = max(1, repeats)
    runs = [_measure_once(weights, imgsz, images_dir, warmup, half, max_images) for _ in range(n)]

    # Pierwszy przebieg idzie na zimnym GPU i systematycznie zanizza wynik
    # (zmierzone: 51,7 wobec 71,1 FPS w kolejnych). Przy >=3 powtorzeniach
    # traktujemy go jako rozgrzewke cieplna i odrzucamy.
    dropped = None
    if n >= 3:
        dropped = runs[0]["end_to_end"]["fps"]
        runs = runs[1:]

    e2e = sorted(r["end_to_end"]["fps"] for r in runs)
    fwd = sorted(r["forward_only"]["fps"] for r in runs)
    out = dict(runs[-1])
    out["repeats"] = len(runs)
    out["warmup_run_discarded_fps"] = dropped
    out["end_to_end"] = {
        **runs[0]["end_to_end"],
        "fps": round(statistics.median(e2e), 2),
        "fps_min": min(e2e),
        "fps_max": max(e2e),
        "fps_all": e2e,
    }
    out["forward_only"] = {
        **runs[0]["forward_only"],
        "fps": round(statistics.median(fwd), 2),
        "fps_min": min(fwd),
        "fps_max": max(fwd),
        "fps_all": fwd,
    }
    spread = 100 * (max(e2e) - min(e2e)) / max(statistics.median(e2e), 1e-9)
    out["fps_spread_pct"] = round(spread, 1)
    extra = f", odrzucony przebieg na zimno {dropped}" if dropped is not None else ""
    print(
        f"[fps] mediana {out['end_to_end']['fps']} FPS end-to-end "
        f"(zakres {min(e2e)}–{max(e2e)}, rozrzut {spread:.0f}%, "
        f"{len(runs)} przebiegi{extra})"
    )
    if spread > 15:
        print(
            "[fps] UWAGA: rozrzut powyzej 15% — GPU prawdopodobnie zmienia stan cieplny. "
            "Powtorz pomiar na ustabilizowanej maszynie."
        )
    return out


def time_predict_fn(
    predict_fn,
    files: list[Path],
    warmup: int = WARMUP_IMAGES,
) -> dict[str, float]:
    """Mierzy czas dowolnej funkcji predykcji na liscie plikow.

    Uzywane przez modele spoza Ultralytics (RF-DETR). Synchronizacja CUDA po
    kazdym wywolaniu, inaczej mierzy sie kolejkowanie zamiast liczenia.
    """
    import torch

    for p in files[:warmup]:
        predict_fn(str(p))
    torch.cuda.synchronize()

    total = []
    for p in files[warmup:]:
        t0 = time.perf_counter()
        predict_fn(str(p))
        torch.cuda.synchronize()
        total.append(time.perf_counter() - t0)
    return {**_percentiles(total), "fps": round(1.0 / statistics.fmean(total), 2),
            "timed_images": len(total)}


def _measure_once(
    weights: str,
    imgsz: int,
    images_dir: Path | str | None = None,
    warmup: int = WARMUP_IMAGES,
    half: bool = True,
    max_images: int | None = None,
) -> dict:
    """Pojedynczy przebieg pomiarowy."""
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
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    r = measure_fps(a.weights, a.imgsz, a.images_dir, a.warmup,
                    half=not a.fp32, repeats=a.repeats)
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=2), encoding="utf-8")
        print(f"Zapisano {Path(a.out).resolve()}")


if __name__ == "__main__":
    main()
