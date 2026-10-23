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


#: Ile razy powtorzyc caly przebieg; raportujemy mediane plus zakres.
DEFAULT_REPEATS = 4

#: Temperatura GPU, ponizej ktorej uznajemy karte za wychlodzona.
#:
#: To NIE jest kwestia rozgrzewki, tylko throttlingu. Te same wagi dawaly
#: 43,8 FPS tuz po 45-minutowym treningu i 76,2 FPS na zimnej karcie — bo
#: gorace GPU laptopa obniza zegary. Odrzucenie pierwszego przebiegu tego nie
#: leczy, skoro kolejne tez sa gorace. Jedyne uczciwe wyjscie to zmierzyc po
#: wychlodzeniu i zapisac temperature oraz zegar OBOK wyniku.
COOLDOWN_TEMP_C = 50
COOLDOWN_MAX_WAIT_S = 900
COOLDOWN_POLL_S = 15


def gpu_state() -> dict[str, float | None]:
    """Temperatura, zegar SM i pobor mocy z nvidia-smi."""
    import subprocess

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,power.draw",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=20,
        ).strip().splitlines()[0]
        t, c, p = (x.strip() for x in out.split(","))
        return {"temp_c": float(t), "clock_sm_mhz": float(c),
                "power_w": None if p.upper().startswith("N") else float(p)}
    except Exception:
        return {"temp_c": None, "clock_sm_mhz": None, "power_w": None}


def wait_for_cooldown(
    target_c: float = COOLDOWN_TEMP_C,
    max_wait_s: int = COOLDOWN_MAX_WAIT_S,
    poll_s: int = COOLDOWN_POLL_S,
) -> dict:
    """Czeka, az GPU zejdzie ponizej ``target_c``. Zwraca, co zastal."""
    t0 = time.perf_counter()
    st = gpu_state()
    if st["temp_c"] is None:
        print("[fps] nie udalo sie odczytac temperatury — pomijam cool-down")
        return {"waited_s": 0, "reached": None, **st}
    start_temp = st["temp_c"]
    while st["temp_c"] is not None and st["temp_c"] > target_c:
        waited = time.perf_counter() - t0
        if waited > max_wait_s:
            print(f"[fps] cool-down przerwany po {waited/60:.1f} min — "
                  f"{st['temp_c']:.0f} °C, cel {target_c} °C. Wynik bedzie zanizony.")
            break
        print(f"[fps] cool-down: {st['temp_c']:.0f} °C > {target_c} °C, "
              f"czekam ({waited/60:.1f} min)...", flush=True)
        time.sleep(poll_s)
        st = gpu_state()
    waited = time.perf_counter() - t0
    print(f"[fps] start pomiaru przy {st['temp_c']:.0f} °C, "
          f"zegar {st['clock_sm_mhz']:.0f} MHz (czekano {waited/60:.1f} min)")
    return {"waited_s": round(waited, 1), "start_temp_c": start_temp, "reached": True, **st}


def measure_fps(
    weights: str,
    imgsz: int,
    images_dir: Path | str | None = None,
    warmup: int = WARMUP_IMAGES,
    half: bool = True,
    max_images: int | None = None,
    repeats: int = DEFAULT_REPEATS,
    cooldown: bool = True,
    cooldown_temp_c: float = COOLDOWN_TEMP_C,
) -> dict:
    """Mierzy FPS modelu Ultralytics na zbiorze testowym (mediana z ``repeats``).

    Przed pomiarem czeka na wychlodzenie karty i zapisuje jej stan cieplny —
    bez tego liczba mowi wiecej o tym, co dzialo sie wczesniej, niz o modelu.
    """
    thermal = wait_for_cooldown(cooldown_temp_c) if cooldown else {"skipped": True}
    n = max(1, repeats)
    runs = [_measure_once(weights, imgsz, images_dir, warmup, half, max_images) for _ in range(n)]
    thermal_after = gpu_state()
    dropped = None

    e2e = sorted(r["end_to_end"]["fps"] for r in runs)
    fwd = sorted(r["forward_only"]["fps"] for r in runs)
    out = dict(runs[-1])
    out["repeats"] = len(runs)
    out["thermal_before"] = thermal
    out["thermal_after"] = thermal_after
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
    t_a = thermal_after.get("temp_c")
    print(
        f"[fps] mediana {out['end_to_end']['fps']} FPS end-to-end "
        f"(zakres {min(e2e)}–{max(e2e)}, rozrzut {spread:.0f}%, {len(runs)} przebiegi"
        + (f", {t_a:.0f} °C po pomiarze" if t_a else "") + ")"
    )
    if spread > 15:
        print(
            "[fps] UWAGA: rozrzut powyzej 15% — karta zmienia zegary w trakcie pomiaru. "
            "Wydluz cool-down albo ustaw profil wysokiej wydajnosci."
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
