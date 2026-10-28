#!/usr/bin/env python
"""Trening RF-DETR wg protokolu z BENCHMARK.md.

Ten sam podzial, ten sam zbior testowy i ten sam ewaluator co dla Ultralytics —
rozni sie tylko framework. Wyniki laduja w ``results/<model>_<imgsz>[_2cat]/``
w identycznym formacie, zeby leaderboard nie musial wiedziec, skad pochodza.

UWAGA na mapowanie klas: RF-DETR zwraca ``class_id``, ktore MUSI odpowiadac
``category_id`` z COCO. Skrypt to sprawdza i przerywa, zamiast po cichu
wyprodukowac predykcje w zlym numerowaniu — raz juz kosztowalo to bieg, ktory
dal mAP 0,0000 przy 31 tys. detekcji.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

from afo.coco import CATEGORY_ID_OFFSET
from afo.download import dataset_root
from afo.vram import DEFAULT_LIMIT_FRAC, check_peak, reset_peak, sysmem_fallback_active

#: batch dobrany pomiarem na RTX 4090 Laptop (16 GB). Patrz BENCHMARK.md §5.
DEFAULT_BATCH = {640: 8, 1280: 2}
EFFECTIVE_BATCH = 16  # grad_accum dobierany tak, zeby to utrzymac

SIZES = {"nano": "RFDETRNano", "small": "RFDETRSmall", "base": "RFDETRBase"}

CONF_THRESHOLD = 0.001
MAX_DET = 300


def git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _predict(model, path: str, threshold: float, state: dict):
    """predict() z obejsciem niespojnosci rfdetr co do rozdzielczosci.

    RF-DETR base ma patch 14 i 4 okna, czyli blok 56, a ``train`` przyjmuje
    rozdzielczosc 640, ktora przez 56 sie nie dzieli. ``predict`` to odrzuca,
    wiec model da sie nauczyc, a potem nie da sie go uzyc. Przy pierwszym takim
    bledzie wyliczamy najblizszy zgodny ksztalt i trzymamy go dla reszty biegu.

    To ODSTEPSTWO od protokolu: wnioskowanie idzie w innej rozdzielczosci niz
    trening. Zapisywane w metrics.json jako `inference_shape_override`, zeby
    wiersz nie udawal porownywalnego z biegami, ktore maja obie te same.
    """
    shape = state.get("shape")
    if shape is not None:
        return model.predict(path, threshold=threshold, shape=shape)
    try:
        return model.predict(path, threshold=threshold)
    except ValueError as exc:
        m = re.search(r"block_size=(\d+)", str(exc))
        if not m:
            raise
        block = int(m.group(1))
        res = state["resolution"]
        cand = max(block, round(res / block) * block)
        state["shape"] = (cand, cand)
        print(
            f"\n[predict] UWAGA: rozdzielczosc {res} nie dzieli sie przez {block} — "
            f"biblioteka odmawia wnioskowania. Przechodze na {cand}. Trening byl "
            f"na {res}, wiec ten wiersz NIE jest wprost porownywalny z biegami, "
            f"gdzie trening i wnioskowanie maja te sama rozdzielczosc."
        )
        return model.predict(path, threshold=threshold, shape=state["shape"])


def build_predictions(model, files, id_of, out_path: Path, threshold: float,
                      valid_ids: set[int], resolution: int,
                      out_state: dict | None = None) -> Path:
    """Predykcje w formacie COCO detection results, w ukladzie wspolrzednych GT."""
    import numpy as np
    from PIL import Image
    from tqdm import tqdm

    results = []
    seen_class_ids: set[int] = set()
    dropped: dict[int, int] = {}
    state = out_state if out_state is not None else {}
    state.setdefault("resolution", resolution)
    state.setdefault("shape", None)
    for p in tqdm(files, desc="predict", unit="img"):
        img_id, gt_w, gt_h = id_of[p.name]
        with Image.open(p) as im:
            src_w, src_h = im.size
        sx, sy = gt_w / src_w, gt_h / src_h

        det = _predict(model, str(p), threshold, state)
        if det is None or len(det) == 0:
            continue
        xyxy = np.asarray(det.xyxy)
        cls = np.asarray(det.class_id).astype(int)
        conf = np.asarray(det.confidence)
        order = np.argsort(-conf)[:MAX_DET]
        for i in order:
            x1, y1, x2, y2 = xyxy[i]
            raw = int(cls[i])
            seen_class_ids.add(raw)
            # RF-DETR indeksuje klasy OD ZERA, a COCO od jedynki — ustalone
            # empirycznie: przy progu 0,25 rozklad klas 0..5 pokrywa sie z
            # rozkladem GT co do dziesiatych procenta (85,1% vs 85,2% dla
            # human). Model zwraca tez sporadyczna klase o indeksie
            # len(classes), ktora nie odpowiada niczemu i ma znikome score.
            cat = raw + CATEGORY_ID_OFFSET
            if cat not in valid_ids:
                dropped[raw] = dropped.get(raw, 0) + 1
                continue
            results.append({
                "image_id": img_id,
                "category_id": cat,
                "bbox": [round(float(x1) * sx, 2), round(float(y1) * sy, 2),
                         round(float(x2 - x1) * sx, 2), round(float(y2 - y1) * sy, 2)],
                "score": round(float(conf[i]), 5),
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results), encoding="utf-8")
    print(f"[predict] {len(results)} detekcji, surowe class_id modelu: "
          f"{sorted(seen_class_ids)} -> category_id +{CATEGORY_ID_OFFSET}")
    if dropped:
        n = sum(dropped.values())
        print(f"[predict] odrzucono {n} detekcji o class_id poza zakresem: "
              f"{ {k: v for k, v in sorted(dropped.items())} }")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", required=True, choices=sorted(SIZES))
    ap.add_argument("--imgsz", type=int, default=640, choices=[640, 1280])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--limit-frac", type=float, default=DEFAULT_LIMIT_FRAC)
    ap.add_argument(
        "--categories", default="6categories", choices=["6categories", "2categories"]
    )
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--resume", action="store_true",
        help="wznow przerwany trening z last.ckpt zamiast zaczynac od zera",
    )
    ap.add_argument("--post-only", action="store_true")
    ap.add_argument(
        "--train-hours", type=float, default=None,
        help="rzeczywisty czas treningu przy --post-only; bez tego zapisze sie null, "
             "a zerowy czas wygladalby jak darmowy trening",
    )
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("Brak CUDA.")
    gpu = torch.cuda.get_device_name(0)

    two_cat = args.categories == "2categories"
    suffix = "_2cat" if two_cat else ""
    model_name = f"rfdetr-{args.size}"
    run_name = f"{model_name}_{args.imgsz}{suffix}"

    root = dataset_root()
    data_dir = root / (f"rfdetr_yolo1280{suffix}" if two_cat else "rfdetr_yolo1280")
    gt = root / ("coco2cat" if two_cat else "coco") / "test.json"
    images_dir = root / f"yolo1280{suffix}" / "images" / "test"
    for p in (data_dir, gt, images_dir):
        if not p.exists():
            raise SystemExit(f"Brak {p} — przygotuj dane (afo.yolo / afo.rfdetr_data / export_coco)")

    results_root = Path(args.results_dir).resolve()
    out_dir = results_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = results_root / "_runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    class Tee:
        def __init__(self, path):
            self.f = open(path, "w", encoding="utf-8")
        def write(self, s):
            sys.__stdout__.write(s); self.f.write(s); self.f.flush()
        def flush(self):
            sys.__stdout__.flush(); self.f.flush()

    sys.stdout = Tee(out_dir / "log.txt")

    batch = args.batch or DEFAULT_BATCH[args.imgsz]
    accum = max(1, EFFECTIVE_BATCH // batch)
    print(f"=== {run_name} ===")
    print(f"GPU: {gpu}")
    print(f"sysmem fallback sterownika: {sysmem_fallback_active()}")
    print(f"dane: {data_dir}\nbatch {batch} x grad_accum {accum} = efektywny {batch * accum}\n")

    import rfdetr

    cls = getattr(rfdetr, SIZES[args.size])

    train_s = args.train_hours * 3600 if args.train_hours else None
    if not args.post_only:
        reset_peak()
        t0 = time.perf_counter()
        model = cls(resolution=args.imgsz)
        # last.ckpt niesie pelny stan treningu (wagi, optymalizator, licznik
        # epok), wiec wznowienie kontynuuje budzet epok zamiast go resetowac.
        last = run_dir / "last.ckpt"
        resume_from = str(last) if (args.resume and last.is_file()) else None
        if args.resume and not resume_from:
            print(f"--resume podane, ale brak {last} — zaczynam od zera")
        elif resume_from:
            print(f"wznawiam z {last}")
        model.train(
            dataset_dir=str(data_dir), epochs=args.epochs, batch_size=batch,
            grad_accum_steps=accum, output_dir=str(run_dir), num_workers=args.workers,
            early_stopping=True, early_stopping_patience=args.patience,
            tensorboard=False, wandb=False, resume=resume_from,
        )
        train_s = time.perf_counter() - t0
        c = check_peak(args.limit_frac)
        print(f"\ntrening: {train_s/3600:.2f} h | {c}")
        if not c.fits:
            print("UWAGA: szczyt przekroczyl limit — bieg mogl thrashowac po PCIe.")
        del model
        torch.cuda.empty_cache()

    best = next(
        (run_dir / n for n in ("checkpoint_best_ema.pth", "checkpoint_best_total.pth",
                               "checkpoint_best_regular.pth") if (run_dir / n).is_file()),
        None,
    )
    if best is None:
        raise SystemExit(f"Brak checkpointu w {run_dir}")
    print(f"checkpoint: {best.name}")

    # --- predykcje -----------------------------------------------------------
    gt_doc = json.loads(gt.read_text(encoding="utf-8"))
    id_of = {i["file_name"]: (i["id"], i["width"], i["height"]) for i in gt_doc["images"]}
    expected_ids = {c["id"] for c in gt_doc["categories"]}
    files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    missing = [p.name for p in files if p.name not in id_of]
    if missing:
        raise SystemExit(f"{len(missing)} obrazow spoza test.json: {missing[:5]}")

    pred_state: dict = {}
    infer = cls(pretrain_weights=str(best), resolution=args.imgsz,
                num_classes=len(expected_ids))
    pred_path = build_predictions(infer, files, id_of, out_dir / "predictions_test.json",
                                  CONF_THRESHOLD, expected_ids, args.imgsz,
                                  pred_state)

    dets = json.loads(pred_path.read_text(encoding="utf-8"))
    got = {d["category_id"] for d in dets}
    if dets and not got <= expected_ids:
        raise SystemExit(
            f"Niezgodne numerowanie klas: predykcje maja {sorted(got)}, "
            f"a test.json oczekuje {sorted(expected_ids)}. Przerwane — inaczej mAP "
            f"wyszloby zerowe bez zadnego komunikatu."
        )

    # --- ewaluacja -----------------------------------------------------------
    from afo.evaluate import evaluate

    metrics = evaluate(gt, pred_path, imgsz=args.imgsz)
    metrics["categories"] = args.categories
    if not two_cat:
        from afo.evaluate import evaluate_two_category

        metrics["two_category"] = evaluate_two_category(gt, pred_path, imgsz=args.imgsz)

    # --- FPS -----------------------------------------------------------------
    # Jak w torze Ultralytics: pomiar zaraz po treningu bylby pomiarem goracej
    # karty, nie modelu. Zmierzyc osobno, po cool-downie.
    metrics["speed"] = None
    print("\n[fps] pominiete — zmierz osobnym przebiegiem po cool-downie")

    n_params = sum(p.numel() for p in infer.model.model.parameters())
    metrics["model"] = model_name
    metrics["framework"] = "rfdetr"
    if pred_state.get("shape"):
        metrics["inference_shape_override"] = {
            "trained_at": args.imgsz,
            "inferred_at": pred_state["shape"][0],
            "why": "rfdetr predict() wymaga rozdzielczosci podzielnej przez "
                   "patch_size*num_windows, czego train() nie egzekwuje",
            "comparability": "wiersz NIE jest wprost porownywalny z biegami, "
                             "gdzie trening i wnioskowanie maja te sama rozdzielczosc",
        }
    metrics["params_M"] = round(n_params / 1e6, 2)
    metrics["gflops"] = None
    metrics["train"] = {
        "epochs_budget": args.epochs, "patience": args.patience, "batch": batch,
        "grad_accum_steps": accum, "effective_batch": batch * accum, "seed": args.seed,
        "hours": None if train_s is None else round(train_s / 3600, 3), "gpu": str(gpu),
    }
    metrics["weights_hf"] = f"jgasienica/afo-{model_name}-{args.imgsz}{suffix.replace('_','-')}"

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "train_config.yaml").write_text(
        yaml.safe_dump({
            "_afo_protocol": "BENCHMARK.md", "framework": "rfdetr",
            "model": model_name, "imgsz": args.imgsz, "categories": args.categories,
            "epochs": args.epochs, "patience": args.patience, "batch_size": batch,
            "grad_accum_steps": accum, "seed": args.seed, "dataset_dir": str(data_dir),
            "checkpoint": best.name, "_gpu": str(gpu), "_git_rev": git_rev(),
            "_python": platform.python_version(), "_torch": str(torch.__version__),
            "_train_hours": metrics["train"]["hours"],
            "_command": " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
        }, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"\nZapisano {out_dir.resolve()}")


if __name__ == "__main__":
    main()
