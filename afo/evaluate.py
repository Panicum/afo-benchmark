"""Jeden ewaluator dla wszystkich modeli benchmarku AFO.

pycocotools na oficjalnym tescie. Liczy mAP50, mAP50-95, AP na klase,
AP w kubelkach rozmiaru AFO (mierzonych W PIKSELACH WEJSCIA MODELU) oraz
standardowe COCO AP-S/M/L (w pikselach oryginalu, dla porownywalnosci).

Sztuczka z kubelkami: IoU jest niezmiennicze wzgledem jednolitego skalowania,
wiec zamiast skalowac ramki wystarczy przeskalowac pole `area` o (imgsz/max(W,H))^2
i podac progi areaRng w tej samej skali. Dopasowania sa identyczne, a filtr
rozmiaru dziala w skali, ktora widzi detektor.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

#: kubelki AFO, progi jako (dolny bok, gorny bok) w pikselach wejscia modelu
AFO_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("tiny", 0.0, 16.0),
    ("small", 16.0, 32.0),
    ("medium", 32.0, 96.0),
    ("large", 96.0, 1e5),
)

#: standardowe zakresy COCO (piksele oryginalu)
COCO_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("S", 0.0, 32.0),
    ("M", 32.0, 96.0),
    ("L", 96.0, 1e5),
)

#: maksimum detekcji na obraz. AFO bywa gesty (do 94 obiektow), wiec 100 z COCO
#: potrafi obcinac recall — raportujemy oba.
MAX_DETS_PRIMARY = 300
MAX_DETS_COCO = 100


def _rng(buckets: Sequence[tuple[str, float, float]]) -> list[list[float]]:
    return [[lo * lo, hi * hi] for _, lo, hi in buckets]


def _ap_from_eval(cocoeval, area_idx: int, maxdet_idx: int, iou_idx: int | None = None) -> float:
    """Srednie AP z macierzy precision dla danego zakresu pola."""
    p = cocoeval.eval["precision"]  # [T, R, K, A, M]
    s = p[:, :, :, area_idx, maxdet_idx] if iou_idx is None else p[iou_idx : iou_idx + 1, :, :, area_idx, maxdet_idx]
    s = s[s > -1]
    return float(np.mean(s)) if s.size else float("nan")


def _ap_per_class(cocoeval, area_idx: int, maxdet_idx: int) -> list[float]:
    p = cocoeval.eval["precision"]
    out = []
    for k in range(p.shape[2]):
        s = p[:, :, k, area_idx, maxdet_idx]
        s = s[s > -1]
        out.append(float(np.mean(s)) if s.size else float("nan"))
    return out


def _scale_areas(coco_dict_or_obj, scale_by_img: dict[int, float], is_gt: bool):
    """Mnozy pole `area` przez kwadrat skali danego obrazu (in place)."""
    anns = coco_dict_or_obj.anns if hasattr(coco_dict_or_obj, "anns") else {}
    for ann in anns.values():
        s = scale_by_img.get(ann["image_id"], 1.0)
        ann["area"] = ann["area"] * s * s


def evaluate(
    gt_json: Path | str,
    dt_json: Path | str,
    imgsz: int | None = None,
    class_names: Sequence[str] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Pelna ewaluacja jednego zestawu predykcji.

    Parametry
    ---------
    gt_json : COCO JSON zbioru testowego (z ``scripts/export_coco.py``)
    dt_json : predykcje w formacie COCO detection results
    imgsz   : rozdzielczosc wejscia modelu — potrzebna do kubelkow AFO.
              None wylacza kubelki AFO (zostaja tylko standardowe COCO).
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt_json, dt_json = Path(gt_json), Path(dt_json)

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(gt_json))
        dets = json.loads(dt_json.read_text(encoding="utf-8"))
        dt = gt.loadRes(dets) if dets else None

    cats = [gt.cats[i] for i in sorted(gt.cats)]
    names = list(class_names) if class_names else [c["name"] for c in cats]
    cat_ids = [c["id"] for c in cats]

    n_gt = len(gt.getAnnIds())
    result: dict[str, Any] = {
        "gt_json": str(gt_json),
        "dt_json": str(dt_json),
        "imgsz": imgsz,
        "n_images": len(gt.imgs),
        "n_gt_objects": n_gt,
        "n_detections": len(dets),
        "max_dets_primary": MAX_DETS_PRIMARY,
        "class_names": names,
    }

    if dt is None or not dets:
        if verbose:
            print("BRAK DETEKCJI — wszystkie metryki zerowe.")
        result["empty_predictions"] = True
        return result

    # --- 1. standardowa ewaluacja COCO (pola w pikselach oryginalu) --------
    with contextlib.redirect_stdout(io.StringIO()):
        e = COCOeval(gt, dt, "bbox")
        e.params.maxDets = [1, MAX_DETS_COCO, MAX_DETS_PRIMARY]
        e.params.areaRng = [[0.0, 1e10]] + _rng(COCO_BUCKETS)
        e.params.areaRngLbl = ["all"] + [b[0] for b in COCO_BUCKETS]
        e.evaluate()
        e.accumulate()

    i_primary = 2  # indeks MAX_DETS_PRIMARY w maxDets
    i_coco = 1
    iou50 = 0  # iouThrs[0] == 0.50

    result["mAP50_95"] = _ap_from_eval(e, 0, i_primary)
    result["mAP50"] = _ap_from_eval(e, 0, i_primary, iou_idx=iou50)
    result["mAP50_95_maxdets100"] = _ap_from_eval(e, 0, i_coco)
    for i, (name, _, _) in enumerate(COCO_BUCKETS, start=1):
        result[f"AP_{name}_coco"] = _ap_from_eval(e, i, i_primary)

    per_class = _ap_per_class(e, 0, i_primary)
    per_class50 = [
        float(np.mean(x[x > -1])) if (x := e.eval["precision"][iou50, :, k, 0, i_primary]).size else float("nan")
        for k in range(len(names))
    ]
    gt_per_class = {n: len(gt.getAnnIds(catIds=[c])) for n, c in zip(names, cat_ids)}
    result["per_class"] = {
        n: {
            "AP50_95": per_class[k],
            "AP50": per_class50[k],
            "n_gt": gt_per_class[n],
        }
        for k, n in enumerate(names)
    }

    # --- 2. kubelki AFO w skali wejscia modelu ----------------------------
    if imgsz:
        scale = {
            img_id: imgsz / max(img["width"], img["height"]) for img_id, img in gt.imgs.items()
        }
        with contextlib.redirect_stdout(io.StringIO()):
            gt_s = COCO(str(gt_json))
            dt_s = gt_s.loadRes(dets)
            _scale_areas(gt_s, scale, is_gt=True)
            _scale_areas(dt_s, scale, is_gt=False)

            e2 = COCOeval(gt_s, dt_s, "bbox")
            e2.params.maxDets = [1, MAX_DETS_COCO, MAX_DETS_PRIMARY]
            e2.params.areaRng = [[0.0, 1e10]] + _rng(AFO_BUCKETS)
            e2.params.areaRngLbl = ["all"] + [b[0] for b in AFO_BUCKETS]
            e2.evaluate()
            e2.accumulate()

        # liczebnosc GT w kazdym kubelku — liczba zawsze obok metryki
        counts = dict.fromkeys((b[0] for b in AFO_BUCKETS), 0)
        for ann in gt_s.anns.values():
            side = float(np.sqrt(max(ann["area"], 0.0)))
            for name, lo, hi in AFO_BUCKETS:
                if lo <= side < hi:
                    counts[name] += 1
                    break

        result["afo_buckets"] = {
            name: {
                "AP50_95": _ap_from_eval(e2, i, i_primary),
                "AP50": _ap_from_eval(e2, i, i_primary, iou_idx=iou50),
                "n_gt": counts[name],
            }
            for i, (name, _, _) in enumerate(AFO_BUCKETS, start=1)
        }
        result["afo_buckets_note"] = (
            f"Pola mierzone po przeskalowaniu do imgsz={imgsz}; kubelkow NIE wolno "
            f"porownywac miedzy roznymi imgsz (inna populacja obiektow)."
        )

    if verbose:
        print_report(result)
    return result


def print_report(r: dict[str, Any]) -> None:
    """Czytelny raport na stdout — kazda metryka z liczebnoscia."""
    print(f"\n=== AFO test — {Path(r['dt_json']).name} (imgsz={r['imgsz']}) ===")
    print(f"obrazow {r['n_images']}, obiektow GT {r['n_gt_objects']}, detekcji {r['n_detections']}")
    if r.get("empty_predictions"):
        print("BRAK DETEKCJI")
        return
    print(f"\nmAP50-95 {r['mAP50_95']:.4f}   mAP50 {r['mAP50']:.4f}   "
          f"(maxDets={r['max_dets_primary']}; przy 100: {r['mAP50_95_maxdets100']:.4f})")

    print("\nAP na klase (AP50-95 / AP50, n obiektow GT):")
    for name, d in r["per_class"].items():
        print(f"  {name:16s} {d['AP50_95']:.4f}  {d['AP50']:.4f}   n={d['n_gt']:5d}")

    if "afo_buckets" in r:
        print(f"\nKubelki AFO (w pikselach wejscia, imgsz={r['imgsz']}):")
        for name, d in r["afo_buckets"].items():
            ap = "   —  " if np.isnan(d["AP50_95"]) else f"{d['AP50_95']:.4f}"
            print(f"  {name:8s} AP50-95 {ap}   n={d['n_gt']:5d}")
        print("  UWAGA: nieporownywalne miedzy roznymi imgsz.")

    print("\nStandardowe COCO (piksele oryginalu, dla porownania z innymi pracami):")
    for b, _, _ in COCO_BUCKETS:
        v = r.get(f"AP_{b}_coco", float("nan"))
        print(f"  AP-{b}  {'  —  ' if np.isnan(v) else f'{v:.4f}'}")


#: Wersja dwuklasowa z pracy (sekcja 3.1): human+buoy -> small_obj,
#: reszta -> large_obj. Klucze to identyfikatory kategorii COCO (1..6).
TWO_CAT_REMAP = {1: 1, 4: 1, 2: 2, 3: 2, 5: 2, 6: 2}
TWO_CAT_NAMES = {1: "small_obj", 2: "large_obj"}


def evaluate_two_category(
    gt_json: Path | str,
    dt_json: Path | str,
    imgsz: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Ewaluacja po scaleniu 6 klas do 2, jak w Tabeli 2 oryginalnej pracy.

    UWAGA na interpretacje: to NIE jest model uczony na dwoch klasach. To model
    6-klasowy, ktorego predykcje i GT scalono po fakcie. Daje liczbe o tej samej
    DEFINICJI KLAS co praca, ale nie o tym samym treningu — model nadal placil
    za rozroznianie szesciu kategorii. Traktowac jako dolne oszacowanie.
    """
    import json as _json
    import tempfile

    gt = _json.loads(Path(gt_json).read_text(encoding="utf-8"))
    dt = _json.loads(Path(dt_json).read_text(encoding="utf-8"))

    gt = dict(gt)
    gt["categories"] = [
        {"id": i, "name": n, "supercategory": "afo"} for i, n in sorted(TWO_CAT_NAMES.items())
    ]
    gt["annotations"] = [
        {**a, "category_id": TWO_CAT_REMAP[a["category_id"]]} for a in gt["annotations"]
    ]
    dt = [{**d, "category_id": TWO_CAT_REMAP[d["category_id"]]} for d in dt]

    tmp = Path(tempfile.mkdtemp())
    gt_p, dt_p = tmp / "gt_2cat.json", tmp / "dt_2cat.json"
    gt_p.write_text(_json.dumps(gt), encoding="utf-8")
    dt_p.write_text(_json.dumps(dt), encoding="utf-8")

    r = evaluate(gt_p, dt_p, imgsz=imgsz, verbose=False)
    r["categories"] = "2 (small_obj = human+bouy; large_obj = board/boat/sailboat/kayak)"
    r["caveat"] = (
        "Model uczony na 6 klasach, predykcje scalone po fakcie — nie to samo, "
        "co trening na wersji dwuklasowej."
    )
    if verbose:
        print("\n=== Wersja DWUKLASOWA (definicja klas jak w Tabeli 2 pracy) ===")
        print(f"AP (50-95) {r['mAP50_95']:.4f}   AP@50 {r['mAP50']:.4f}")
        for name, d in r["per_class"].items():
            print(f"  {name:10s} AP50-95 {d['AP50_95']:.4f}  AP@50 {d['AP50']:.4f}  n={d['n_gt']}")
        print(f"  {r['caveat']}")
    return r


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("predictions", help="COCO detection results JSON")
    ap.add_argument("--gt", default=None, help="test.json (domyslnie <cache>/coco/test.json)")
    ap.add_argument("--imgsz", type=int, default=None, help="rozdzielczosc wejscia modelu")
    ap.add_argument("--out", default=None, help="zapisz metryki do JSON")
    args = ap.parse_args()

    gt = args.gt
    if gt is None:
        from .download import dataset_root

        gt = dataset_root() / "coco" / "test.json"

    r = evaluate(gt, args.predictions, imgsz=args.imgsz)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(r, indent=2), encoding="utf-8")
        print(f"\nZapisano {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
