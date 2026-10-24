"""Jeden ewaluator dla wszystkich modeli benchmarku AFO.

pycocotools na oficjalnym tescie. Liczy mAP50, mAP50-95, AP na klase,
AP w kubelkach rozmiaru AFO (w STALEJ skali referencyjnej, patrz
``REFERENCE_SIDE``) oraz standardowe COCO AP-S/M/L (w pikselach oryginalu,
dla porownywalnosci z innymi pracami).

Sztuczka z kubelkami: IoU jest niezmiennicze wzgledem jednolitego skalowania,
wiec zamiast skalowac ramki wystarczy przeskalowac pole `area` o kwadrat skali
i podac progi areaRng w tej samej skali. Dopasowania sa identyczne, a filtr
rozmiaru dziala w wybranej skali.

Osobno liczona jest diagnostyka PUNKTU PRACY: mAP jest bezprogowe, a wdrozenie
SAR chodzi na jednym progu ufnosci. ``operating_point`` podaje prog
maksymalizujacy F1 oraz liczbe falszywych alarmow na obraz przy tym progu —
liczbe, ktorej z AP nie widac.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

#: Dluzszy bok SKALI REFERENCYJNEJ, w ktorej mierzone sa kubelki rozmiaru.
#:
#: Kluczowa decyzja: skala jest STALA i niezalezna od imgsz, z jakim model byl
#: uczony. Gdyby kubelki liczyc w pikselach wejscia modelu, ten sam obiekt
#: bylby "tiny" przy 640 i "small" przy 1280 — czyli AP-tiny z dwoch biegow
#: opisywaloby inne populacje i nie dalo sie ich porownac. A porownanie 640
#: wobec 1280 jest calym sensem tych biegow.
#:
#: 640 jako odniesienie, bo to podstawowa rozdzielczosc benchmarku; wartosc
#: jest umowna, wazne, ze jedna dla wszystkich wierszy tabeli.
REFERENCE_SIDE = 640

#: kubelki AFO, progi jako (dolny bok, gorny bok) w pikselach skali referencyjnej
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


def _portable(path: Path | str) -> str:
    """Sciezka niezalezna od maszyny: cache AFO jako <AFO_ROOT>, reszta wzgledna."""
    import os

    p = Path(path).resolve()
    try:
        from .download import default_root

        root = default_root()
        if p.is_relative_to(root):
            return "<AFO_ROOT>/" + p.relative_to(root).as_posix()
    except Exception:
        pass
    try:
        return p.relative_to(Path(os.getcwd()).resolve()).as_posix()
    except ValueError:
        return p.name


def operating_point(
    gt_doc: dict,
    dets: list[dict],
    iou_thr: float = 0.5,
    n_steps: int = 99,
    return_curve: bool = False,
) -> dict[str, Any]:
    """Prog ufnosci maksymalizujacy F1 i liczba falszywych alarmow na obraz.

    Po co, skoro jest mAP: AP jest BEZPROGOWE — calkuje po calej krzywej
    precision-recall. Wdrozony system SAR chodzi na jednym, konkretnym progu,
    i wtedy liczy sie "ile razy na kadr detektor pokaze cos, czego nie ma".
    Tej liczby w AP nie widac, a to ona decyduje, czy operator ufa systemowi.

    Dopasowanie zachlanne po malejacej ufnosci, IoU 0,5, BEZ rozroznienia klas:
    w SAR wykrycie obiektu jest wazniejsze niz przypisanie kategorii.
    """
    boxes_by_img: dict[int, list[tuple[float, float, float, float]]] = {}
    for a in gt_doc["annotations"]:
        x, y, w, h = a["bbox"]
        boxes_by_img.setdefault(a["image_id"], []).append((x, y, x + w, y + h))
    n_images = len(gt_doc["images"])
    n_gt = sum(len(v) for v in boxes_by_img.values())
    if not dets or not n_gt:
        return {"threshold": float("nan"), "note": "brak detekcji albo brak GT"}

    dets_by_img: dict[int, list[tuple[float, tuple]]] = {}
    for d in dets:
        x, y, w, h = d["bbox"]
        dets_by_img.setdefault(d["image_id"], []).append(
            (float(d["score"]), (x, y, x + w, y + h))
        )

    def iou(a, b) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    # dla kazdej detekcji: najwyzszy prog, przy ktorym nadal jest TP
    scored: list[tuple[float, bool]] = []
    for img_id, dd in dets_by_img.items():
        gts = list(boxes_by_img.get(img_id, []))
        used = [False] * len(gts)
        for score, box in sorted(dd, key=lambda t: -t[0]):
            best_i, best_v = -1, iou_thr
            for i, g in enumerate(gts):
                if used[i]:
                    continue
                v = iou(box, g)
                if v >= best_v:
                    best_i, best_v = i, v
            if best_i >= 0:
                used[best_i] = True
                scored.append((score, True))
            else:
                scored.append((score, False))

    # Pelna krzywa. Prog 0 wlaczony celowo: bez niego nie widac SUFITU RECALLU,
    # czyli ilu obiektow model nie znajduje przy ZADNEJ ufnosci. Dla SAR to
    # wazniejsza liczba niz ktorykolwiek punkt pracy.
    curve = []
    for k in range(0, n_steps + 1):
        thr = k / (n_steps + 1)
        tp = sum(1 for s, ok in scored if s >= thr and ok)
        fp = sum(1 for s, ok in scored if s >= thr and not ok)
        fn = n_gt - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / n_gt
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        curve.append({
            "threshold": round(thr, 3), "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn,
            "fp_per_image": fp / n_images, "fn_per_image": fn / n_images,
        })

    out: dict[str, Any] = dict(max(curve, key=lambda d: d["f1"]))
    out["n_images"] = n_images
    out["n_gt_objects"] = n_gt
    out["iou_threshold"] = iou_thr
    out["class_agnostic"] = True

    # Punkty przy ZADANYM RECALLU. F1 wazy przeoczenie i falszywy alarm rowno,
    # a w SAR przeoczony czlowiek w wodzie kosztuje wiecej niz jeden falszywy
    # alarm. Operator pyta "ile bede mial fatygi przy czulosci 80/90%", i to
    # jest pytanie, na ktore ta tabela odpowiada.
    at_recall: dict[str, Any] = {}
    for target in (0.8, 0.9):
        feasible = [c for c in curve if c["recall"] >= target]
        if feasible:
            # najwyzsza precyzja sposrod progow osiagajacych zadany recall
            at_recall[f"r{int(target * 100)}"] = max(feasible, key=lambda d: d["precision"])
        else:
            at_recall[f"r{int(target * 100)}"] = {
                "reachable": False,
                "max_recall": max(c["recall"] for c in curve),
                "note": f"model nie osiaga recallu {target} przy zadnym progu",
            }
    out["at_recall"] = at_recall

    ceiling = max(curve, key=lambda d: d["recall"])
    out["recall_ceiling"] = {
        "recall": ceiling["recall"],
        "threshold": ceiling["threshold"],
        "missed_objects": ceiling["fn"],
        "note": (
            "Najwyzszy osiagalny recall, przy progu zbiegajacym do zera. Obiekty "
            "ponad ten sufit nie sa znajdowane przy ZADNEJ ufnosci — zadne "
            "strojenie progu ich nie odzyska."
        ),
    }

    out["note"] = (
        "Progi dobrane na zbiorze TESTOWYM — to gorne ograniczenia, bo wybrane na "
        "tych samych danych, na ktorych mierzone. Do wdrozenia prog dobiera sie na "
        "walidacji. Dopasowanie BEZ rozroznienia klas, wiec ta diagnostyka wazy "
        "obiektami, podczas gdy mAP wazy klasami — przy 85% udziale klasy human "
        "obie moga dac inna kolejnosc modeli i obie sa poprawne."
    )
    if return_curve:
        # Pelna krzywa nie trafia do metrics.json (100 punktow na bieg to szum
        # w pliku, ktory ma byc czytelny) — tylko na zadanie, dla wykresow.
        out["curve"] = curve
    return out


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
    imgsz   : rozdzielczosc wejscia modelu. Zapisywana w wyniku jako metadana;
              NIE wplywa na kubelki rozmiaru — te licza sie zawsze w skali
              referencyjnej REFERENCE_SIDE, zeby byly porownywalne miedzy
              biegami o roznym imgsz.
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
        # Sciezki zapisujemy przenosnie: artefakty trafiaja do publicznego repo,
        # a bezwzgledna sciezka z czyjegos dysku nie niesie tam zadnej informacji.
        "gt_json": _portable(gt_json),
        "dt_json": _portable(dt_json),
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

    # --- 2. kubelki AFO w stalej skali referencyjnej -----------------------
    if True:
        scale = {
            img_id: REFERENCE_SIDE / max(img["width"], img["height"])
            for img_id, img in gt.imgs.items()
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
        result["afo_buckets_reference_side"] = REFERENCE_SIDE
        result["afo_buckets_note"] = (
            f"Pola mierzone po przeskalowaniu kazdego obrazu do dluzszego boku "
            f"{REFERENCE_SIDE} px. Skala jest stala i niezalezna od imgsz, wiec "
            f"kubelki opisuja te sama populacje obiektow we wszystkich biegach "
            f"i SA porownywalne miedzy 640 a 1280."
        )

    # diagnostyka punktu pracy — liczba, ktorej mAP nie pokazuje
    try:
        result["operating_point"] = operating_point(
            json.loads(gt_json.read_text(encoding="utf-8")), dets
        )
    except Exception as exc:  # diagnostyka nie moze wywrocic ewaluacji
        print(f"UWAGA: nie udalo sie policzyc punktu pracy ({type(exc).__name__}: {exc})")

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
        ref = r.get("afo_buckets_reference_side", REFERENCE_SIDE)
        print(f"\nKubelki AFO (stala skala referencyjna: dluzszy bok {ref} px):")
        for name, d in r["afo_buckets"].items():
            ap = "   —  " if np.isnan(d["AP50_95"]) else f"{d['AP50_95']:.4f}"
            print(f"  {name:8s} AP50-95 {ap}   n={d['n_gt']:5d}")
        print("  Skala stala — porownywalne miedzy biegami o roznym imgsz.")

    print("\nStandardowe COCO (piksele oryginalu, dla porownania z innymi pracami):")
    for b, _, _ in COCO_BUCKETS:
        v = r.get(f"AP_{b}_coco", float("nan"))
        print(f"  AP-{b}  {'  —  ' if np.isnan(v) else f'{v:.4f}'}")

    if "operating_point" in r:
        op = r["operating_point"]
        print("\nPunkt pracy (prog maksymalizujacy F1, IoU 0,5, bez rozroznienia klas):")
        print(f"  prog ufnosci  {op['threshold']:.3f}")
        print(f"  precyzja {op['precision']:.4f}   recall {op['recall']:.4f}   "
              f"F1 {op['f1']:.4f}")
        print(f"  falszywych alarmow na obraz: {op['fp_per_image']:.2f}   "
              f"przeoczen na obraz: {op['fn_per_image']:.2f}")
        for key, label in (("r80", "recall 80%"), ("r90", "recall 90%")):
            d = (op.get("at_recall") or {}).get(key, {})
            if d.get("reachable") is False:
                print(f"  przy {label}: NIEOSIAGALNE (max recall {d['max_recall']:.3f})")
            elif d:
                print(f"  przy {label}: prog {d['threshold']:.2f}, precyzja "
                      f"{d['precision']:.3f}, FP na obraz {d['fp_per_image']:.2f}")


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
    # sciezki wskazuja katalog tymczasowy — zastepujemy opisem zrodla
    r["gt_json"] = f"{_portable(gt_json)} (scalone do 2 klas)"
    r["dt_json"] = f"{_portable(dt_json)} (scalone do 2 klas)"
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
