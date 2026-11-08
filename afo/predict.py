"""Predykcje na zbiorze testowym w formacie COCO detection results.

Mapowanie identyfikatorow obrazow bierze sie WYLACZNIE z test.json — po nazwie
pliku. Gdyby ktorys obraz nie mial odpowiednika, funkcja protestuje zamiast po
cichu zgubic detekcje.

Sa tu oba frameworki benchmarku. Predykcja RF-DETR mieszkala wczesniej w
``scripts/train_rfdetr.py``, wiec nie dalo sie jej zaimportowac z zainstalowanego
pakietu — a wiersza leaderboardu nie da sie odtworzyc bez niej. Skrypt treningowy
importuje teraz stad, zeby implementacja byla jedna.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .coco import CATEGORY_ID_OFFSET
from .download import dataset_root

#: klasy rfdetr wg rozmiaru modelu
RFDETR_CLASS = {"nano": "RFDETRNano", "small": "RFDETRSmall", "base": "RFDETRBase"}

#: prog pewnosci. Nisko, bo mAP liczy sie z pelnej krzywej precision-recall;
#: obcinanie na wysokim progu zanizyloby recall i zafalszowalo wynik.
CONF_THRESHOLD = 0.001

#: maksimum detekcji na obraz — zgodne z MAX_DETS_PRIMARY w afo/evaluate.py
MAX_DET = 300


def _gt_index(gt_json: Path) -> dict[str, tuple[int, int, int]]:
    """nazwa pliku -> (image_id, szerokosc, wysokosc) z GT."""
    gt = json.loads(Path(gt_json).read_text(encoding="utf-8"))
    return {img["file_name"]: (img["id"], img["width"], img["height"]) for img in gt["images"]}


def gt_index(gt_json: Path | str) -> dict[str, tuple[int, int, int]]:
    """Publiczny odpowiednik ``_gt_index`` — przydatny przy wlasnym wnioskowaniu."""
    return _gt_index(Path(gt_json))


def _rfdetr_predict(model, path: str, threshold: float, state: dict):
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


def build_rfdetr_predictions(model, files, id_of, out_path: Path, threshold: float,
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

        det = _rfdetr_predict(model, str(p), threshold, state)
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


def predict_rfdetr(
    weights: str,
    size: str,
    imgsz: int,
    out_path: Path | str,
    gt_json: Path | str,
    images_dir: Path | str,
    num_classes: int,
    conf: float = CONF_THRESHOLD,
    shape_override: int | None = None,
) -> Path:
    """Wnioskowanie RF-DETR z gotowych wag — ta sama sciezka, co przy publikacji.

    ``shape_override`` wymusza ksztalt wejscia (pole ``inference_shape_override``
    z metrics.json). Bez niego ksztalt dobiera sie sam przy pierwszym bledzie.
    """
    import rfdetr

    cls = getattr(rfdetr, RFDETR_CLASS[size])
    model = cls(pretrain_weights=str(weights), resolution=imgsz, num_classes=num_classes)

    id_of = _gt_index(Path(gt_json))
    files = sorted(
        p for p in Path(images_dir).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    missing = [p.name for p in files if p.name not in id_of]
    if missing:
        raise RuntimeError(f"{len(missing)} obrazow spoza {gt_json}: {missing[:10]}")

    valid_ids = {i + CATEGORY_ID_OFFSET for i in range(num_classes)}
    state = {"resolution": imgsz,
             "shape": (shape_override, shape_override) if shape_override else None}
    return build_rfdetr_predictions(
        model, files, id_of, Path(out_path), conf, valid_ids, imgsz, out_state=state
    )


def predict_ultralytics(
    weights: str,
    imgsz: int,
    out_path: Path | str,
    gt_json: Path | str | None = None,
    images_dir: Path | str | None = None,
    conf: float = CONF_THRESHOLD,
    max_det: int = MAX_DET,
    half: bool = True,
    device: str | int | None = None,
) -> Path:
    """Uruchamia model Ultralytics na tescie i zapisuje wyniki w formacie COCO.

    ``device=None`` wybiera GPU, jesli jest, a w przeciwnym razie CPU. Wczesniej
    bylo tu zaszyte ``device=0`` i na maszynie bez CUDA funkcja konczyla sie
    bledem zamiast policzyc wolniej.

    Na CPU wylaczamy FP16 — ta precyzja nie ma tam sensu. Opublikowane liczby
    powstaly w FP16 na GPU, wiec wynik z CPU moze sie roznic na ostatnim miejscu
    po przecinku; to rozna arytmetyka, nie inny protokol.
    """
    from ultralytics import YOLO

    # korzen zbioru rozwiazujemy tylko wtedy, gdy jest potrzebny na domyslna
    # sciezke — inaczej wywolanie z jawnymi sciezkami wymagaloby pobranego
    # zbioru tam, gdzie wystarczy sam katalog obrazow
    if gt_json is None or images_dir is None:
        root = dataset_root()
        gt_json = Path(gt_json) if gt_json else root / "coco" / "test.json"
        images_dir = Path(images_dir) if images_dir else root / "yolo1280" / "images" / "test"
    gt_json, images_dir = Path(gt_json), Path(images_dir)
    out_path = Path(out_path)

    id_of = _gt_index(gt_json)
    files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))

    missing = [p.name for p in files if p.name not in id_of]
    if missing:
        raise RuntimeError(
            f"{len(missing)} obrazow z {images_dir} nie ma odpowiednika w {gt_json}: "
            f"{missing[:10]}"
        )
    not_predicted = set(id_of) - {p.name for p in files}
    if not_predicted:
        raise RuntimeError(
            f"{len(not_predicted)} obrazow z test.json nie ma pliku w {images_dir}: "
            f"{sorted(not_predicted)[:10]}"
        )

    if device is None:
        import torch

        device = 0 if torch.cuda.is_available() else "cpu"
    if device == "cpu" and half:
        half = False
    print(f"[predict] device={device} half={half}")

    model = YOLO(weights)
    results = []
    n_rescaled = 0
    from tqdm import tqdm

    for p in tqdm(files, desc="predict", unit="img"):
        r = model.predict(
            str(p), imgsz=imgsz, conf=conf, max_det=max_det, device=device,
            half=half, verbose=False, augment=False,
        )[0]
        img_id, gt_w, gt_h = id_of[p.name]

        # Predykcja idzie po pliku z katalogu `images_dir`, ktory moze byc
        # PRZESKALOWANA kopia (patrz afo.yolo --max-side). Ramki musza wrocic do
        # ukladu wspolrzednych GT, inaczej IoU z test.json wychodzi zero przy
        # poprawnie nauczonym modelu — sprawdzone boles­nie: mAP 0,0000 przy
        # 31 tys. detekcji i val mAP50-95 0,617.
        src_h, src_w = r.orig_shape
        sx, sy = gt_w / src_w, gt_h / src_h
        if (sx, sy) != (1.0, 1.0):
            n_rescaled += 1

        b = r.boxes
        if b is None or len(b) == 0:
            continue
        xyxy = b.xyxy.cpu().numpy()
        cls = b.cls.cpu().numpy().astype(int)
        conf_ = b.conf.cpu().numpy()
        for (x1, y1, x2, y2), k, s in zip(xyxy, cls, conf_):
            results.append({
                "image_id": img_id,
                "category_id": int(k) + CATEGORY_ID_OFFSET,
                "bbox": [round(float(x1) * sx, 2), round(float(y1) * sy, 2),
                         round(float(x2 - x1) * sx, 2), round(float(y2 - y1) * sy, 2)],
                "score": round(float(s), 5),
            })

    if n_rescaled:
        print(
            f"[predict] ramki przeskalowane do ukladu GT dla {n_rescaled}/{len(files)} "
            f"obrazow (predykcja na przeskalowanych kopiach)"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results), encoding="utf-8")
    n_img_with_det = len({r["image_id"] for r in results})
    print(
        f"[predict] {len(results)} detekcji na {n_img_with_det}/{len(files)} obrazach "
        f"-> {out_path}"
    )
    return out_path
