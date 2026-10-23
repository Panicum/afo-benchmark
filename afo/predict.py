"""Predykcje na zbiorze testowym w formacie COCO detection results.

Mapowanie identyfikatorow obrazow bierze sie WYLACZNIE z test.json — po nazwie
pliku. Gdyby ktorys obraz nie mial odpowiednika, funkcja protestuje zamiast po
cichu zgubic detekcje.
"""

from __future__ import annotations

import json
from pathlib import Path

from .coco import CATEGORY_ID_OFFSET
from .download import dataset_root

#: prog pewnosci. Nisko, bo mAP liczy sie z pelnej krzywej precision-recall;
#: obcinanie na wysokim progu zanizyloby recall i zafalszowalo wynik.
CONF_THRESHOLD = 0.001

#: maksimum detekcji na obraz — zgodne z MAX_DETS_PRIMARY w afo/evaluate.py
MAX_DET = 300


def _image_id_map(gt_json: Path) -> dict[str, int]:
    gt = json.loads(Path(gt_json).read_text(encoding="utf-8"))
    return {img["file_name"]: img["id"] for img in gt["images"]}


def predict_ultralytics(
    weights: str,
    imgsz: int,
    out_path: Path | str,
    gt_json: Path | str | None = None,
    images_dir: Path | str | None = None,
    conf: float = CONF_THRESHOLD,
    max_det: int = MAX_DET,
    half: bool = True,
) -> Path:
    """Uruchamia model Ultralytics na tescie i zapisuje wyniki w formacie COCO."""
    from ultralytics import YOLO

    root = dataset_root()
    gt_json = Path(gt_json) if gt_json else root / "coco" / "test.json"
    images_dir = Path(images_dir) if images_dir else root / "yolo1280" / "images" / "test"
    out_path = Path(out_path)

    id_of = _image_id_map(gt_json)
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

    model = YOLO(weights)
    results = []
    from tqdm import tqdm

    for p in tqdm(files, desc="predict", unit="img"):
        r = model.predict(
            str(p), imgsz=imgsz, conf=conf, max_det=max_det, device=0,
            half=half, verbose=False, augment=False,
        )[0]
        img_id = id_of[p.name]
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
                "bbox": [round(float(x1), 2), round(float(y1), 2),
                         round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                "score": round(float(s), 5),
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results), encoding="utf-8")
    n_img_with_det = len({r["image_id"] for r in results})
    print(
        f"[predict] {len(results)} detekcji na {n_img_with_det}/{len(files)} obrazach "
        f"-> {out_path}"
    )
    return out_path
