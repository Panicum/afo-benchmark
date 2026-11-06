#!/usr/bin/env python
"""Karta modelu i wgranie wag na Hugging Face.

Domyslnie TYLKO generuje kartę do pliku; wgranie wymaga jawnego ``--push``.
Powod: repozytorium modelu jest publiczne, a wgranie trudno cofnac.

Karta jest budowana Z ``metrics.json``, a nie z jednego szablonu na wszystko.
Wczesniejsza wersja zakladala Ultralytics i 6 klas, wiec na 11 z 18 publicznych
repozytoriow obiecywala `from ultralytics import YOLO` dla RF-DETR i wypisywala
szesc klas pod modelem dwuklasowym. Kazde zdanie o frameworku, klasach, komendzie
treningu i pliku wag pochodzi teraz z metryk danego biegu.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RFDETR_CLASS = {"nano": "RFDETRNano", "small": "RFDETRSmall", "base": "RFDETRBase"}

TEMPLATE = """---
license: cc-by-nc-sa-3.0
tags:
  - object-detection
  - maritime
  - search-and-rescue
  - drone
  - uav
  - small-object-detection
  - afo
datasets:
  - jgasienica/afo-dataset
metrics:
  - mAP
{library_line}pipeline_tag: object-detection
model-index:
  - name: {slug}
    results:
      - task:
          type: object-detection
        dataset:
          name: AFO (Aerial dataset of Floating Objects)
          type: jgasienica/afo-dataset
          split: test
        metrics:
          - type: mAP50-95
            value: {map5095:.4f}
          - type: mAP50
            value: {map50:.4f}
---

# AFO — {model} @ {imgsz}{title_suffix}

`{model}` fine-tuned from COCO-pretrained weights on **[AFO](https://huggingface.co/datasets/jgasienica/afo-dataset)**, a maritime search-and-rescue dataset of 3,641 drone images with 39,991 hand-annotated objects.{task_sentence}

Produced by the reproducible benchmark at **[github.com/Panicum/afo-benchmark](https://github.com/Panicum/afo-benchmark)** under the protocol in [`BENCHMARK.md`](https://github.com/Panicum/afo-benchmark/blob/main/BENCHMARK.md). Every number below is measured on the **official test split** with one shared pycocotools evaluator, so it is comparable to every other row of that leaderboard — and to nothing else.

## Results on the AFO test split

| metric | value |
|---|---|
| mAP50-95 | **{map5095:.4f}** |
| mAP50 | {map50:.4f} |
| mAP50-95 @ maxDets=100 | {map100:.4f} |

### Per-class AP (test)

| class | AP50-95 | AP50 | test objects |
|---|---|---|---|
{per_class}

### AP by object size

| bucket | √area at reference scale | AP50-95 | test objects |
|---|---|---|---|
{buckets}

Every image is rescaled to a **fixed 640 px longest side** before bucketing, because AFO images are 2.5K–4K and in native pixels the `tiny` bucket would hold 15 objects in the whole dataset. The scale is deliberately independent of the training resolution, so these rows are comparable across runs at different `imgsz`.

## Training

| | |
|---|---|
| data | official AFO train split (2,787 images, 26,960 objects), coordinates clipped to [0,1] |
| labels | `{categories}` |
| selection | best val mAP50-95; test touched once, after training |
| epochs | {epochs} budget, patience {patience} |
| batch | {batch} |
| seed | {seed} |
| image size | {imgsz} |
| augmentation | framework defaults, unmodified |
| hardware | {gpu} |
| wall-clock | {hours} h |

```bash
{train_cmd}
```

## Usage

```python
{usage}
```

{classes_block}

## Limitations

- **The dataset is heavily imbalanced.** Treat the per-class table as the real result; mAP averages over classes with very different support.
- **One run, one seed.** No variance estimate — differences under roughly 1 mAP point against another row should not be read as real.
- **Val-to-test gap is large by design.** The test split is drawn from videos absent from train and val, so these numbers describe generalisation to unseen scenes, not unseen frames.
- Trained on imagery resized to a 1280 px longest side; see BENCHMARK.md §1.

## License

**CC BY-NC-SA 3.0, inherited from the training data.** AFO is released under CC BY-NC-SA 3.0, so these weights are **non-commercial use only** and derivatives must carry the same licence.

## Citation

```bibtex
@article{{gasienica2021afo,
  title    = {{An ensemble deep learning method with optimized weights
              for drone-based water rescue and surveillance}},
  author   = {{G{{\\k{{a}}}}sienica-J{{\\'o}}zkowy, Jan and Knapik, Mateusz and Cyganek, Bogus{{\\l}}aw}},
  journal  = {{Integrated Computer-Aided Engineering}},
  volume   = {{28}},
  number   = {{3}},
  pages    = {{221--235}},
  year     = {{2021}},
  doi      = {{10.3233/ICA-210649}}
}}
```
"""

BUCKET_RANGE = {"tiny": "< 16²", "small": "16²–32²", "medium": "32²–96²", "large": "> 96²"}

TWO_CAT_NOTE = (
    "The two classes are the **semantic merge from Table 2 of the original paper** — "
    "`small_obj` = human + buoy, `large_obj` = board + boat + sailboat + kayak — so this "
    "model's numbers are directly comparable to the published ones. It is a merge by object "
    "kind, not by box area, so `small_obj` is not COCO's AP-small."
)


def slug_of(metrics: dict) -> str:
    """Nazwa pliku wag i wpisu model-index — ta sama konwencja co repo na HF."""
    suffix = "-2cat" if metrics.get("categories") == "2categories" else ""
    return f"afo-{metrics['model']}-{metrics['imgsz']}{suffix}"


def weights_path(run: Path, metrics: dict) -> Path | None:
    """Wagi leza w innym miejscu dla kazdego frameworka.

    Ultralytics kopiuje ``best.pt`` do katalogu wyniku; RF-DETR zostawia
    checkpoint w katalogu roboczym ``results/_runs/<bieg>/``. Szukanie tylko
    ``best.pt`` bylo powodem, dla ktorych cztery repozytoria RF-DETR mialy
    karte obiecujaca wagi i zadnego pliku.
    """
    work = run.parent / "_runs" / run.name
    if metrics.get("framework") == "rfdetr":
        cands = [work / "checkpoint_best_ema.pth", work / "checkpoint_best_total.pth"]
    else:
        cands = [run / "best.pt", work / "weights" / "best.pt"]
    return next((c for c in cands if c.is_file()), None)


def usage_block(metrics: dict, weights_name: str) -> str:
    if metrics.get("framework") == "rfdetr":
        size = metrics["model"].split("-", 1)[1]
        cls = RFDETR_CLASS.get(size, "RFDETRBase")
        ovr = metrics.get("inference_shape_override")
        shape, note = "", ""
        if ovr:
            s = ovr["inferred_at"]
            shape = f", shape=({s}, {s})"
            # komentarz w osobnej linii, nie w wywolaniu: inline zakomentowalby
            # domykajacy nawias i wklejony kod by sie nie uruchomil
            note = (f"# this checkpoint rejects {metrics['imgsz']} at inference — "
                    f"see the note below\n")
        return (
            f"from rfdetr import {cls}\n\n"
            f'model = {cls}(\n'
            f'    pretrain_weights="{weights_name}",\n'
            f'    resolution={metrics["imgsz"]},\n'
            f'    num_classes={len(metrics["class_names"])},\n'
            f')\n'
            f'{note}'
            f'detections = model.predict("drone_frame.jpg", threshold=0.25{shape})\n'
            f"# detections.class_id holds 0-based indices into the class list below"
        )
    return (
        f"from ultralytics import YOLO\n\n"
        f'model = YOLO("{weights_name}")\n'
        f'results = model("drone_frame.jpg", imgsz={metrics["imgsz"]})'
    )


def build_card(metrics: dict, weights_name: str) -> str:
    pc = "\n".join(
        f"| `{k}` | {v['AP50_95']:.4f} | {v['AP50']:.4f} | {v['n_gt']:,} |"
        for k, v in metrics["per_class"].items()
    )
    bk = "\n".join(
        f"| {k} | {BUCKET_RANGE.get(k, '')} | "
        + ("— (no objects)" if v["n_gt"] == 0 else f"{v['AP50_95']:.4f}")
        + f" | {v['n_gt']:,} |"
        for k, v in (metrics.get("afo_buckets") or {}).items()
    )

    two_cat = metrics.get("categories") == "2categories"
    names = metrics.get("class_names") or []
    classes = ", ".join(f"`{n}`" for n in names)
    classes_block = f"Classes, in index order: {classes}."
    if two_cat:
        classes_block += "\n\n" + TWO_CAT_NOTE
    elif "bouy" in names:
        classes_block += (" The `bouy` spelling is a typo in the original dataset release "
                          "and is kept so class indices stay correct.")

    ovr = metrics.get("inference_shape_override")
    if ovr:
        classes_block += (
            f"\n\nThis checkpoint was **trained at {metrics['imgsz']}** but its `predict()` "
            f"rejects {metrics['imgsz']} as an input shape (the backbone requires a multiple "
            f"of its window size), so every number on this card was measured at "
            f"**{ovr['inferred_at']}** — pass that shape to reproduce them."
        )

    script = "train_rfdetr.py" if metrics.get("framework") == "rfdetr" else "train_ultralytics.py"
    cmd = f"python scripts/{script} --model {metrics['model']} --imgsz {metrics['imgsz']}"
    if two_cat:
        cmd += " --categories 2categories"

    if two_cat:
        task_sentence = (" Two merged classes, following Table 2 of the original AFO paper.")
    else:
        task_sentence = " Six classes."

    # library_name steruje widgetem na HF; dla RF-DETR nie ma takiej wartosci,
    # a zle podany ultralytics podsuwal czytelnikowi kod, ktory sie nie uruchomi.
    library = ("library_name: ultralytics\n"
               if metrics.get("framework") != "rfdetr" else "")

    t = metrics.get("train", {})
    return TEMPLATE.format(
        model=metrics["model"],
        slug=slug_of(metrics),
        imgsz=metrics["imgsz"],
        title_suffix=" (2-class)" if two_cat else "",
        task_sentence=task_sentence,
        categories=metrics.get("categories", "6categories"),
        map5095=metrics["mAP50_95"],
        map50=metrics["mAP50"],
        map100=metrics.get("mAP50_95_maxdets100", float("nan")),
        per_class=pc,
        buckets=bk,
        epochs=t.get("epochs_budget", "?"),
        patience=t.get("patience", "?"),
        batch=t.get("batch", "?"),
        seed=t.get("seed", "?"),
        gpu=t.get("gpu", "?"),
        hours=t.get("hours", "?"),
        train_cmd=cmd,
        usage=usage_block(metrics, weights_name),
        classes_block=classes_block,
        library_line=library,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="np. results/yolo11n_640")
    ap.add_argument("--repo", default=None, help="domyslnie z metrics.json")
    ap.add_argument("--push", action="store_true", help="wgraj na HF (publiczne!)")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--card-only", action="store_true",
                    help="z --push: wyslij tylko karte i metryki, pomin wagi")
    args = ap.parse_args()

    run = Path(args.run_dir)
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))

    w = weights_path(run, metrics)
    weights_name = f"{slug_of(metrics)}{w.suffix if w else '.pt'}"
    card = build_card(metrics, weights_name)
    card_path = run / "README.md"
    card_path.write_text(card, encoding="utf-8")
    print(f"Karta: {card_path.resolve()}")
    print(f"Wagi:  {w.resolve() if w else 'BRAK — karta opisuje plik, ktorego nie ma'}")

    repo = args.repo or metrics.get("weights_hf")
    if not args.push:
        print(f"\n(bez --push nic nie wysylam; docelowe repo: {repo})")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(card_path), path_in_repo="README.md",
        repo_id=repo, repo_type="model",
    )
    if w is not None and not args.card_only:
        print(f"wysylam {w.stat().st_size / 1048576:.0f} MB jako {weights_name} ...")
        api.upload_file(
            path_or_fileobj=str(w), path_in_repo=weights_name,
            repo_id=repo, repo_type="model",
        )
    for extra in ("metrics.json", "train_config.yaml"):
        p = run / extra
        if p.is_file():
            api.upload_file(
                path_or_fileobj=str(p), path_in_repo=extra, repo_id=repo, repo_type="model"
            )
    print(f"Wgrane: https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
