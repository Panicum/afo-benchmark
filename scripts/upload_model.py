#!/usr/bin/env python
"""Karta modelu i wgranie wag na Hugging Face.

Domyslnie TYLKO generuje kartę do pliku; wgranie wymaga jawnego ``--push``.
Powod: repozytorium modelu jest publiczne, a wgranie trudno cofnac.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
library_name: ultralytics
pipeline_tag: object-detection
base_model: {model}
model-index:
  - name: afo-{model}-{imgsz}
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

# AFO — {model} @ {imgsz}

`{model}` fine-tuned from COCO-pretrained weights on **[AFO](https://huggingface.co/datasets/jgasienica/afo-dataset)**, a maritime search-and-rescue dataset of 3,641 drone images with 39,991 hand-annotated objects in six classes.

Produced by the reproducible benchmark at **[github.com/jgasienica/afo-dataset](https://github.com/jgasienica/afo-dataset)** under the protocol in [`BENCHMARK.md`](https://github.com/jgasienica/afo-dataset/blob/main/BENCHMARK.md). Every number below is measured on the **official test split** with one shared pycocotools evaluator, so it is comparable to every other row of that leaderboard — and to nothing else.

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

### AP by object size, measured at the model's input resolution

| bucket | √area at input | AP50-95 | test objects |
|---|---|---|---|
{buckets}

Every image is rescaled to a **fixed 640 px longest side** before bucketing, because AFO images are 2.5K–4K and in native pixels the `tiny` bucket would hold 15 objects in the whole dataset. The scale is deliberately independent of the training resolution, so these columns **are** comparable across runs at different `imgsz`.

## Training

| | |
|---|---|
| data | official AFO train split (2,787 images, 26,960 objects), coordinates clipped to [0,1] |
| selection | best val mAP50-95; test touched once, after training |
| epochs | {epochs} budget, patience {patience} |
| batch | {batch} |
| seed | {seed} |
| image size | {imgsz} |
| augmentation | framework defaults, unmodified |
| hardware | {gpu} |
| wall-clock | {hours} h |

```bash
python scripts/train_ultralytics.py --model {model} --imgsz {imgsz}
```

## Speed

{speed_block}

## Usage

```python
from ultralytics import YOLO

model = YOLO("afo-{model}-{imgsz}.pt")
results = model("drone_frame.jpg", imgsz={imgsz})
```

Classes, in index order: `human`, `wind/sup-board`, `boat`, `bouy`, `sailboat`, `kayak`. The `bouy` spelling is a typo in the original dataset release and is kept so class indices stay correct.

## Limitations — read before deploying

- **mAP is threshold-free; deployment is not.** AP integrates the whole precision-recall curve, while a fielded system runs at one confidence threshold. False alarms *are* penalised by mAP (any detection not matching a ground-truth box lowers precision, and the test split is overwhelmingly open water by area), but the number an operator cares about — false alarms per frame at the chosen threshold — is not visible in AP. The `operating_point` block in `metrics.json` reports it.
- **The dataset is heavily imbalanced**: `human` is 83% of all objects, `sailboat` 0.4%. Treat the per-class table as the real result; mAP averages over classes that have very different support.
- **One run, one seed.** No variance estimate. Differences under roughly 1 mAP point against another row should not be read as real.
- **Val-to-test gap is large by design.** The test split is nine videos absent from train and val, so these numbers describe generalisation to unseen scenes, not to unseen frames.
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


def build_card(metrics: dict) -> str:
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

    sp = metrics.get("speed") or {}
    if sp:
        e2e, fwd = sp.get("end_to_end", {}), sp.get("forward_only", {})
        speed_block = (
            f"Batch 1, FP16, `{sp.get('gpu')}`, averaged over {sp.get('timed_images')} "
            f"test images after {sp.get('warmup_images')} warm-up images.\n\n"
            f"| | FPS | mean | median | p95 |\n|---|---|---|---|---|\n"
            f"| end-to-end (preprocess + forward + NMS) | **{e2e.get('fps')}** | "
            f"{e2e.get('mean_ms')} ms | {e2e.get('median_ms')} ms | {e2e.get('p95_ms')} ms |\n"
            f"| forward only | {fwd.get('fps')} | {fwd.get('mean_ms')} ms | "
            f"{fwd.get('median_ms')} ms | {fwd.get('p95_ms')} ms |\n\n"
            f"This is a **laptop** GPU (76 SMs against a desktop 4090's 128) and it throttles. "
            f"Repeated measurements of these same weights varied by tens of percent depending on "
            f"thermal state. Use this figure to compare rows measured in the same session, not as "
            f"an absolute capability claim."
        )
    else:
        speed_block = "_Not measured._"

    t = metrics.get("train", {})
    return TEMPLATE.format(
        model=metrics["model"],
        imgsz=metrics["imgsz"],
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
        speed_block=speed_block,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="np. results/yolo11n_640")
    ap.add_argument("--repo", default=None, help="domyslnie z metrics.json")
    ap.add_argument("--push", action="store_true", help="wgraj na HF (publiczne!)")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    run = Path(args.run_dir)
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    card = build_card(metrics)
    card_path = run / "README.md"
    card_path.write_text(card, encoding="utf-8")
    print(f"Karta: {card_path.resolve()}")

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
    w = run / "best.pt"
    if w.is_file():
        api.upload_file(
            path_or_fileobj=str(w),
            path_in_repo=f"afo-{metrics['model']}-{metrics['imgsz']}.pt",
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
