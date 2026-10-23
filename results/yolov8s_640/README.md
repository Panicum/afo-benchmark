---
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
base_model: yolov8s
model-index:
  - name: afo-yolov8s-640
    results:
      - task:
          type: object-detection
        dataset:
          name: AFO (Aerial dataset of Floating Objects)
          type: jgasienica/afo-dataset
          split: test
        metrics:
          - type: mAP50-95
            value: 0.2630
          - type: mAP50
            value: 0.5231
---

# AFO — yolov8s @ 640

`yolov8s` fine-tuned from COCO-pretrained weights on **[AFO](https://huggingface.co/datasets/jgasienica/afo-dataset)**, a maritime search-and-rescue dataset of 3,641 drone images with 39,991 hand-annotated objects in six classes.

Produced by the reproducible benchmark at **[github.com/jgasienica/afo-dataset](https://github.com/jgasienica/afo-dataset)** under the protocol in [`BENCHMARK.md`](https://github.com/jgasienica/afo-dataset/blob/main/BENCHMARK.md). Every number below is measured on the **official test split** with one shared pycocotools evaluator, so it is comparable to every other row of that leaderboard — and to nothing else.

## Results on the AFO test split

| metric | value |
|---|---|
| mAP50-95 | **0.2630** |
| mAP50 | 0.5231 |
| mAP50-95 @ maxDets=100 | 0.2629 |

### Per-class AP (test)

| class | AP50-95 | AP50 | test objects |
|---|---|---|---|
| `human` | 0.1421 | 0.4728 | 6,511 |
| `wind/sup-board` | 0.4649 | 0.9130 | 634 |
| `boat` | 0.3234 | 0.5939 | 143 |
| `bouy` | 0.0379 | 0.1131 | 59 |
| `sailboat` | 0.0634 | 0.0792 | 28 |
| `kayak` | 0.5466 | 0.9666 | 264 |

### AP by object size, measured at the model's input resolution

| bucket | √area at input | AP50-95 | test objects |
|---|---|---|---|
| tiny | < 16² | 0.2592 | 4,424 |
| small | 16²–32² | 0.2848 | 2,930 |
| medium | 32²–96² | 0.3898 | 285 |
| large | > 96² | — (no objects) | 0 |

Boxes are scaled by `imgsz / max(W, H)` before bucketing, because AFO images are 2.5K–4K and in native pixels the `tiny` bucket would hold 15 objects in the whole dataset. **These columns are not comparable across different `imgsz`** — they describe different populations of objects.

## Training

| | |
|---|---|
| data | official AFO train split (2,787 images, 26,960 objects), coordinates clipped to [0,1] |
| selection | best val mAP50-95; test touched once, after training |
| epochs | 100 budget, patience 20 |
| batch | 64 |
| seed | 0 |
| image size | 640 |
| augmentation | framework defaults, unmodified |
| hardware | NVIDIA GeForce RTX 4090 Laptop GPU |
| wall-clock | 1.243 h |

```bash
python scripts/train_ultralytics.py --model yolov8s --imgsz 640
```

## Speed

Batch 1, FP16, `NVIDIA GeForce RTX 4090 Laptop GPU`, averaged over 464 test images after 50 warm-up images.

| | FPS | mean | median | p95 |
|---|---|---|---|---|
| end-to-end (preprocess + forward + NMS) | **69.99** | 14.039 ms | 13.145 ms | 20.337 ms |
| forward only | 158.66 | 6.303 ms | 5.513 ms | 11.238 ms |

This is a **laptop** GPU (76 SMs against a desktop 4090's 128) and it throttles. Repeated measurements of these same weights varied by tens of percent depending on thermal state. Use this figure to compare rows measured in the same session, not as an absolute capability claim.

## Usage

```python
from ultralytics import YOLO

model = YOLO("afo-yolov8s-640.pt")
results = model("drone_frame.jpg", imgsz=640)
```

Classes, in index order: `human`, `wind/sup-board`, `boat`, `bouy`, `sailboat`, `kayak`. The `bouy` spelling is a typo in the original dataset release and is kept so class indices stay correct.

## Limitations — read before deploying

- **No metric here measures false alarms.** The AFO test split contains **zero background images** (train has 720, val 8). A detector that fires on wave crests, foam or sun glint is not penalised by any number above. For maritime SAR that is the most expensive error class, and it is unmeasured by construction.
- **The dataset is heavily imbalanced**: `human` is 83% of all objects, `sailboat` 0.4%. Treat the per-class table as the real result; mAP averages over classes that have very different support.
- **One run, one seed.** No variance estimate. Differences under roughly 1 mAP point against another row should not be read as real.
- **Val-to-test gap is large by design.** The test split is nine videos absent from train and val, so these numbers describe generalisation to unseen scenes, not to unseen frames.
- Trained on imagery resized to a 1280 px longest side; see BENCHMARK.md §1.

## License

**CC BY-NC-SA 3.0, inherited from the training data.** AFO is released under CC BY-NC-SA 3.0, so these weights are **non-commercial use only** and derivatives must carry the same licence.

## Citation

```bibtex
@article{gasienica2021afo,
  title    = {An ensemble deep learning method with optimized weights
              for drone-based water rescue and surveillance},
  author   = {G{\k{a}}sienica-J{\'o}zkowy, Jan and Knapik, Mateusz and Cyganek, Bogus{\l}aw},
  journal  = {Integrated Computer-Aided Engineering},
  volume   = {28},
  number   = {3},
  pages    = {221--235},
  year     = {2021},
  doi      = {10.3233/ICA-210649}
}
```
