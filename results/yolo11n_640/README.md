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
base_model: yolo11n
model-index:
  - name: afo-yolo11n-640
    results:
      - task:
          type: object-detection
        dataset:
          name: AFO (Aerial dataset of Floating Objects)
          type: jgasienica/afo-dataset
          split: test
        metrics:
          - type: mAP50-95
            value: 0.2417
          - type: mAP50
            value: 0.4716
---

# AFO — yolo11n @ 640

`yolo11n` fine-tuned from COCO-pretrained weights on **[AFO](https://huggingface.co/datasets/jgasienica/afo-dataset)**, a maritime search-and-rescue dataset of 3,641 drone images with 39,991 hand-annotated objects in six classes.

Produced by the reproducible benchmark at **[github.com/jgasienica/afo-dataset](https://github.com/jgasienica/afo-dataset)** under the protocol in [`BENCHMARK.md`](https://github.com/jgasienica/afo-dataset/blob/main/BENCHMARK.md). Every number below is measured on the **official test split** with one shared pycocotools evaluator, so it is comparable to every other row of that leaderboard — and to nothing else.

## Results on the AFO test split

| metric | value |
|---|---|
| mAP50-95 | **0.2417** |
| mAP50 | 0.4716 |
| mAP50-95 @ maxDets=100 | 0.2416 |

### Per-class AP (test)

| class | AP50-95 | AP50 | test objects |
|---|---|---|---|
| `human` | 0.1641 | 0.5380 | 6,511 |
| `wind/sup-board` | 0.5038 | 0.9452 | 634 |
| `boat` | 0.2348 | 0.3770 | 143 |
| `bouy` | 0.0157 | 0.0512 | 59 |
| `sailboat` | 0.0766 | 0.0985 | 28 |
| `kayak` | 0.4552 | 0.8195 | 264 |

### AP by object size, measured at the model's input resolution

| bucket | √area at input | AP50-95 | test objects |
|---|---|---|---|
| tiny | < 16² | 0.2222 | 4,424 |
| small | 16²–32² | 0.2721 | 2,930 |
| medium | 32²–96² | 0.3813 | 285 |
| large | > 96² | — (no objects) | 0 |

Every image is rescaled to a **fixed 640 px longest side** before bucketing, because AFO images are 2.5K–4K and in native pixels the `tiny` bucket would hold 15 objects in the whole dataset. The scale is deliberately independent of the training resolution, so these columns **are** comparable across runs at different `imgsz`.

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
| wall-clock | 0.723 h |

```bash
python scripts/train_ultralytics.py --model yolo11n --imgsz 640
```

## Speed

Batch 1, FP16, `NVIDIA GeForce RTX 4090 Laptop GPU`, averaged over 464 test images after 50 warm-up images.

| | FPS | mean | median | p95 |
|---|---|---|---|---|
| end-to-end (preprocess + forward + NMS) | **63.23** | 13.74 ms | 13.076 ms | 18.161 ms |
| forward only | 122.2 | 6.617 ms | 6.251 ms | 9.045 ms |

This is a **laptop** GPU (76 SMs against a desktop 4090's 128) and it throttles. Repeated measurements of these same weights varied by tens of percent depending on thermal state. Use this figure to compare rows measured in the same session, not as an absolute capability claim.

## Usage

```python
from ultralytics import YOLO

model = YOLO("afo-yolo11n-640.pt")
results = model("drone_frame.jpg", imgsz=640)
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
