# AFO detection benchmark — protocol

This file is the contract. It is written **before** any model is trained, and every result in the leaderboard follows it. A result that deviates in any respect is not comparable and is not merged — see [CONTRIBUTING.md](CONTRIBUTING.md).

Status: **protocol fixed, runs in progress.** The leaderboard lives in [README.md](README.md).

---

## 1. Data

- **Source:** `jgasienica/afo-dataset` on Hugging Face, downloaded to `~/.cache/afo`. No dataset files in this repository.
- **Splits:** the **official** `train.txt` / `validation.txt` / `test.txt` as released. Never re-split, never re-balance, never de-duplicate.
- **Labels:** `6categories/`, Darknet format, exported to COCO JSON by `scripts/export_coco.py`.
- **Coordinate clipping:** **on**. The 29 label lines with a coordinate outside `[0,1]` are clipped to `[0,1]` at load time. Source files are never rewritten.
- **Classes:** 6, in the index order of `6categories.names`. Class 3 is spelled `bouy` in the released file; that spelling is authoritative for indices.
- **`i3_200.jpg`** belongs to no split list and is used neither for training nor for evaluation.

**Train on `train`. Select on `val`. Report on `test`. The test split is touched exactly once per model, after training has finished.**

### What the test split cannot measure

The test split contains **zero background images** (train has 720, val 8). No number on this leaderboard measures false alarms on empty water. A detector that fires on wave crests, sun glint or foam is not penalised anywhere in the table. For a maritime SAR dataset this is the most expensive error class, and it is unmeasured by construction. Treat every mAP in the table as "accuracy given that something is there".

Training uses the full official train split, background images included — 25.8% of it is free negative supervision, and removing it would deviate from the protocol.

---

## 2. Models

All initialised from **COCO-pretrained** weights. No AFO-specific architecture changes.

| family | sizes | package |
|--------|-------|---------|
| YOLOv8 | n, s, m | `ultralytics` |
| YOLO11 | n, s, m | `ultralytics` |
| RF-DETR | nano, small, base | `rfdetr` |
| RT-DETR | l | `ultralytics` (optional) |

**Resolution:** `imgsz=640` for every model. Additionally `imgsz=1280` for the best YOLO and the best RF-DETR by val mAP50-95 — AFO objects are small relative to the frame, so input resolution is expected to be the single largest factor, and the benchmark is designed to show that.

---

## 3. Training

- **Seed:** fixed (`0`) for every run.
- **Epoch budget:** 100, identical for all models.
- **Early stopping:** on val mAP50-95, patience 20.
- **Model selection:** the checkpoint with the best **val** mAP50-95. Test is never used for selection.
- **Batch size:** the largest power of two that fits 16 GB VRAM, recorded per run in `train_config.yaml`. Gradient accumulation for RF-DETR at 1280 where needed.
- **Augmentation:** each framework's **defaults**, unchanged. The exact resolved values are dumped into `results/<run>/train_config.yaml` so "defaults" is never an unverifiable claim.
- **No per-model tuning.** The protocol is identical for everyone; a model that underperforms under it underperforms in the table.
- **One run per model.** **Seed variance is not reported** — single-seed differences below roughly 1 mAP point should not be read as real. See TODO.

---

## 4. Evaluation

One evaluator for every model: `afo/evaluate.py`, pycocotools against `~/.cache/afo/coco/test.json`. No test-time augmentation. No multi-scale. No ensembling.

Reported per run:

- **mAP50**, **mAP50-95** (COCO primary metric, averaged over the 6 classes)
- **per-class AP50-95** — mandatory, because `human` is 83% of objects and a good mAP can hide four dead classes
- **AP by object size** — see below
- **params (M)**, **GFLOPs**
- **FPS**, batch 1, FP16, on the GPU named below, averaged over the test split after 50 warm-up images

### Size buckets — measured at the model's input resolution

Boxes are scaled by `imgsz / max(W, H)` — exactly what the detector's letterbox does — **before** being assigned to a bucket:

| bucket | √area at model input |
|--------|----------------------|
| tiny | < 16² px |
| small | 16² – 32² px |
| medium | 32² – 96² px |
| large | > 96² px |

**Why not native pixels.** AFO images are 2.5K–4K, so in native pixels 98.7% of objects are medium or large and the `tiny` bucket holds **15 objects in the entire dataset**. AP computed on 15 objects is noise, not a measurement. Measured at the input the detector actually sees, the same buckets separate the data: at 640 the test split is 57.8% tiny / 38.4% small / 3.8% medium / 0.0% large; at 1280 it is 4.6 / 53.2 / 41.8 / 0.4.

**Consequence, stated plainly: `AP-tiny` at 640 and `AP-tiny` at 1280 are computed over different populations of objects and must not be compared to each other across resolutions.** Within one resolution the column is comparable across models, which is what the leaderboard ranks. Every size column in the table carries its object count.

COCO's standard `AP-S` / `AP-M` / `AP-L` (native pixels, 32²/96² thresholds) are reported alongside for comparability with other papers, with the caveat that on AFO they degenerate to "almost everything is M or L".

---

## 5. Hardware

**NVIDIA GeForce RTX 4090 Laptop GPU, 16 GB VRAM**, driver 617.14.

This is the **laptop** part, not a desktop 4090: fewer SMs, a lower power envelope, and subject to thermal throttling. Its throughput is roughly half a desktop 4090's. Every FPS figure in the leaderboard is labelled with this exact name and with the resolution it was measured at. Runs are performed plugged in, on a performance power profile.

---

## 6. Outputs

Per run, committed to the repository:

```
results/<model>_<imgsz>/
  metrics.json          all numbers above, machine-readable
  predictions_test.json COCO detection results on test
  train_config.yaml     resolved training configuration incl. augmentation
  log.txt               training and evaluation log
```

Checkpoints are **not** committed. Each best checkpoint is uploaded to Hugging Face as `jgasienica/afo-<model>-<imgsz>` with a model card stating the dataset, this protocol, the metrics, the training command, and the licence note: *trained on CC BY-NC-SA data — non-commercial use only*.

---

## 7. Reading the leaderboard honestly

- Ranking is by **test mAP50-95**. Differences under ~1 point are within single-seed noise and should not be called a win.
- `human` dominates the object count, so **mAP is not a summary of SAR usefulness**. Read per-class AP.
- No column measures false alarms (§1).
- FPS is from one laptop GPU in one thermal state; use it for relative comparison between rows, not as an absolute capability claim.

---

## 8. TODO — explicitly out of scope for this phase

- [ ] **Seed-variance study.** Currently one run per model; the table cannot distinguish a 0.5-point difference from noise.
- [ ] **A background-only evaluation set**, so false-alarm rate on empty water becomes measurable. The 720 train background images are the obvious source, but using them means deviating from the official split, so this needs its own protocol.
- [ ] **A de-duplicated split variant**, quantifying what the 7 cross-split duplicate pairs (one of them val↔test) are worth.
- [ ] **Cross-dataset generalisation** to SeaDronesSee and MOBDrone.
- [ ] **Edge inference** (Jetson / NPU) FPS.
- [ ] Higher resolutions than 1280, and tiled inference at native resolution.
