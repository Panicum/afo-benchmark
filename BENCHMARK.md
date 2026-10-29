# AFO detection benchmark — protocol

This file is the contract. It is written **before** any model is trained, and every result in the leaderboard follows it. A result that deviates in any respect is not comparable and is not merged — see [CONTRIBUTING.md](CONTRIBUTING.md).

Status: **protocol fixed, 18 runs complete, matrix unfinished.** What is done and what is missing is listed in the README's Status table — including the FPS column, which is empty because the measurement protocol changed and the new pass has not been run. The leaderboard lives in [README.md](README.md).

---

## 1. Data

- **Source:** `jgasienica/afo-dataset` on Hugging Face, downloaded to `~/.cache/afo`. No dataset files in this repository.
- **Splits:** the **official** `train.txt` / `validation.txt` / `test.txt` as released. Never re-split, never re-balance, never de-duplicate.
- **Labels:** `6categories/` or `2categories/` depending on the arm (see below), Darknet format, exported to COCO JSON by `scripts/export_coco.py`.
- **Coordinate clipping:** **on**. The 29 label lines with a coordinate outside `[0,1]` are clipped to `[0,1]` at load time. Source files are never rewritten.
- **Classes:** 6, in the index order of `6categories.names`. Class 3 is spelled `bouy` in the released file; that spelling is authoritative for indices.
- **`i3_200.jpg`** belongs to no split list and is used neither for training nor for evaluation.

**Train on `train`. Select on `val`. Report on `test`. The test split is touched exactly once per model, after training has finished.**

### Two label sets, two leaderboards

AFO ships three label directories. Two are benchmarked here, and they are **separate tasks with separate tables**:

| arm | labels | classes | why |
|---|---|---|---|
| **6-class** | `6categories/` | human, wind/sup-board, boat, bouy, sailboat, kayak | the harder task; no published baselines exist |
| **2-class** | `2categories/` | `small_obj` = human+buoy, `large_obj` = board+boat+sailboat+kayak | the version the 2021 paper benchmarked; keeps results comparable to Table 2 of [doi:10.3233/ICA-210649](https://doi.org/10.3233/ICA-210649) |

The two-class split is **semantic, not a box-area threshold**, despite the names (paper §3.1). Its `AP@50 Small Object` column is per-class AP for the merged class and must not be read as COCO's AP-small.

A score on one arm says nothing about the other, so **rows are never mixed between the tables**. `scripts/make_leaderboard.py` routes each result by the `categories` field in its `metrics.json` and prints anything it could not place.

Collapsing a 6-class model's predictions into the two-class scheme is recorded under `two_category` in every 6-class `metrics.json`, because it costs nothing and bounds the arm that was not run. It is **not** a two-class leaderboard row: measured on YOLO11n @640, it understates a real two-class run by 0.49 AP / 2.75 AP@50.

`1category/` is left unbenchmarked for now.

### Pre-resizing (applies to every run)

AFO images are 2.5K–4K. Training at `imgsz=640` with default mosaic augmentation decodes four source JPEGs per sample, which makes the pipeline **CPU-bound on JPEG decode, not GPU-bound** — measured: `yolo11n` and `yolo11m` differ by 13% per epoch despite a ~7× difference in FLOPs.

Every image is therefore resized **once, offline**, to a longest side of **1280 px** (`cv2.INTER_AREA`, JPEG quality 95) by `python -m afo.yolo --max-side 1280`. This cut epoch time from 145 s to 78 s and the dataset from 4.5 GB to 1.1 GB.

Labels are normalised, so they are unchanged. Both benchmark resolutions (640 and 1280) are ≤ 1280, so no run is upsampling. The one honest caveat: at `imgsz=1280` a 3840 px source now reaches the network via two resampling steps (3840→1280→letterbox) instead of one. The difference is negligible but it is a deviation from "raw files in", and it is recorded here rather than left implicit.

### What the test split cannot measure

The test split contains **zero background images** (train has 720, val 8). It is tempting to conclude that false alarms on water therefore go unmeasured. That conclusion is wrong.

This is detection, not classification. Every detection that matches no ground-truth box counts as a false positive and lowers precision, regardless of whether the frame contains other objects. The median AFO object covers 0.085% of its frame, so the test split is overwhelmingly open water by area — there is no shortage of water on which a detector can be caught firing. mAP penalises wave-crest detections correctly.

What the absence of empty frames actually costs is narrower and worth stating precisely:

1. **The operational figure is not measured directly.** "How many false alarms will an operator see per patrol frame" is naturally expressed on frames with nothing in them, and there are none in test. It is estimable from the annotated frames, which is what the operating-point diagnostic in §4 does.
2. **mAP is threshold-free; deployment is not.** AP integrates over the whole precision-recall curve. A fielded SAR system runs at one confidence threshold, and the thing that decides whether an operator trusts it — false alarms per frame at that threshold — is invisible in AP. This gap is real and has nothing to do with background images; it would exist even if test were half empty frames.

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
- **Batch size:** the largest power of two whose **measured peak allocation stays under 85% of physical VRAM**, probed per run and recorded in `train_config.yaml`. Gradient accumulation for RF-DETR at 1280 where needed.
- **Augmentation:** each framework's **defaults**, unchanged. The exact resolved values are dumped into `results/<run>/train_config.yaml` so "defaults" is never an unverifiable claim. For Ultralytics these resolve to `mosaic=1.0` (closed for the last 10 epochs), `scale=0.5`, `translate=0.1`, `fliplr=0.5`, `flipud=0.0`, `hsv_h=0.015`, `hsv_s=0.7`, `hsv_v=0.4`, `erasing=0.4`, `mixup=0.0`, `copy_paste=0.0`, `degrees=0.0`.
- **No per-model tuning.** The protocol is identical for everyone; a model that underperforms under it underperforms in the table.
- **One run per model**, with one exception: the model that ends up at the **top of each leaderboard is re-run with three seeds**, so the headline number carries a spread instead of resting on a single draw. Everything else is single-seed, and single-seed differences below roughly 1 mAP point should not be read as real. (Observed: `yolo11n` and `yolov8n` at 640 differ by 0.16 points — that is a tie, not a ranking.)
- **Both arms use the identical recipe.** The only difference between a 6-class and a 2-class run is `--categories`.

### Why batch size is capped by measurement, not by OOM

On Windows the NVIDIA driver does **not** raise `OutOfMemoryError` when an allocation exceeds VRAM. It silently moves the excess to system RAM and streams tensors over PCIe. The run completes, reports no error, and is several times slower.

Measured on this project: RF-DETR base at 640 requested **22.8 GB on a 16 GB card** at batch 16 and took **2812 s/epoch against 437 s at batch 4** — 6.4× slower, with no warning of any kind. A budget built from that measurement would have been wrong by hours per model.

So: "fits" means *measured peak ≤ 85% of physical VRAM*, never *no exception was raised*. This is enforced in [`afo/vram.py`](afo/vram.py), used by both the batch probe and the training scripts, and `scripts/estimate_budget.py` discards any calibration row whose peak exceeded physical VRAM.

---

## 4. Evaluation

One evaluator for every model: `afo/evaluate.py`, pycocotools against `~/.cache/afo/coco/test.json`. No test-time augmentation. No multi-scale. No ensembling.

Reported per run:

- **mAP50**, **mAP50-95** (COCO primary metric, averaged over the 6 classes)
- **per-class AP50-95** — mandatory, because `human` is 83% of objects and a good mAP can hide four dead classes
- **AP by object size** — see below
- **params (M)**, **GFLOPs**
- **FPS**, batch 1, FP16, on the GPU named below, averaged over the test split after 50 warm-up images, with `torch.cuda.synchronize()` around each timing so the figure is computation and not queueing. Reported twice: **end-to-end** (preprocess + forward + NMS) and **forward only**; the first is what runs on a boat. **Measured in a separate pass after training, not at the end of a run** — see §5.

**Detections per image: 300, not COCO's 100.** AFO is dense — up to 94 objects in one image, and 304 images hold 40 or more. At a cap of 100, a handful of confident false positives can push true positives past the cut and depress recall. `mAP50-95` at the COCO cap of 100 is reported alongside as `mAP50_95_maxdets100`.

### Size buckets — a fixed reference scale

Every image is rescaled to a **640 px longest side** before its boxes are bucketed. The scale is a constant of the benchmark (`afo.evaluate.REFERENCE_SIDE`) and is **deliberately independent of the resolution the model was trained at**:

| bucket | √area at the 640 px reference scale |
|--------|-------------------------------------|
| tiny | < 16² px |
| small | 16² – 32² px |
| medium | 32² – 96² px |
| large | > 96² px |

**Why not native pixels.** AFO images are 2.5K–4K, so in native pixels 98.7% of objects are medium or large and the `tiny` bucket holds **15 objects in the entire dataset**. AP computed on 15 objects is noise, not a measurement. At the reference scale the buckets separate the data properly: the test split is 57.8% tiny / 38.4% small / 3.8% medium.

**Why not the model's own input resolution.** That was the first design and it was wrong. Scaling by `imgsz / max(W, H)` makes the bucket boundaries move with the run: an object that is `tiny` at 640 becomes `small` at 1280, so `AP-tiny` from the two runs describes two different sets of objects and cannot be compared. Since **comparing 640 against 1280 is the entire point of running both**, that choice destroyed the measurement it existed to make. A fixed scale keeps the population identical across every row, so the size columns rank models *and* resolutions on the same objects.

COCO's standard `AP-S` / `AP-M` / `AP-L` (native pixels, 32²/96² thresholds) are reported alongside for comparability with other papers, with the caveat that on AFO they degenerate to "almost everything is M or L".

### Operating point — the number AP hides

`mAP` integrates over the whole precision-recall curve. A deployed SAR system runs at **one** confidence threshold, and what decides whether an operator keeps the system switched on is how often it cries wolf at that threshold. AP cannot show this.

Every run therefore also reports `operating_point` in `metrics.json`: the confidence threshold maximising class-agnostic F1 at IoU 0.5, with precision, recall, F1, and **false alarms and misses per image** at that threshold.

Reported alongside the F1-optimal point:

- **False alarms per image at 80% and 90% recall.** F1 weights a miss and a false alarm equally; a missed person in the water does not cost the same as one spurious box, so the operator's question is "what does this cost me at the sensitivity I need". On the models measured so far neither target is reachable at all, which is itself the answer.
- **The recall ceiling** — the highest recall attainable as the threshold goes to zero. Objects above it are never detected at any confidence and no threshold tuning recovers them. On AFO at 640 this sits at **73–77%**, so between 1,700 and 2,100 of the 7,639 test objects are simply not found.

Two caveats that keep it honest. The thresholds are chosen on the *test* split, so these are upper bounds — a deployed threshold would be picked on val. And matching is class-agnostic, which means this diagnostic weights **objects** while mAP weights **classes**; with `human` at 85% of test objects the two can rank models differently, and both rankings are correct answers to different questions.

---

## 5. Hardware

**NVIDIA GeForce RTX 4090 Laptop GPU, 16 GB VRAM**, driver 617.14.

This is the **laptop** part, not a desktop 4090: **76 SMs against 128**, a lower power envelope, and subject to thermal throttling. Every FPS figure in the leaderboard is labelled with this exact name and with the resolution it was measured at. Runs are performed plugged in, on a performance power profile.

**Driver setting, required before any run:** NVIDIA Control Panel → Manage 3D Settings → **CUDA – Sysmem Fallback Policy → Prefer No Sysmem Fallback**. With the default policy an over-budget allocation silently spills to system RAM and streams over PCIe instead of raising `OutOfMemoryError` (§3). With it off, the failure is loud and immediate, and [`afo/vram.py`](afo/vram.py) becomes a second line of defence rather than the only one.

### FPS is measured in a separate pass, after a cool-down

A laptop GPU that has been training for hours is hot and running at reduced clocks. The same weights measured **43.8 FPS immediately after a 45-minute training run and 76.2 FPS on a cold card** — a 74% spread that says nothing about the model.

Discarding the first pass does not fix this: the second, third and fourth passes are hot too. So:

- FPS is **not** measured by the training scripts. They write `"speed": null` and move on.
- It is filled in afterwards by `python scripts/benchmark_fps.py --all`, which before each model waits for the GPU to drop below **50 °C** (up to 15 minutes) and then measures.
- **Temperature, SM clock and power are recorded in `metrics.json` next to the number**, before and after the measurement, so a reader can see the thermal state the figure came from.
- A spread above 15% across repeats raises a warning in the log: the card was still changing clocks and the figure should be re-taken.

Preconditions for a valid measurement: plugged in, high-performance power profile, no other GPU load.

### Measured epoch times and run order

Calibrated on this GPU against the pre-resized dataset (2 or 3 timed epochs each):

**Short calibrations overestimate badly — use completed runs.** Timing 2–3 epochs amortises dataset scanning, label caching and model setup over too few epochs. Calibration put `yolo11n` @640 at 78 s/epoch; the completed 100-epoch run came in at **26 s/epoch** (0.72 h total), a 3× overestimate. Every RF-DETR figure below is still from a 2-epoch calibration and is therefore an **upper bound** of unknown tightness.

| model | arm | imgsz | batch | s/epoch | source |
|---|---|---|---|---|---|
| `yolo11n` | 6-class | 640 | 64 | **26** | completed 100-epoch run (0.72 h) |
| `yolo11n` | 2-class | 640 | 64 | **23** | completed 100-epoch run (0.65 h) |
| `yolo11m` | — | 640 | 16 | 88 | 3-epoch calibration (upper bound) |
| `yolo11s` | — | 1280 | 8 | 166 | 3-epoch calibration (upper bound) |
| `rfdetr-nano` | — | 640 | 8 | 154 | 2-epoch calibration (upper bound) |
| `rfdetr-base` | — | 640 | 8 | 238 | 2-epoch calibration (upper bound) |

The RF-DETR base row also illustrates the VRAM rule above: the same model takes 238 s/epoch at batch 8, 437 s at batch 4 (GPU underused), and 2812 s at batch 16 (thrashing). Batch 8 is both the largest that fits and the fastest.

**Budget.** With two arms the matrix doubles, but the corrected YOLO timing more than pays for it: a YOLO run at 640 costs ~0.7 h, not the ~2.2 h first estimated. Twelve YOLO runs (six models × two arms) come to roughly **8–9 h**. The RF-DETR and 1280 rows dominate what remains and are still estimated from short calibrations.

Because this is a laptop that is also a working machine, runs go **one model at a time, lightest first**, with the heavy rows deferred to days when the machine is free:

1. YOLOv8 n/s/m and YOLO11 n/s/m at 640, both arms — ~0.7 h each, ~8–9 h total
2. RT-DETR-l at 640, RF-DETR nano at 640, best YOLO at 1280
3. RF-DETR small at 640
4. RF-DETR base at 640, then at 1280 — only on free days

The order is a scheduling decision and has no effect on any reported number; every run follows the identical protocol regardless of when it executes.

---

## 6. Outputs

**An interrupted run is not a lost run.** Ultralytics writes `best.pt` continuously and RF-DETR writes `checkpoint_best_ema.pth`; either is enough to produce every metric in this protocol without training again. Finish it with `python scripts/recover_run.py --run-all`, which finds runs that have weights but no `metrics.json` and completes them through `--post-only`.

**Never delete `results/_runs/<run>/` to "clean up" after an interruption.** The weights are the only expensive artefact a run produces — everything else is minutes of CPU. This rule is written down because it was learned the hard way: an `rtdetr-l` run stopped at epoch 88 of 100 was deleted as cleanup, discarding 4.7 h of GPU time that `--post-only` would have turned into a finished row in under a minute.

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
- mAP does penalise false alarms on water (§1) — but it is threshold-free. For the operating-point view, read `operating_point` in `metrics.json`: the F1-optimal threshold and the false alarms per image at it.
- FPS is from one laptop GPU; use it for relative comparison between rows, not as an absolute capability claim. Each row records the temperature and SM clock it was measured at.
- **At batch 1 the small models are overhead-bound, not compute-bound.** Preprocessing, the Python call and NMS dominate, so `n` and `s` can land within a few FPS of each other despite a 3.5× difference in parameters. That is a property of *this* measurement — batch 1, this GPU, this framework — and not a general claim that the bigger model is free. At larger batch, or on a device where compute dominates, the gap reappears. Read the FPS column as "what this laptop does, one frame at a time", which is the SAR case, and nothing wider.
- **A better model on mAP can be the worse model at an operating point.** `yolov8s` leads mAP50-95 and emits the most false alarms per image of the three measured so far. Decide which of the two questions you are asking before quoting a winner.

---

## 8. TODO — explicitly out of scope for this phase

- [ ] **Seed-variance study across the whole matrix.** Only the top model of each arm gets three seeds; the rest are single-seed, so mid-table gaps under ~1 point remain indistinguishable from noise.
- [ ] **A background-only evaluation set.** Not because false alarms are unmeasurable without it — they are measured, see §1 and the operating-point diagnostic — but because "false alarms per empty patrol frame" is the figure an operator reasons with, and reporting it directly beats estimating it. The 720 train background images are the obvious source, but using them deviates from the official split, so it needs its own protocol. A good requirement for a future AFO v2: empty frames in the test split.
- [ ] **A de-duplicated split variant**, quantifying what the 7 cross-split duplicate pairs (one of them val↔test) are worth.
- [ ] **A one-class arm** on `1category/`, the third label set the dataset ships. Pure "is there anything in the water", the weakest-supervision version of the task.
- [ ] **Cross-dataset generalisation** to SeaDronesSee and MOBDrone.
- [ ] **Edge inference** (Jetson / NPU) FPS.
- [ ] Higher resolutions than 1280, and tiled inference at native resolution.
