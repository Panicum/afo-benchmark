# AFO — Aerial dataset of Floating Objects

**AFO** is a maritime search-and-rescue object detection dataset: **3,641 drone images** with **39,991 hand-annotated objects** in **6 classes** (human, wind/sup-board, boat, buoy, sailboat, kayak), captured over water from a UAV. Most objects are very small — the median object covers under a thousandth of the frame — which is the point of the dataset: it measures detectors where maritime SAR actually operates. It was released with Gąsienica-Józkowy, Knapik & Cyganek, *"An ensemble deep learning method with optimized weights for drone-based water rescue and surveillance"*, **Integrated Computer-Aided Engineering 28(3), 221–235, 2021**, [doi:10.3233/ICA-210649](https://doi.org/10.3233/ICA-210649). Canonical copy: [**🤗 `jgasienica/afo-dataset`**](https://huggingface.co/datasets/jgasienica/afo-dataset) · original release: [Kaggle](https://www.kaggle.com/datasets/jangsienicajzkowy/afo-aerial-dataset-of-floating-objects) (5,400+ downloads) · mirrors: [Voxel51/FiftyOne](https://docs.voxel51.com/), [Dataset Ninja](https://datasetninja.com/afo), [Roboflow Universe](https://universe.roboflow.com/).

This repository is the missing toolkit: a loader, a COCO export, a single shared evaluator, and reproducible baselines with a leaderboard. AFO has 140+ citations but surveys note it is used as a benchmark less than SeaDronesSee — because until now there was nothing to run and nothing to compare against.

![Six frames from AFO with ground-truth boxes: a 94-object crowd, a marina, a kayak on open water, a windsurf beach, an empty surf frame, and a harbour](docs/figures/samples.png)

> **No dataset files live in this repository.** Everything downloads from Hugging Face into `~/.cache/afo` (override with `AFO_ROOT`). The official splits and labels are never modified in place.

---

## Quick start

```bash
pip install -e .
```

```python
from afo import AFODataset

ds = AFODataset("test", download=True)     # downloads to ~/.cache/afo on first call
image, anns = ds[0]                        # PIL.Image + list[(class_id, cx, cy, w, h)]
print(ds, anns[0], ds.class_names[anns[0].class_id])
```

```
AFODataset(split='test', images=514, objects=7639, categories='6categories', clip=True)
Annotation(class_id=0, cx=0.7617, cy=0.9069, w=0.0559, h=0.0792) human
```

Export the official splits to COCO JSON (written to the cache, not the repo):

```bash
python scripts/export_coco.py --download          # -> ~/.cache/afo/coco/{train,val,test}.json
```

Reproduce the tables and figures below:

```bash
python scripts/stats.py                 # every number in this README
python scripts/make_figures.py          # every figure, into docs/figures/
```

---

## Classes

<!-- classes:start -->
| id | class | train | val | test | total | share |
|----|-------|-------|-----|------|-------|-------|
| 0 | `human` | 22,272 | 4,391 | 6,511 | **33,174** | 82.95% |
| 1 | `wind/sup-board` | 2,749 | 539 | 634 | **3,922** | 9.81% |
| 2 | `boat` | 506 | 53 | 143 | **702** | 1.76% |
| 3 | `bouy` | 428 | 100 | 59 | **587** | 1.47% |
| 4 | `sailboat` | 115 | 17 | 28 | **160** | 0.40% |
| 5 | `kayak` | 890 | 292 | 264 | **1,446** | 3.62% |
<!-- classes:end -->

Class `3` is spelled **`bouy`** in the released `6categories.names` — a typo in the original release, kept as-is because that file is the source of truth for class indices. `afo.display_name()` maps it to "buoy" for figures.

![Horizontal bar chart of objects per class: human 33,174 at 83%, then wind/sup-board 3,922, kayak 1,446, boat 702, buoy 587, sailboat 160](docs/figures/classes.png)

The dataset is **heavily imbalanced**: `human` is 83% of all objects and `sailboat` 0.4% — a 207× ratio. A model can score well on mAP while being useless on four of the six classes, so the benchmark reports per-class AP.

## Splits

The official splits ship with the dataset as `train.txt`, `validation.txt`, `test.txt` in `PART_1/PART_1/`. The **test set is built from nine videos that appear in neither train nor val**, so it measures generalisation to unseen scenes rather than to unseen frames of seen scenes — a split by sequence, not by frame. Keep these splits: cross-paper comparability is the whole reason they are fixed.

<!-- splits:start -->
| split | images | % images | objects | % objects | objects/image |
|-------|--------|----------|---------|-----------|---------------|
| train | 2,787 | 76.54% | 26,960 | 67.42% | 9.67 |
| val | 339 | 9.31% | 5,392 | 13.48% | 15.91 |
| test | 514 | 14.12% | 7,639 | 19.10% | 14.86 |
| **listed** | **3,640** | 99.97% | **39,991** | 100.00% | 10.99 |
| unlisted (`i3_200.jpg`) | 1 | 0.03% | — | — | — |
<!-- splits:end -->

The split is far more even **by object** (67.4 / 13.5 / 19.1) than **by image** (76.5 / 9.3 / 14.1). The apparent density gap — train 9.7 objects/image against val 15.9 and test 14.9 — is almost entirely **background images**, not busier scenes: counting only images that contain at least one object, the three splits sit at **13.0 / 16.3 / 14.9**. See the next note.

**729 images (20.0%) carry an empty label file — zero annotated objects.** They are deliberate negatives: open water, surf, beach. Their distribution across the splits is extremely uneven:

| split | images with 0 objects | share of split |
|-------|----------------------|----------------|
| train | 720 | **25.8%** |
| val | 8 | 2.4% |
| test | **0** | **0.0%** |

A quarter of the training set is background, so training gets substantial negative supervision for free. False alarms are still penalised on the test split — any detection not matching a box lowers precision, and with a median object at 0.085% of the frame the test set is overwhelmingly open water by area. What is not directly measured is false alarms per *empty* patrol frame; the benchmark estimates it from the annotated ones instead.

Object counts per image run from 0 to 94; 304 images hold 40 or more.

![Two charts: a histogram of objects per image with a long tail out to 94, and a bar chart of background-image share by split — 25.8% of train, 2.4% of val, 0% of test](docs/figures/density.png)

**Image resolutions:** 2560×1440 (46.5%), 3840×2160 (36.1%), 2720×1530 (14.8%), 1920×1080 (2.6%).

---

## Object size — read this before benchmarking

AFO objects are small **relative to the frame**, not small in absolute pixels. The median object is **81.6 px** across (√area) but covers only **0.085%** of its image, because the images are 2.5K–4K. The distinction decides how you report AP by size.

![The same four size buckets counted three ways: in native pixels almost everything is medium or large, at imgsz 640 62% is tiny, at imgsz 1280 54% is small](docs/figures/object-size.png)

In native pixels 98.7% of objects land in medium+large and the `tiny` bucket holds **15 objects in the whole dataset** — an AP-tiny column computed that way would be noise.

The benchmark therefore buckets objects at a **fixed reference scale** — every image rescaled to a 640 px longest side — which is why 62% of objects land in `tiny`. The scale is deliberately *not* the model's own input resolution: if it were, an object would be `tiny` at 640 and `small` at 1280, and the two runs could not be compared — which would defeat the purpose of running both. See [BENCHMARK.md](BENCHMARK.md) §4.

Object size per class (native px, median √area): `bouy` **35.9** · `human` **77.2** · `kayak` **102.0** · `boat` **125.5** · `wind/sup-board` **175.4** · `sailboat` **238.8**. The two smallest classes are also the two that the benchmark finds hardest — and `human` alone is 83% of all objects, so the dataset's difficulty and its imbalance point the same way. Full breakdown in [notebooks/01_explore.ipynb](notebooks/01_explore.ipynb).

---

## Data notes

These are verified properties of the dataset **as released**. They are documented here, not silently patched — the benchmark evaluates AFO as everyone else receives it.

- **3,641 images, not 3,647.** The paper reports 3,647; six images were never released. Every published count in this repository is computed from the files, not copied from the paper.
- **29 label lines have a coordinate outside `[0, 1]`.** Slightly-off hand annotations at frame edges. **Recommendation: clip on load.** `AFODataset(..., clip=True)` is the default and does exactly that, reporting the count; `clip=False` returns the raw values. The files on disk are never rewritten.
- **Class 3 is spelled `bouy` in `6categories.names`.** A typo in the original release. The loader reads the file verbatim so class indices stay correct; only display is corrected.
- **11 pairs of byte-identical images**, all consecutive `_0`/`_1` frames duplicated at capture. Seven straddle a split boundary, and **one of those is val↔test** (`r4_0.jpg` / `r4_1.jpg`) — a single test image of 514 has a twin in val, the only leak that can touch a reported score. **Keep the official splits anyway:** a private "clean" split makes your numbers incomparable. `scripts/stats.py` lists all eleven.
- **`i3_200.jpg` appears in no split list.** It is present in the images and has labels, but belongs to none of train/val/test. `AFODataset("all")` includes it; the three split loaders do not. It is excluded from training and from evaluation.
All of the above are asserted in [tests/test_counts.py](tests/test_counts.py), so a change to the upstream dataset breaks the tests rather than passing unnoticed.

---

## Benchmark

Reproducible baselines trained on the official train split, selected on val, reported on **test only**, no test-time augmentation, one shared pycocotools evaluator. Full protocol: [**BENCHMARK.md**](BENCHMARK.md). How to add your result: [CONTRIBUTING.md](CONTRIBUTING.md).

AFO ships two label sets, and they are **two different tasks**. This repository keeps a separate leaderboard for each, because a score on one says nothing about the other.

**18 runs are complete; the matrix is not.** Still missing: RF-DETR `small` and `base` in the 2-class arm, RT-DETR-l in both, 1280 for the best RF-DETR, and seed-variance runs. Inference speed is not benchmarked at all — the only GPU here is a shared laptop, and repeated measurements of the same weights varied by up to 55%. See [BENCHMARK.md](BENCHMARK.md).

### 1. Six classes — the harder task

`human`, `wind/sup-board`, `boat`, `bouy`, `sailboat`, `kayak`. No baselines for this version have been published before, which is the reason it is here: it is the setting for anyone who needs to know *what* is in the water, and it exposes the 207× class imbalance that the two-class version averages away.

<!-- leaderboard:start -->

**imgsz = 640**

| model | mAP50 | mAP50-95 | AP-tiny (n=4424) | AP-small (n=2930) | params (M) | weights |
|---|---|---|---|---|---|---|
| `rfdetr-base` | 0.7279 | **0.4080** | 0.2778 | 0.3537 | 31.87 | [HF](https://huggingface.co/jgasienica/afo-rfdetr-base-640) |
| `rfdetr-nano` | 0.7334 | **0.3937** | 0.2662 | 0.3478 | 30.56 | [HF](https://huggingface.co/jgasienica/afo-rfdetr-nano-640) |
| `rfdetr-small` | 0.6625 | **0.3584** | 0.2147 | 0.3276 | 32.03 | [HF](https://huggingface.co/jgasienica/afo-rfdetr-small-640) |
| `yolov8m` | 0.6442 | **0.3324** | 0.2450 | 0.3043 | 25.86 | [HF](https://huggingface.co/jgasienica/afo-yolov8m-640) |
| `yolo11m` | 0.6137 | **0.3170** | 0.2917 | 0.2876 | 20.06 | [HF](https://huggingface.co/jgasienica/afo-yolo11m-640) |
| `yolo11s` | 0.5290 | **0.2737** | 0.2820 | 0.3030 | 9.43 | [HF](https://huggingface.co/jgasienica/afo-yolo11s-640) |
| `yolov8s` | 0.5231 | **0.2630** | 0.2592 | 0.2848 | 11.14 | [HF](https://huggingface.co/jgasienica/afo-yolov8s-640) |
| `yolo11n` | 0.4716 | **0.2417** | 0.2222 | 0.2721 | 2.61 | [HF](https://huggingface.co/jgasienica/afo-yolo11n-640) |
| `yolov8n` | 0.4928 | **0.2401** | 0.2383 | 0.2560 | 3.14 | [HF](https://huggingface.co/jgasienica/afo-yolov8n-640) |

**imgsz = 1280**

| model | mAP50 | mAP50-95 | AP-tiny (n=4424) | AP-small (n=2930) | params (M) | weights |
|---|---|---|---|---|---|---|
| `yolov8m` | 0.7013 | **0.3749** | 0.2716 | 0.3172 | 25.86 | [HF](https://huggingface.co/jgasienica/afo-yolov8m-1280) |

Sorted by test mAP50-95. Size buckets use a **fixed reference scale** — every image rescaled to a 640 px longest side, independent of the training resolution — so AP-tiny means the same population of objects in every row and 640 is directly comparable with 1280. One run per model, no seed-variance estimate: treat gaps under ~1 point as noise. Inference speed is not benchmarked: the only available GPU is a shared laptop, where repeated measurements of the same weights varied by up to 55%. Per-run diagnostics beyond mAP are in `operating_point` in each `metrics.json` — see [BENCHMARK.md](BENCHMARK.md).
<!-- leaderboard:end -->

#### Beyond mAP

![Precision-recall curves for the best model of each family, and false alarms per image rising with recall](docs/figures/pr-curves.png)

**Every model leaves objects it can never find.** At a confidence threshold approaching zero the best reaches **87.4%** recall and the weakest 72.6% — between 965 and 2,095 of the 7,639 test objects are undetectable at any threshold. Only 3 of 9 models reach 80% recall at all; the right-hand panel shows what that sensitivity costs in false alarms. Per-run numbers are under `operating_point` in `metrics.json` (class-agnostic, IoU 0.5, thresholds chosen on test, so optimistic).

### 2. Two classes — continuing the published line

`small_obj` (human + buoy) against `large_obj` (board + boat + sailboat + kayak). This is the version the 2021 paper benchmarked, so a result here extends a line that already exists in the literature instead of starting a private one. It is also closer to how maritime SAR actually works: finding the object matters more than naming it.

```bash
python scripts/train_ultralytics.py --model yolo11n --imgsz 640 --categories 2categories
```

<!-- leaderboard2cat:start -->
| model | source | imgsz | AP | AP@50 | AP@50 `small_obj` | AP@50 `large_obj` | weights |
|---|---|---|---|---|---|---|---|
| SSD300 (MobileNet v2) | paper 2021 | 300 | 14.35 | 41.40 | 24.34 | 58.45 | — |
| YOLOv3 (Darknet53) | paper 2021 | 544 | 24.40 | 68.58 | 51.08 | 86.08 | — |
| `yolov8n` @640 | **this repo** | 640 | **29.71** | **67.45** | 44.23 | 90.67 | [HF](https://huggingface.co/jgasienica/afo-yolov8n-640-2cat) |
| `yolov8s` @640 | **this repo** | 640 | **31.43** | **70.14** | 46.24 | 94.05 | [HF](https://huggingface.co/jgasienica/afo-yolov8s-640-2cat) |
| YOLOv4 (CSPDarknet53-PANet-SPP) | paper 2021 | 544 | 31.50 | 71.13 | 54.58 | 87.67 | — |
| `yolo11n` @640 | **this repo** | 640 | **32.06** | **72.87** | 56.90 | 88.84 | [HF](https://huggingface.co/jgasienica/afo-yolo11n-640-2cat) |
| `yolo11s` @640 | **this repo** | 640 | **32.08** | **70.67** | 48.04 | 93.30 | [HF](https://huggingface.co/jgasienica/afo-yolo11s-640-2cat) |
| `yolov8m` @640 | **this repo** | 640 | **33.09** | **71.69** | 49.53 | 93.85 | [HF](https://huggingface.co/jgasienica/afo-yolov8m-640-2cat) |
| Faster R-CNN (R50-FPN) | paper 2021 | 1333 | 33.35 | 78.51 | 67.08 | 89.94 | — |
| `yolo11m` @640 | **this repo** | 640 | **33.49** | **72.97** | 54.87 | 91.07 | [HF](https://huggingface.co/jgasienica/afo-yolo11m-640-2cat) |
| RetinaNet (R50) | paper 2021 | 1333 | 34.93 | 79.82 | 69.31 | 90.31 | — |
| RetinaNet (R101) | paper 2021 | 1333 | 35.83 | 79.36 | 65.00 | 93.71 | — |
| Faster R-CNN (R101-FPN) | paper 2021 | 1333 | 36.25 | 78.97 | 64.11 | 93.82 | — |
| `yolo11m` @1280 | **this repo** | 1280 | **36.29** | **77.72** | 62.15 | 93.29 | [HF](https://huggingface.co/jgasienica/afo-yolo11m-1280-2cat) |
| `rfdetr-nano` @640 | **this repo** | 640 | **36.44** | **81.23** | 67.63 | 94.82 | [HF](https://huggingface.co/jgasienica/afo-rfdetr-nano-640-2cat) |
| Ensemble 3 (best AP@50) | paper 2021 | 1333 | 37.20 | 82.16 | 70.31 | 94.00 | — |
| Ensemble 4 (paper's contribution) | paper 2021 | 1333 | 37.71 | 81.55 | 68.39 | 94.71 | — |

AP in percent, as in the paper. `small_obj` = human + buoy, `large_obj` = board + boat + sailboat + kayak — a **semantic** merge, not a box-area threshold, so these columns are per-class AP and not COCO's AP-small/AP-large. Paper rows are reproduced from Table 2 of [doi:10.3233/ICA-210649](https://doi.org/10.3233/ICA-210649) for context; they were produced with different resolutions (300²–1333×750) and a fixed 20k-iteration schedule, so read them as historical reference rather than as rows of the same experiment.
<!-- leaderboard2cat:end -->

---

## How this compares to the original paper

**The paper's Table 2 measures a different thing.** It is computed on the **two-class** version of AFO, where `small_obj` = human + buoy and `large_obj` = the four vessel classes (paper §3.1). That is a *semantic* merge despite the names, so its `AP@50 Small Object` column is per-class AP for the merged class — not COCO's AP-small. Resolutions (544², 1333×750, 300²) and schedules (fixed 20k iterations) also differ from this protocol. The two-class leaderboard above interleaves both so the comparison sits in one place.

**Where this benchmark now stands against it.** `rfdetr-nano` at 640 reaches **36.44 AP / 81.23 AP@50**, passing every single model in Table 2 — including Faster R-CNN R101 (36.25 / 78.97) — and landing **1.3 AP short of the paper's best ensemble** (37.71). On `small_obj` it reaches 67.63 against RetinaNet-R50's 69.31, closing a gap that stood at 12.4 points when only YOLO models had been run.

**Verified against the release.** Every per-class count in the paper's Table 1 matches the files exactly, as do the split shares and the "over 99% of objects below 1% of image area" claim (measured 99.39%). Two things do not: the paper reports **3,647 images against 3,641 released** (the six missing must be background frames, since the object count matches), and *"more than 30% of images contain more than 20 instances"* measures **22.2%** — the 30% figure fits a threshold of 10 objects (30.8%), which looks like a transcription slip. Neither affects any conclusion in the paper.

Caveats on every comparison here: differing resolutions and schedules, and single-seed runs.

---

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
  publisher= {IOS Press},
  doi      = {10.3233/ICA-210649}
}
```

---

## License

- **Data** — [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/). Non-commercial, share-alike, attribution required. **Models trained on AFO inherit this restriction**; the checkpoints published with this benchmark are for research use only.
- **Code** in this repository — MIT.

See [LICENSE](LICENSE) for both texts. The sample frames in `docs/figures/` are drawn from the dataset and carry the data licence; they are regenerated by `scripts/make_figures.py` rather than curated by hand.

## Acknowledgements

Collected under Polish National Science Center grant **2016/21/B/ST6/01461**.

Footage by **Agnieszka Malonik Taggart**, **Wojciech Sulewski**, **Dariusz Nawrocki** and **Wojciech Kubiela**, whose drone material made the dataset possible.

The original experiments in the paper used [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet).

This toolkit and benchmark were built with [Claude Code](https://claude.com/claude-code).

## Author

**Jan Gąsienica-Józkowy** — [Google Scholar](https://scholar.google.com/citations?user=) · ORCID [0009-0009-7120-9083](https://orcid.org/0009-0009-7120-9083) · Hugging Face [`jgasienica`](https://huggingface.co/jgasienica)
