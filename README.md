# AFO — Aerial dataset of Floating Objects

**AFO** is a maritime search-and-rescue object detection dataset: **3,641 drone images** with **39,991 hand-annotated objects** in **6 classes** (human, wind/sup-board, boat, buoy, sailboat, kayak), captured over water from a UAV. Most objects are very small — the median object covers under a thousandth of the frame — which is the point of the dataset: it measures detectors where maritime SAR actually operates. It was released with Gąsienica-Józkowy, Knapik & Cyganek, *"An ensemble deep learning method with optimized weights for drone-based water rescue and surveillance"*, **Integrated Computer-Aided Engineering 28(3), 221–235, 2021**, [doi:10.3233/ICA-210649](https://doi.org/10.3233/ICA-210649). Canonical copy: [**🤗 `jgasienica/afo-dataset`**](https://huggingface.co/datasets/jgasienica/afo-dataset) · original release: [Kaggle](https://www.kaggle.com/datasets/jangsienicajzkowy/afo-aerial-dataset-of-floating-objects) (5,400+ downloads) · mirrors: [Voxel51/FiftyOne](https://docs.voxel51.com/), [Dataset Ninja](https://datasetninja.com/afo), [Roboflow Universe](https://universe.roboflow.com/).

This repository is the missing toolkit: a loader, a COCO export, a single shared evaluator, and reproducible baselines with a leaderboard. AFO has 140+ citations but surveys note it is used as a benchmark less than SeaDronesSee — because until now there was nothing to run and nothing to compare against.

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

Reproduce the dataset tables below:

```bash
python scripts/stats.py
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

A quarter of the training set is background and the test set contains none. Two consequences, both of which the benchmark has to state rather than hide: training sees substantial negative supervision for free, and **the test split cannot measure false alarms on empty water** — a detector that fires constantly on wave crests is not penalised by any number on the leaderboard. For maritime SAR that is the error that costs most. A background-only evaluation set is listed as future work in [BENCHMARK.md](BENCHMARK.md).

Object counts per image run from 0 to 94; 304 images hold 40 or more.

**Image resolutions:** 2560×1440 (46.5%), 3840×2160 (36.1%), 2720×1530 (14.8%), 1920×1080 (2.6%).

---

## Object size — read this before benchmarking

AFO objects are small **relative to the frame**, not small in absolute pixels. The median object is **81.6 px** across (√area) but covers only **0.085%** of its image, because the images are 2.5K–4K. The distinction decides how you report AP by size.

| bucket | √area | native pixels | at `imgsz=640` | at `imgsz=1280` |
|--------|-------|---------------|----------------|-----------------|
| tiny | < 16² | 15 (0.04%) | 24,951 (62.4%) | 3,320 (8.3%) |
| small | 16²–32² | 509 (1.27%) | 12,917 (32.3%) | 21,631 (54.1%) |
| medium | 32²–96² | 25,533 (63.9%) | 2,075 (5.2%) | 14,795 (37.0%) |
| large | > 96² | 13,934 (34.8%) | 48 (0.12%) | 245 (0.6%) |

Measured in native pixels, 98.7% of objects fall in medium+large and the `tiny` bucket holds **15 objects in the entire dataset** — all of them in test. An AP-tiny column computed that way is noise, not a measurement.

Measured at the resolution the detector actually sees, the same buckets separate the data: feeding a 2560×1440 image to a 640 network shrinks every object ~4×, which is why 62% of objects are sub-16 px at the input and why 1280 is expected to matter. [BENCHMARK.md](BENCHMARK.md) states which convention the leaderboard uses.

Per-class size (native px, median / p95): `human` is the smallest class and the overwhelming majority of objects; `sailboat` and `boat` are the largest. Full breakdown in [notebooks/01_explore.ipynb](notebooks/01_explore.ipynb).

---

## Data notes

These are verified properties of the dataset **as released**. They are documented here, not silently patched — the benchmark evaluates AFO as everyone else receives it.

- **3,641 images, not 3,647.** The paper reports 3,647; six images were never released. Every published count in this repository is computed from the files, not copied from the paper.
- **29 label lines have a coordinate outside `[0, 1]`.** Slightly-off hand annotations at frame edges. **Recommendation: clip on load.** `AFODataset(..., clip=True)` is the default and does exactly that, reporting the count; `clip=False` returns the raw values. The files on disk are never rewritten.
- **Class 3 is spelled `bouy` in `6categories.names`.** A typo in the original release. The loader reads the file verbatim so class indices stay correct; only display is corrected.
- **11 pairs of byte-identical images; 7 of those pairs straddle a split boundary.** All 11 are consecutive `_0`/`_1` frames — the first two frames of a sequence, duplicated at capture:

  | pair | splits | |
  |------|--------|---|
  | `e_0.jpg` / `e_1.jpg` | val / train | leak |
  | `i1_0.jpg` / `i1_1.jpg` | val / train | leak |
  | `k1_0.jpg` / `k1_1.jpg` | val / train | leak |
  | `r2_0.jpg` / `r2_1.jpg` | val / train | leak |
  | `r4_0.jpg` / `r4_1.jpg` | val / test | leak |
  | `w1_0.jpg` / `w1_1.jpg` | val / train | leak |
  | `z2_0.jpg` / `z2_1.jpg` | val / train | leak |
  | `k5_0.jpg` / `k5_1.jpg` | test / test | within split |
  | `s1_0.jpg` / `s1_1.jpg`, `s2_*`, `s5_*` | train / train | within split |

  Six of the seven leaks are val↔train, which inflates val slightly but does not touch the reported numbers; **one, `r4_0`/`r4_1`, is val↔test**, so a single test image has a byte-identical twin in val. That is 1 image of 514 (0.19% of test) and it is the only leak that can affect a reported score. **Recommendation: keep the official splits** and treat this as a documented constant — a private "clean" split makes your numbers incomparable to everyone else's. A de-duplicated variant is future work in [BENCHMARK.md](BENCHMARK.md).
- **`i3_200.jpg` appears in no split list.** It is present in the images and has labels, but belongs to none of train/val/test. `AFODataset("all")` includes it; the three split loaders do not. It is excluded from training and from evaluation.
- **The classes are heavily imbalanced** (see the table above) and **objects are mostly tiny**, so mAP averaged over classes and over sizes hides most of what matters. The benchmark therefore reports per-class AP and AP in AFO-specific size buckets, not just mAP50-95.

All of the above are asserted in [tests/test_counts.py](tests/test_counts.py), so a change to the upstream dataset breaks the tests rather than passing unnoticed.

---

## Benchmark

Reproducible baselines trained on the official train split, selected on val, reported on **test only**, no test-time augmentation, one shared pycocotools evaluator. Full protocol: [**BENCHMARK.md**](BENCHMARK.md). How to add your result: [CONTRIBUTING.md](CONTRIBUTING.md).

<!-- leaderboard:start -->
*Phase B in progress — table appears here once the first models have trained.*
<!-- leaderboard:end -->

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

See [LICENSE](LICENSE) for both texts.

## Acknowledgements

Collected under Polish National Science Center grant **2016/21/B/ST6/01461**.

Footage by **Agnieszka Malonik Taggart**, **Wojciech Sulewski**, **Dariusz Nawrocki** and **Wojciech Kubiela**, whose drone material made the dataset possible.

The original experiments in the paper used [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet).

## Author

**Jan Gąsienica-Józkowy** — [Google Scholar](https://scholar.google.com/citations?user=) · ORCID [0009-0009-7120-9083](https://orcid.org/0009-0009-7120-9083) · Hugging Face [`jgasienica`](https://huggingface.co/jgasienica)
