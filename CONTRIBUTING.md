# Adding a result to the AFO leaderboard

The leaderboard is only worth reading if every row was produced the same way. That is the whole point of [BENCHMARK.md](BENCHMARK.md), and it is the only thing enforced here.

**Results without a reproducible recipe are not merged.** Not because the number is doubted, but because a number nobody can re-derive is not a benchmark entry.

---

## What a submission contains

One directory, `results/<model>_<imgsz>/`, with four files:

| file | what it is |
|------|------------|
| `metrics.json` | output of `afo/evaluate.py` plus params, GFLOPs and speed |
| `predictions_test.json` | your raw COCO detection results on the **test** split |
| `train_config.yaml` | the fully resolved training configuration, not a summary |
| `log.txt` | the training and evaluation log |

Checkpoints are **not** committed. Upload them to Hugging Face and put the repo id in `metrics.json` under `weights_hf`.

---

## Steps

**1. Build the data exactly as the protocol specifies.**

```bash
python scripts/export_coco.py --download        # COCO JSON, coordinates clipped
python -m afo.yolo --max-side 1280              # Ultralytics layout
```

**2. Train.** Official train split, selection on val, test untouched until the end. Fixed seed, 100-epoch budget, patience 20, your framework's default augmentation.

```bash
python scripts/train_ultralytics.py --model yolo11s --imgsz 640
```

For a framework this repo does not wrap, train however you like — but the protocol constraints are not negotiable, and `train_config.yaml` has to show that you met them.

**3. Evaluate with the shared evaluator.** Do not report numbers from your framework's own validation routine. Ultralytics mAP and pycocotools mAP are not the same number, and a table mixing them is meaningless.

```bash
python -m afo.evaluate results/<run>/predictions_test.json --imgsz 640 \
    --out results/<run>/metrics.json
```

**4. Measure speed on your own hardware and say what it is.**

```bash
python scripts/benchmark_fps.py results/<run>/best.pt --imgsz 640
```

FPS is comparable only within one GPU. State the exact device; the existing rows are an RTX 4090 **Laptop** GPU and are labelled as such.

**5. Regenerate the table and open a PR.**

```bash
python scripts/make_leaderboard.py
```

---

## What gets a submission rejected

- **Numbers from a different evaluator.** Must be `afo/evaluate.py`.
- **A modified split.** No re-splitting, no de-duplication, no dropping the background images, no "cleaned" variant. Those are interesting experiments — they just are not this table. Open an issue instead.
- **Test-time augmentation, multi-scale inference or ensembling.** The protocol says none.
- **Test used for model selection.** Selection is on val. If you tried several configurations and picked by test score, the number is not comparable and the PR will say so.
- **No recipe.** A `metrics.json` without the configuration that produced it.
- **Training-set contamination.** If your COCO-pretrained weights were also fine-tuned on AFO or on data overlapping it, say so in the PR; it does not disqualify the entry, but it goes in a footnote.

## What is welcome

- New architectures, including ones nobody has tried on maritime SAR imagery.
- Results on other hardware — a second FPS column for a different GPU is useful.
- **Negative results.** A model that underperforms under this protocol is a real finding, and the table is more honest for containing it.
- Fixes to the protocol itself. If something in BENCHMARK.md measures the wrong thing, that is a bug worth more than any single row.

## Reporting a data problem

AFO as released has known issues — 29 out-of-range coordinate lines, 11 duplicate image pairs, 729 background images, a `bouy` typo. They are documented in the README and asserted in `tests/test_counts.py`, deliberately **not** fixed in place.

If you find something new, open an issue with the affected filenames. Do not submit a PR that edits the dataset: the benchmark evaluates AFO as everyone else receives it, and a silent fix here would make every published number incomparable.
