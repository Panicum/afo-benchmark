#!/usr/bin/env python
"""Generuje kaggle/afo-quickstart.ipynb.

Notebook powstaje ze skryptu, a nie z recznej edycji JSON-a: po kazdym probnym
uruchomieniu trzeba go odtworzyc z czystymi wyjsciami, a przy okazji latwo
pilnowac, zeby zadna komorka nie przekroczyla ~25 linii.

    python kaggle/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

MAX_CELL_LINES = 25

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


# --- 1. tytul -------------------------------------------------------------

md("""
# AFO — quick start, visualisation and benchmark baselines

**AFO (Aerial dataset of Floating Objects)** — 3,641 drone images, 39,991 hand-annotated objects,
6 classes, built for maritime search-and-rescue perception. Most objects are swimmers and boards
seen from tens of metres up, often a few dozen pixels across.

Paper: [doi:10.3233/ICA-210649](https://doi.org/10.3233/ICA-210649) ·
Benchmark: [github.com/Panicum/afo-benchmark](https://github.com/Panicum/afo-benchmark) ·
Canonical copy: [huggingface.co/datasets/jgasienica/afo-dataset](https://huggingface.co/datasets/jgasienica/afo-dataset)

Weights and the evaluator come from the benchmark repository; the numbers below reproduce two rows
of its leaderboard. No training happens here.
""")

# --- 2. setup -------------------------------------------------------------

md("## Setup")

code('''
import importlib.util
import subprocess
import sys

SPEC = "afo[baselines] @ git+https://github.com/Panicum/afo-benchmark"
missing = [m for m in ("afo", "ultralytics", "rfdetr") if importlib.util.find_spec(m) is None]
if missing:
    print("installing, missing:", ", ".join(missing))
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", SPEC], check=True)
else:
    print("already installed")
''')

code('''
import json
import os
import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

SEED = 0
random.seed(SEED)
np.random.seed(SEED)

# setdefault, zeby ten sam notebook dal sie uruchomic lokalnie z wlasnym AFO_ROOT
os.environ.setdefault("AFO_ROOT", "/kaggle/input/afo-aerial-dataset-of-floating-objects")
WORK = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()

from afo.classes import display_name
from afo.dataset import SPLITS, AFODataset
from afo.download import dataset_root

print("dataset:", dataset_root())
print("output :", WORK)
''')

code('''
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ds = {s: AFODataset(s) for s in SPLITS}

CLASSES = ds["train"].class_names            # pisownia z .names — zrodlo prawdy
LABELS = [display_name(c) for c in CLASSES]  # poprawiona, tylko na wykresy
PALETTE = plt.get_cmap("tab10").colors

for s in SPLITS:
    print(f"{s:6s} {len(ds[s]):5d} images  {ds[s].num_objects():6d} objects")
''')

code('''
def clip_of(rec):
    """Ujecie zrodlowe: 'a_339.jpg' -> 'a'."""
    return rec.filename.rsplit("_", 1)[0]


def sample_distinct_clips(records, n, seed=SEED):
    """n kadrow, kazdy z innego ujecia.

    Zbior powstal z 50 filmow, wiec losowanie wprost potrafi dac osiem niemal
    identycznych klatek jednego klipu — a wtedy siatka nie pokazuje zbioru,
    tylko jedno miejsce.
    """
    out, used = [], set()
    for rec in random.Random(seed).sample(records, len(records)):
        if clip_of(rec) in used:
            continue
        out.append(rec)
        used.add(clip_of(rec))
        if len(out) == n:
            break
    return out
''')

# --- 3. oglad danych ------------------------------------------------------

md("## 1. Look at the data")

code('''
def draw_split(split, n=8, ncols=4, seed=SEED):
    recs = sample_distinct_clips(ds[split].records, n, seed)
    nrows = -(-len(recs) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.8 * nrows))
    for ax, rec in zip(np.ravel(axes), recs):
        img = rec.load_image()
        W, H = img.size
        ax.imshow(img)
        for a in rec.annotations:
            x1, y1, x2, y2 = a.to_xyxy(W, H)
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=1.2,
                                   edgecolor=PALETTE[a.class_id % len(PALETTE)]))
        ax.set_title(f"{rec.filename} · {len(rec.annotations)} obj", fontsize=8)
        ax.axis("off")
    for ax in np.ravel(axes)[len(recs):]:
        ax.axis("off")
    handles = [Rectangle((0, 0), 1, 1, color=PALETTE[i]) for i in range(len(CLASSES))]
    fig.legend(handles, LABELS, loc="lower center", ncol=len(CLASSES), frameon=False)
    fig.suptitle(f"{split} — {n} random images, one per source clip", fontsize=13)
    # jawne odstepy: tight_layout przy axis("off") nie widzi tytulow i rzad drugi
    # nadpisywal obrazy z pierwszego
    fig.subplots_adjust(0.01, 0.10, 0.99, 0.90, wspace=0.03, hspace=0.16)
    plt.show()
''')

code('''
for s in SPLITS:
    draw_split(s)
''')

code('''
hist = {s: ds[s].class_histogram() for s in SPLITS}
totals = {c: sum(hist[s][c] for s in SPLITS) for c in CLASSES}
order = sorted(CLASSES, key=lambda c: -totals[c])
x = np.arange(len(order))

fig, ax = plt.subplots(figsize=(10, 4.5))
bottom = np.zeros(len(order))
for i, s in enumerate(SPLITS):
    vals = np.array([hist[s][c] for c in order], dtype=float)
    ax.bar(x, vals, bottom=bottom, label=s, color=PALETTE[i])
    bottom += vals
ax.set_yscale("log")
ax.set_xticks(x, [display_name(c) for c in order], rotation=20, ha="right")
ax.set_ylabel("objects (log scale)")
ax.set_title(f"Class imbalance: {totals[order[0]] / totals[order[-1]]:.0f}x "
             f"between the most and least frequent class")
ax.legend()
for xi, c in zip(x, order):
    ax.text(xi, totals[c], f"{totals[c]:,}", ha="center", va="bottom", fontsize=9)
fig.tight_layout()
plt.show()
''')

code('''
# udzial powierzchni ramki w powierzchni obrazu — wspolrzedne sa znormalizowane,
# wiec nie trzeba czytac rozmiarow plikow
rel = np.array([a.w * a.h for s in SPLITS for rec in ds[s].records for a in rec.annotations])
med = float(np.median(rel))

fig, ax = plt.subplots(figsize=(10, 4.5))
bins = np.logspace(np.log10(rel.min() * 100), np.log10(rel.max() * 100), 50)
ax.hist(rel * 100, bins=bins, color=PALETTE[0])
ax.set_xscale("log")
ax.axvline(med * 100, color="crimson", lw=2, label=f"median {med * 100:.4f}% of image area")
ax.set_xlabel("box area as % of image area (log scale)")
ax.set_ylabel("objects")
ax.set_title("Object size relative to the frame")
ax.legend()
fig.tight_layout()
plt.show()

print(f"objects      {len(rel):,}")
print(f"median       {med * 100:.4f}% of image area")
print(f"95th pct     {np.percentile(rel, 95) * 100:.4f}%")
''')

md("""
`human` accounts for 33,174 of 39,991 objects and `sailboat` for 160, a 207x gap, so a
class-averaged mAP is dominated by classes with almost no support. The median box covers a few
thousandths of a percent of the frame, which is why input resolution decides more here than
model size.
""")

# --- 4. COCO --------------------------------------------------------------

md("## 2. Export the test split to COCO")

code('''
from afo.coco import export_coco

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    coco_path = export_coco("test", out_path=WORK / "coco" / "test.json")

coco = json.loads(coco_path.read_text())
print(f"{len(coco['images'])} images, {len(coco['annotations'])} annotations")
print("first annotation:")
print(json.dumps(coco["annotations"][0], indent=2))
''')

# --- 5. odtworzenie wynikow ----------------------------------------------

md("""
## 3. Reproduce two rows of the leaderboard

The published numbers were measured on images rescaled to a 1280 px longest side, so that step is
reproduced here as well — predicting on the native 2.5K–4K files is a different resampling chain
and gives different numbers. Boxes are mapped back to the ground-truth coordinate space before
evaluation.
""")

code('''
from afo.yolo import prepare_images

TEST_1280 = WORK / "yolo1280" / "images" / "test"
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    prepare_images("test", TEST_1280, max_side=1280)
''')

code('''
from huggingface_hub import hf_hub_download

MODELS = {
    "yolo11n": {"repo": "jgasienica/afo-yolo11n-640", "file": "afo-yolo11n-640.pt"},
    "rfdetr-nano": {"repo": "jgasienica/afo-rfdetr-nano-640", "file": "afo-rfdetr-nano-640.pth"},
}
for name, m in MODELS.items():
    m["path"] = hf_hub_download(m["repo"], m["file"])
    # wartosci odniesienia czytamy z metrics.json W REPOZYTORIUM MODELU, a nie
    # przepisane recznie — porownanie ma dotyczyc tego, co opublikowane
    m["published"] = json.loads(Path(hf_hub_download(m["repo"], "metrics.json")).read_text())
    print(f"{name:12s} {Path(m['path']).name:26s} published mAP50-95 = {m['published']['mAP50_95']:.4f}")
''')

code('''
from afo.evaluate import evaluate
from afo.predict import predict_ultralytics

m = MODELS["yolo11n"]
pred = predict_ultralytics(m["path"], imgsz=640, out_path=WORK / "pred_yolo11n.json",
                           gt_json=coco_path, images_dir=TEST_1280)
m["result"] = evaluate(coco_path, pred, imgsz=640, class_names=CLASSES, verbose=False)
print(f"mAP50-95 {m['result']['mAP50_95']:.4f}   mAP50 {m['result']['mAP50']:.4f}")
''')

code('''
from afo.predict import predict_rfdetr

m = MODELS["rfdetr-nano"]
pred = predict_rfdetr(m["path"], size="nano", imgsz=640, out_path=WORK / "pred_rfdetr.json",
                      gt_json=coco_path, images_dir=TEST_1280, num_classes=len(CLASSES))
m["result"] = evaluate(coco_path, pred, imgsz=640, class_names=CLASSES, verbose=False)
print(f"mAP50-95 {m['result']['mAP50_95']:.4f}   mAP50 {m['result']['mAP50']:.4f}")
''')

code('''
def metrics_row(r):
    b = r["afo_buckets"]
    return [r["mAP50"], r["mAP50_95"], b["tiny"]["AP50_95"], b["small"]["AP50_95"]]


COLS = ["mAP50", "mAP50-95", "AP-tiny", "AP-small"]
head = f"{'model':14s} {'source':11s}" + "".join(f"{c:>10s}" for c in COLS)
print(head)
print("-" * len(head))
worst = 0.0
for name, m in MODELS.items():
    mine, pub = metrics_row(m["result"]), metrics_row(m["published"])
    for label, vals in (("this run", mine), ("published", pub)):
        print(f"{name:14s} {label:11s}" + "".join(f"{v:10.4f}" for v in vals))
    worst = max(worst, max(abs(a - b) for a, b in zip(mine, pub)))

print(f"\\nlargest absolute difference vs the published row: {worst:.6f}")
assert worst < 5e-5, "does not reproduce to 4 decimals — report it, do not adjust anything"
print("reproduced to 4 decimal places")
''')

# --- 6. jakosciowo --------------------------------------------------------

md("## 4. Ground truth against predictions")

code('''
sizes = {im["file_name"]: (im["width"], im["height"]) for im in coco["images"]}
id_of = {im["file_name"]: im["id"] for im in coco["images"]}

by_img = {}
for d in json.loads((WORK / "pred_rfdetr.json").read_text()):
    if d["score"] >= 0.25:
        by_img.setdefault(d["image_id"], []).append(d)


def median_side_at_640(rec):
    """Mediana sqrt(powierzchni) ramki po przeskalowaniu do 640 px dluzszego boku."""
    W, H = sizes[rec.filename]
    s = 640 / max(W, H)
    sides = [float(np.sqrt(a.w * W * a.h * H)) * s for a in rec.annotations]
    return float(np.median(sides)) if sides else np.inf
''')

code('''
# po jednym kadrze na ujecie: sasiednie klatki tego samego klipu sa niemal
# identyczne i zmarnowalyby panel
ranked = sorted((r for r in ds["test"].records if r.annotations), key=median_side_at_640)
rng = random.Random(SEED)
picked, used = [], set()
for rec in ranked[:2] + rng.sample(ranked, len(ranked)):  # 2 najdrobniejsze, potem losowo
    if clip_of(rec) in used:
        continue
    picked.append(rec)
    used.add(clip_of(rec))
    if len(picked) == 6:
        break

for r in picked:
    print(f"{r.filename:16s} {len(r.annotations):3d} obj   "
          f"median side at 640px: {median_side_at_640(r):5.1f}px")
''')

code('''
fig, axes = plt.subplots(2, 3, figsize=(18, 8))
for ax, rec in zip(np.ravel(axes), picked):
    img = rec.load_image()
    W, H = img.size
    ax.imshow(img)
    for a in rec.annotations:
        x1, y1, x2, y2 = a.to_xyxy(W, H)
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                               fill=False, lw=1.4, edgecolor="lime"))
    dets = by_img.get(id_of[rec.filename], [])
    for d in dets:
        x, y, w, h = d["bbox"]
        ax.add_patch(Rectangle((x, y), w, h, fill=False, lw=1.0, edgecolor="red"))
    ax.set_title(f"{rec.filename} · GT {len(rec.annotations)} · pred {len(dets)}", fontsize=9)
    ax.axis("off")
fig.suptitle("green = ground truth, red = RF-DETR nano at score >= 0.25", fontsize=13)
fig.tight_layout()
plt.show()
''')

# --- 7. zakonczenie -------------------------------------------------------

md("""
## Protocol and how to add a result

Full protocol: [BENCHMARK.md](https://github.com/Panicum/afo-benchmark/blob/main/BENCHMARK.md).

- Train on the official `train` split only, select on `val`, touch `test` once — the splits are
  used exactly as released.
- One shared pycocotools evaluator for every row, `maxDets=300` because a frame can hold 94 objects.
- Size buckets are measured at a fixed 640 px reference scale, so rows at different `imgsz` stay
  comparable.

Results are welcome as a pull request to the benchmark repository: a `metrics.json` produced by
that evaluator, plus the command that made it.

```bibtex
@article{gasienica2021afo,
  author  = {G{\\k{a}}sienica-J{\\'o}zkowy, Jan and Knapik, Mateusz and Cyganek, Bogus{\\l}aw},
  title   = {An ensemble deep learning method with optimized weights
             for drone-based water rescue and surveillance},
  journal = {Integrated Computer-Aided Engineering},
  year    = {2021}, volume = {28}, number = {3}, pages = {221--235},
  doi     = {10.3233/ICA-210649}
}
```

The data is **CC BY-NC-SA 3.0 — non-commercial use only**, and the published weights inherit that
licence because they were trained on it.
""")


def build() -> Path:
    cells = []
    for n, (kind, text) in enumerate(CELLS):
        lines = text.splitlines(keepends=True)
        if kind == "code" and len(lines) > MAX_CELL_LINES:
            raise SystemExit(f"komorka ma {len(lines)} linii (limit {MAX_CELL_LINES}):\n{text[:200]}")
        # nbformat >= 4.5 wymaga identyfikatora komorki; bez niego walidator
        # ostrzega, a w przyszlych wersjach ma to byc blad. Numer kolejny
        # wystarcza i jest stabilny miedzy odbudowami.
        cell = {"cell_type": kind, "id": f"afo-{n:02d}", "metadata": {}, "source": lines}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = Path(__file__).parent / "afo-quickstart.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    n_code = sum(1 for k, _ in CELLS if k == "code")
    longest = max(len(t.splitlines()) for k, t in CELLS if k == "code")
    print(f"{out}: {len(cells)} komorek ({n_code} kodu), najdluzsza {longest} linii")
    return out


if __name__ == "__main__":
    build()
