#!/usr/bin/env python
"""Figury do README — przykladowe kadry i wykresy opisujace zbior.

Zasady kolorystyki (walidowane pod katem daltonizmu, OKLab dE >= 8 dla par
sasiednich): kolor niesie TOZSAMOSC klasy i jest ten sam na wszystkich
figurach, zeby czytelnik raz nauczyl sie, ze "human to niebieski".
Wykresy ilosciowe uzywaja jednego koloru — dlugosc slupka juz niesie wartosc.

Wyjscie: docs/figures/*.png (commitowane, male pliki).
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from afo.classes import display_name
from afo.dataset import SPLITS, AFODataset

# --- paleta ----------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

#: sloty kategorialne w kolejnosci indeksow klas AFO
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
PRIMARY = SERIES[0]

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": AXIS,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.titlecolor": INK,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


def _clean(ax, grid_axis: str | None = "y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis, zorder=0)
        ax.set_axisbelow(True)
    else:
        ax.grid(False)


# --- figura 1: przykladowe kadry -------------------------------------------

def fig_samples(ds_all: AFODataset, out: Path, n_cols: int = 3, n_rows: int = 2) -> Path:
    """Kadry dobrane tak, zeby pokazac wszystkie 6 klas i rozne warunki.

    Wybor jest deterministyczny: dla kazdej klasy bierzemy kadr, w ktorym ta
    klasa wystepuje najliczniej, a na koncu dokladamy najgestszy kadr w zbiorze.
    """
    names = list(ds_all.class_names)
    by_class: dict[int, tuple[int, object]] = {}
    densest: tuple[int, object | None] = (0, None)
    empty_frames = []
    for rec in ds_all.records:
        counts: dict[int, int] = {}
        for a in rec.annotations:
            counts[a.class_id] = counts.get(a.class_id, 0) + 1
        if len(rec.annotations) > densest[0]:
            densest = (len(rec.annotations), rec)
        if not rec.annotations:
            empty_frames.append(rec)
        for cid, c in counts.items():
            if cid not in by_class or c > by_class[cid][0]:
                by_class[cid] = (c, rec)

    # Najgestszy kadr idzie pierwszy — sam w sobie pokrywa klase "human".
    # Potem dokladamy po jednym kadrze na klase JESZCZE nieobecna, zeby nie
    # marnowac paneli na powtorzenie tej samej sceny, i na koniec jeden kadr
    # tla (20% zbioru to puste kadry — to tez trzeba pokazac).
    picks, seen, covered = [], set(), set()

    def take(rec) -> None:
        if rec is None or rec.filename in seen:
            return
        picks.append(rec)
        seen.add(rec.filename)
        covered.update(a.class_id for a in rec.annotations)

    take(densest[1])
    for cid in sorted(by_class, key=lambda c: by_class[c][0]):  # od najrzadszych
        if cid not in covered:
            take(by_class[cid][1])
    if empty_frames and len(picks) < n_cols * n_rows:
        take(empty_frames[len(empty_frames) // 2])
    for cid in sorted(by_class, key=lambda c: -by_class[c][0]):
        if len(picks) >= n_cols * n_rows:
            break
        take(by_class[cid][1])
    picks = picks[: n_cols * n_rows]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 2.9 * n_rows))
    for ax, rec in zip(np.ravel(axes), picks):
        img = rec.load_image()
        W, H = img.size
        ax.imshow(img)
        for a in rec.annotations:
            x1, y1, x2, y2 = a.to_xyxy(W, H)
            ax.add_patch(
                Rectangle(
                    (x1, y1), x2 - x1, y2 - y1, fill=False,
                    lw=1.1, edgecolor=SERIES[a.class_id % len(SERIES)],
                )
            )
        n_obj = len(rec.annotations)
        label = f"{n_obj} obj" if n_obj else "background — no objects"
        ax.set_title(
            f"{rec.filename} · {label} · {W}×{H}",
            fontsize=8.5, color=INK_2, pad=4,
        )
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    for ax in np.ravel(axes)[len(picks):]:
        ax.axis("off")

    handles = [
        plt.Line2D([], [], color=SERIES[i % len(SERIES)], lw=3.5, label=display_name(n))
        for i, n in enumerate(names)
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=len(names), frameon=False,
        fontsize=10, bbox_to_anchor=(0.5, -0.005),
    )
    fig.tight_layout(rect=[0, 0.055, 1, 1], h_pad=1.6)
    fig.savefig(out, dpi=85, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")
    return out


# --- figura 2: rozklad klas -------------------------------------------------

def fig_classes(ds: dict[str, AFODataset], out: Path) -> Path:
    names = list(ds["train"].class_names)
    totals = {n: sum(ds[s].class_histogram()[n] for s in SPLITS) for n in names}
    order = sorted(names, key=lambda n: totals[n])
    idx = {n: i for i, n in enumerate(names)}
    vals = [totals[n] for n in order]
    total = sum(vals)

    fig, ax = plt.subplots(figsize=(9, 3.6))
    y = np.arange(len(order))
    ax.barh(y, vals, height=0.62, color=[SERIES[idx[n] % len(SERIES)] for n in order], zorder=3)
    ax.set_yticks(y, [display_name(n) for n in order])
    _clean(ax, grid_axis="x")
    ax.set_xlabel("objects in the dataset")
    ax.set_xlim(0, max(vals) * 1.16)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}".replace(",", " "))
    )

    # kazdy slupek dostaje etykiete — przy 207x przewagi najmniejsze sa niewidoczne
    for yi, n in zip(y, order):
        v = totals[n]
        ax.text(
            v + max(vals) * 0.012, yi, f"{v:,}  ({100 * v / total:.1f}%)".replace(",", " "),
            va="center", ha="left", fontsize=9, color=INK_2,
        )

    ratio = max(totals.values()) / max(min(totals.values()), 1)
    ax.set_title(
        f"Class imbalance: {ratio:.0f}× between the most and least frequent class",
        fontsize=12, pad=10, loc="left",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")
    return out


# --- figura 3: rozmiar obiektow w trzech skalach ----------------------------

BUCKETS = [("tiny", 0, 16), ("small", 16, 32), ("medium", 32, 96), ("large", 96, np.inf)]


def fig_sizes(ds: dict[str, AFODataset], out: Path) -> Path:
    sides, maxdim = [], []
    for s in SPLITS:
        for rec in ds[s].records:
            W, H = rec.size()
            for a in rec.annotations:
                sides.append(math.sqrt(max((a.w * W) * (a.h * H), 0.0)))
                maxdim.append(max(W, H))
    sides = np.array(sides)
    maxdim = np.array(maxdim, dtype=float)

    views = [("native resolution", None), ("model input, imgsz = 640", 640),
             ("model input, imgsz = 1280", 1280)]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), sharey=True)
    labels = [b[0] for b in BUCKETS]
    x = np.arange(len(labels))

    for ax, (title, imgsz) in zip(axes, views):
        sc = sides if imgsz is None else sides * (imgsz / maxdim)
        counts = [int(((sc >= lo) & (sc < hi)).sum()) for _, lo, hi in BUCKETS]
        ax.bar(x, counts, width=0.6, color=PRIMARY, zorder=3)
        for xi, v in zip(x, counts):
            ax.text(
                xi, v + len(sides) * 0.015,
                f"{100 * v / len(sides):.1f}%", ha="center", va="bottom",
                fontsize=9.5, color=INK_2,
            )
        ax.set_xticks(x, labels)
        ax.set_title(title, fontsize=11, color=INK, pad=8)
        _clean(ax, grid_axis="y")
        ax.set_ylim(0, len(sides) * 0.80)

    axes[0].set_ylabel("objects")
    axes[0].set_yticks(
        [0, 10000, 20000, 30000], ["0", "10 000", "20 000", "30 000"]
    )
    fig.suptitle(
        "AFO objects are small relative to the frame, not in absolute pixels — "
        "the same buckets, measured three ways",
        fontsize=12.5, y=1.03, x=0.5, ha="center", color=INK,
    )
    fig.text(
        0.5, -0.07,
        "Bucket = √(box area). In native pixels the “tiny” bucket holds 15 objects in the whole "
        "dataset; rescaled to what the detector sees, it holds 62%.",
        ha="center", fontsize=9.5, color=MUTED,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")
    return out


# --- figura 4: obiekty na obraz + obrazy tla --------------------------------

def fig_density(ds: dict[str, AFODataset], out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))

    counts = [len(r.annotations) for s in SPLITS for r in ds[s].records]
    ax = axes[0]
    ax.hist(counts, bins=np.arange(0, max(counts) + 3, 2), color=PRIMARY, zorder=3)
    _clean(ax, grid_axis="y")
    ax.set_xlabel("objects in one image")
    ax.set_ylabel("images")
    ax.set_title(
        f"Crowded frames: up to {max(counts)} objects in a single image",
        fontsize=11, loc="left", pad=8,
    )

    ax = axes[1]
    empty = {s: sum(1 for r in ds[s].records if not r.annotations) for s in SPLITS}
    tot = {s: len(ds[s]) for s in SPLITS}
    x = np.arange(len(SPLITS))
    share = [100 * empty[s] / tot[s] for s in SPLITS]
    ax.bar(x, share, width=0.55, color=PRIMARY, zorder=3)
    for xi, s in zip(x, SPLITS):
        ax.text(
            xi, share[SPLITS.index(s)] + 0.7,
            f"{empty[s]} of {tot[s]}", ha="center", va="bottom", fontsize=9.5, color=INK_2,
        )
    ax.set_xticks(x, list(SPLITS))
    ax.set_ylabel("% of split")
    ax.set_ylim(0, max(share) * 1.3)
    _clean(ax, grid_axis="y")
    ax.set_title(
        "Background images (no objects) — and none at all in test",
        fontsize=11, loc="left", pad=8,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="docs/figures")
    ap.add_argument("--root", default=None)
    ap.add_argument("--skip-samples", action="store_true", help="pomin (wolne, czyta obrazy)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    warnings.simplefilter("ignore")

    print("wczytywanie zbioru...")
    ds = {s: AFODataset(s, root=args.root) for s in SPLITS}
    print("figury:")
    fig_classes(ds, out / "classes.png")
    fig_sizes(ds, out / "object-size.png")
    fig_density(ds, out / "density.png")
    if not args.skip_samples:
        fig_samples(AFODataset("all", root=args.root), out / "samples.png")


if __name__ == "__main__":
    main()
