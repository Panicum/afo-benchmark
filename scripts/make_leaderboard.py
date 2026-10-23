#!/usr/bin/env python
"""Buduje tabele wynikow z ``results/*/metrics.json`` i wstawia je do README.

AFO ma DWA zadania i kazde ma wlasna tabele — nie wolno ich mieszac:

* **6 klas** — trudniejsze, nowe; to jest benchmark tego repozytorium.
* **2 klasy** (small_obj = human+buoy, large_obj = reszta) — schemat z Tabeli 2
  oryginalnej pracy, zeby wyniki byly porownywalne z opublikowanymi.

Kazda kolumna rozmiarowa niesie liczebnosc GT, a wiersze przy roznych imgsz sa
rozdzielone — kubelki AP-tiny/AP-small licza sie w pikselach wejscia modelu,
wiec miedzy rozdzielczosciami NIE sa porownywalne (patrz BENCHMARK.md).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MARKERS = {
    "6categories": ("<!-- leaderboard:start -->", "<!-- leaderboard:end -->"),
    "2categories": ("<!-- leaderboard2cat:start -->", "<!-- leaderboard2cat:end -->"),
}

HF_URL = "https://huggingface.co/{repo}"

#: Tabela 2 z pracy (Gasienica-Jozkowy i in. 2021), wersja DWUKLASOWA.
#: Wstawiana do tabeli 2-klasowej jako tlo historyczne, wyraznie oznaczona.
#: AP i AP@50 w procentach, jak w oryginale.
PAPER_ROWS = [
    ("SSD300 (MobileNet v2)", 300, 14.35, 41.40, 24.34, 58.45),
    ("YOLOv3 (Darknet53)", 544, 24.40, 68.58, 51.08, 86.08),
    ("YOLOv4 (CSPDarknet53-PANet-SPP)", 544, 31.50, 71.13, 54.58, 87.67),
    ("Faster R-CNN (R50-FPN)", 1333, 33.35, 78.51, 67.08, 89.94),
    ("RetinaNet (R50)", 1333, 34.93, 79.82, 69.31, 90.31),
    ("RetinaNet (R101)", 1333, 35.83, 79.36, 65.00, 93.71),
    ("Faster R-CNN (R101-FPN)", 1333, 36.25, 78.97, 64.11, 93.82),
    ("Ensemble 4 (paper's contribution)", 1333, 37.71, 81.55, 68.39, 94.71),
    ("Ensemble 3 (best AP@50)", 1333, 37.20, 82.16, 70.31, 94.00),
]


def fmt(v, nd=4, dash="—"):
    if v is None:
        return dash
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:  # NaN
        return dash
    return f"{f:.{nd}f}"


def load(results_dir: Path) -> list[dict]:
    rows = []
    for p in sorted(results_dir.glob("*/metrics.json")):
        try:
            rows.append(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError as e:
            print(f"UWAGA: pomijam {p} — nieczytelny JSON ({e})")
    return rows


def _fps_of(r: dict):
    return (r.get("speed") or {}).get("end_to_end", {}).get("fps")


def _weights_link(r: dict) -> str:
    repo = r.get("weights_hf")
    return f"[HF]({HF_URL.format(repo=repo)})" if repo else "—"


def build_six_class(rows: list[dict]) -> str:
    """Tabela glowna: 6 klas, sortowana po mAP50-95, rozdzielona po imgsz."""
    if not rows:
        return "*Brak wynikow 6-klasowych — uruchom `scripts/train_ultralytics.py`.*"

    by_imgsz: dict[int, list[dict]] = {}
    for r in rows:
        by_imgsz.setdefault(r.get("imgsz") or 0, []).append(r)

    out: list[str] = []
    for imgsz in sorted(by_imgsz):
        group = sorted(by_imgsz[imgsz], key=lambda r: -(r.get("mAP50_95") or -1))
        b0 = group[0].get("afo_buckets") or {}
        n_tiny = b0.get("tiny", {}).get("n_gt")
        n_small = b0.get("small", {}).get("n_gt")

        out.append(f"\n**imgsz = {imgsz}**\n")
        out.append(
            f"| model | mAP50 | mAP50-95 | AP-tiny (n={n_tiny or '?'}) | "
            f"AP-small (n={n_small or '?'}) | params (M) | FPS | weights |"
        )
        out.append("|---|---|---|---|---|---|---|---|")
        for r in group:
            b = r.get("afo_buckets") or {}
            out.append(
                f"| `{r.get('model','?')}` | {fmt(r.get('mAP50'))} | "
                f"**{fmt(r.get('mAP50_95'))}** | "
                f"{fmt(b.get('tiny',{}).get('AP50_95'))} | "
                f"{fmt(b.get('small',{}).get('AP50_95'))} | "
                f"{fmt(r.get('params_M'), 2)} | {fmt(_fps_of(r), 1)} | {_weights_link(r)} |"
            )

    out.append(
        "\nSorted by test mAP50-95. Size buckets use a **fixed reference scale** — every "
        "image rescaled to a 640 px longest side, independent of the training resolution — "
        "so AP-tiny means the same population of objects in every row and 640 is directly "
        "comparable with 1280. One run per model, no seed-variance estimate: treat gaps "
        "under ~1 point as noise. FPS: batch 1, FP16, RTX 4090 **Laptop** GPU, measured in "
        "a separate pass after a thermal cool-down with the temperature recorded alongside. "
        "mAP does penalise false alarms on water; for the operating-point view (F1-optimal "
        "threshold, false alarms per image) read `operating_point` in each `metrics.json` — "
        "see [BENCHMARK.md](BENCHMARK.md)."
    )
    return "\n".join(out)


def build_two_class(rows: list[dict], with_paper: bool = True) -> str:
    """Tabela dwuklasowa, z wierszami z pracy jako tlem."""
    ours = []
    for r in rows:
        pc = r.get("per_class") or {}
        ours.append({
            "name": f"`{r.get('model','?')}` @{r.get('imgsz')}",
            "imgsz": r.get("imgsz"),
            "ap": 100 * (r.get("mAP50_95") or 0),
            "ap50": 100 * (r.get("mAP50") or 0),
            "small": 100 * (pc.get("small_obj", {}).get("AP50") or 0),
            "large": 100 * (pc.get("large_obj", {}).get("AP50") or 0),
            "src": "**this repo**",
            "link": _weights_link(r),
        })

    paper = [
        {"name": n, "imgsz": z, "ap": ap, "ap50": ap50, "small": s, "large": l,
         "src": "paper 2021", "link": "—"}
        for n, z, ap, ap50, s, l in PAPER_ROWS
    ] if with_paper else []

    allrows = sorted(ours + paper, key=lambda d: d["ap"])
    if not ours:
        return "*Brak wynikow 2-klasowych — uruchom z `--categories 2categories`.*"

    out = [
        "| model | source | imgsz | AP | AP@50 | AP@50 `small_obj` | AP@50 `large_obj` | weights |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for d in allrows:
        mine = d["src"].startswith("**")
        ap = f"{d['ap']:.2f}"
        ap50 = f"{d['ap50']:.2f}"
        if mine:
            ap, ap50 = f"**{ap}**", f"**{ap50}**"
        out.append(
            f"| {d['name']} | {d['src']} | {d['imgsz']} | {ap} | {ap50} | "
            f"{d['small']:.2f} | {d['large']:.2f} | {d['link']} |"
        )
    out.append(
        "\nAP in percent, as in the paper. `small_obj` = human + buoy, `large_obj` = "
        "board + boat + sailboat + kayak — a **semantic** merge, not a box-area threshold, "
        "so these columns are per-class AP and not COCO's AP-small/AP-large. Paper rows are "
        "reproduced from Table 2 of [doi:10.3233/ICA-210649](https://doi.org/10.3233/ICA-210649) "
        "for context; they were produced with different resolutions (300²–1333×750) and a "
        "fixed 20k-iteration schedule, so read them as historical reference rather than as "
        "rows of the same experiment."
    )
    return "\n".join(out)


def splice(readme: Path, key: str, table: str) -> bool:
    start, end = MARKERS[key]
    text = readme.read_text(encoding="utf-8")
    if start not in text or end not in text:
        print(f"UWAGA: brak znacznikow {start} / {end} w {readme} — pomijam")
        return False
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    readme.write_text(f"{head}{start}\n{table}\n{end}{tail}", encoding="utf-8")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--print-only", action="store_true")
    ap.add_argument("--no-paper-rows", action="store_true")
    args = ap.parse_args()

    rows = load(Path(args.results_dir))
    six = [r for r in rows if r.get("categories", "6categories") == "6categories"]
    two = [r for r in rows if r.get("categories") == "2categories"]
    other = [r for r in rows if r.get("categories") not in (None, "6categories", "2categories")]
    if other:
        print(f"Pominiete (inna liczba klas): {[r.get('categories') for r in other]}")

    tables = {
        "6categories": build_six_class(six),
        "2categories": build_two_class(two, with_paper=not args.no_paper_rows),
    }
    for k, t in tables.items():
        print(f"\n===== {k} =====")
        print(t)

    if args.print_only:
        return
    readme = Path(args.readme)
    n = sum(splice(readme, k, t) for k, t in tables.items())
    print(f"\nZaktualizowano {readme.resolve()}: {len(six)} wynikow 6-klasowych, "
          f"{len(two)} dwuklasowych, {n}/2 tabel wstawionych")


if __name__ == "__main__":
    main()
