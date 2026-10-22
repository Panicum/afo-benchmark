#!/usr/bin/env python
"""Statystyki zbioru AFO — tabele uzyte w README.

Liczy: obrazy i obiekty na podzial, udzialy procentowe, histogram klas,
rozmiary obiektow w kubelkach AFO (tiny/small/medium/large) oraz znane
nieprawidlowosci (wspolrzedne spoza [0,1], duplikaty bajtowe, obrazy spoza list).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from afo.dataset import SPLITS, AFODataset, all_image_names
from afo.download import dataset_root, ensure_dataset

#: kubelki rozmiaru obiektu wg pierwiastka z powierzchni w pikselach
SIZE_BUCKETS = (
    ("tiny", 0, 16),
    ("small", 16, 32),
    ("medium", 32, 96),
    ("large", 96, float("inf")),
)


def bucket_of(area_px: float) -> str:
    import math

    side = math.sqrt(max(area_px, 0.0))
    for name, lo, hi in SIZE_BUCKETS:
        if lo <= side < hi:
            return name
    return "large"


def fmt_table(rows: list[list[str]], headers: list[str]) -> str:
    """Prosta tabela markdown."""
    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]
    out = ["| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"]
    out.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for r in rows:
        out.append("| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(r))) + " |")
    return "\n".join(out)


def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--no-dupes", action="store_true", help="pomin skan duplikatow (wolny)")
    ap.add_argument("--json-out", default=None, help="zapisz statystyki jako JSON")
    args = ap.parse_args()

    if args.download:
        ensure_dataset(args.root)
    root = dataset_root(args.root)

    import warnings

    warnings.simplefilter("always")

    datasets = {s: AFODataset(s, root=root, clip=True) for s in SPLITS}
    ds_all = AFODataset("all", root=root, clip=True)
    class_names = ds_all.class_names

    total_imgs = len(ds_all)
    total_objs = ds_all.num_objects()

    print(f"\n# AFO — statystyki (root: {root})\n")
    print(f"Obrazow w zbiorze: {total_imgs}")
    print(f"Obiektow (6 klas): {total_objs}")
    print(f"Klasy ({len(class_names)}): {', '.join(class_names)}\n")

    # --- podzialy ---------------------------------------------------------
    rows = []
    for s in SPLITS:
        d = datasets[s]
        rows.append(
            [
                s,
                str(len(d)),
                f"{100 * len(d) / total_imgs:.2f}",
                str(d.num_objects()),
                f"{100 * d.num_objects() / total_objs:.2f}",
                f"{d.num_objects() / max(len(d), 1):.2f}",
            ]
        )
    split_img_sum = sum(len(datasets[s]) for s in SPLITS)
    split_obj_sum = sum(datasets[s].num_objects() for s in SPLITS)
    rows.append(
        [
            "**razem**",
            str(split_img_sum),
            f"{100 * split_img_sum / total_imgs:.2f}",
            str(split_obj_sum),
            f"{100 * split_obj_sum / total_objs:.2f}",
            f"{split_obj_sum / max(split_img_sum, 1):.2f}",
        ]
    )
    print("## Podzialy\n")
    print(
        fmt_table(
            rows,
            ["split", "obrazy", "% obrazow", "obiekty", "% obiektow", "obiektow/obraz"],
        )
    )

    # Obrazy tla (pusty plik etykiet) — rozklad miedzy podzialami jest bardzo
    # nierowny i zmienia interpretacje gestosci obiektow, wiec liczony osobno.
    print("\n### Obrazy bez obiektow (pusty plik etykiet)\n")
    rows = []
    for s in SPLITS:
        d = datasets[s]
        empty = [r for r in d.records if not r.annotations]
        nonempty = len(d) - len(empty)
        rows.append(
            [
                s,
                str(len(empty)),
                f"{100 * len(empty) / max(len(d), 1):.2f}",
                f"{d.num_objects() / max(len(d), 1):.2f}",
                f"{d.num_objects() / max(nonempty, 1):.2f}",
            ]
        )
    print(
        fmt_table(
            rows,
            ["split", "obrazow z 0 obiektow", "% podzialu", "obiektow/obraz", "bez pustych"],
        )
    )
    n_empty_all = sum(1 for r in ds_all.records if not r.annotations)
    print(f"\nW calym zbiorze: {n_empty_all} obrazow tla ({100 * n_empty_all / total_imgs:.2f}%).")
    print(
        "UWAGA: test nie zawiera ANI JEDNEGO obrazu tla — na tym zbiorze nie da sie"
        "\nzmierzyc falszywych alarmow na pustej wodzie."
    )
    counts_per_img = [len(r.annotations) for r in ds_all.records]
    print(
        f"Obiektow na obraz: min={min(counts_per_img)}, max={max(counts_per_img)}, "
        f"obrazow z >=40 obiektami: {sum(1 for c in counts_per_img if c >= 40)}"
    )

    # obrazy spoza list podzialu — raportowane z nazwy
    in_splits = set()
    for s in SPLITS:
        in_splits.update(r.filename for r in datasets[s].records)
    orphans = sorted(set(all_image_names(root)) - in_splits)
    print(f"\nObrazow spoza jakiejkolwiek listy podzialu: {len(orphans)}")
    if orphans:
        print(f"  {orphans}")

    # --- klasy ------------------------------------------------------------
    print("\n## Klasy\n")
    per_split = {s: datasets[s].class_histogram() for s in SPLITS}
    hist_all = ds_all.class_histogram()
    rows = []
    for i, name in enumerate(class_names):
        rows.append(
            [
                str(i),
                name,
                str(per_split["train"][name]),
                str(per_split["val"][name]),
                str(per_split["test"][name]),
                str(hist_all[name]),
                f"{100 * hist_all[name] / total_objs:.2f}",
            ]
        )
    print(fmt_table(rows, ["id", "klasa", "train", "val", "test", "razem", "% zbioru"]))

    # --- rozmiary obiektow -------------------------------------------------
    print("\n## Rozmiary obiektow (pierwiastek z powierzchni, piksele)\n")
    bucket_counts: dict[str, Counter] = {s: Counter() for s in SPLITS}
    all_sides: list[float] = []
    rel_areas: list[float] = []
    import math

    for s in SPLITS:
        for rec in datasets[s].records:
            w_img, h_img = rec.size()
            for a in rec.annotations:
                area = (a.w * w_img) * (a.h * h_img)
                bucket_counts[s][bucket_of(area)] += 1
                all_sides.append(math.sqrt(max(area, 0.0)))
                rel_areas.append(a.w * a.h)

    rows = []
    for name, lo, hi in SIZE_BUCKETS:
        tot = sum(bucket_counts[s][name] for s in SPLITS)
        rng = f"<{hi}^2" if lo == 0 else (f">{lo}^2" if hi == float("inf") else f"{lo}^2-{hi}^2")
        rows.append(
            [
                name,
                rng,
                *[str(bucket_counts[s][name]) for s in SPLITS],
                str(tot),
                f"{100 * tot / max(split_obj_sum, 1):.2f}",
            ]
        )
    print(fmt_table(rows, ["kubelek", "powierzchnia", "train", "val", "test", "razem", "%"]))

    # Te same kubelki, ale liczone w pikselach WEJSCIA MODELU, nie oryginalu.
    # To one decyduja o trudnosci detekcji: obraz 2560x1440 skalowany do 640
    # kurczy obiekt ~4x, do 1280 ~2x.
    print("\n### Te same kubelki po przeskalowaniu do wejscia modelu\n")
    rows = []
    for imgsz in (640, 1280):
        c_all: Counter = Counter()
        c_test: Counter = Counter()
        for s in SPLITS:
            for rec in datasets[s].records:
                w_img, h_img = rec.size()
                sc = imgsz / max(w_img, h_img)
                for a in rec.annotations:
                    bk = bucket_of((a.w * w_img * sc) * (a.h * h_img * sc))
                    c_all[bk] += 1
                    if s == "test":
                        c_test[bk] += 1
        n_all = max(sum(c_all.values()), 1)
        n_test = max(sum(c_test.values()), 1)
        for name, _, _ in SIZE_BUCKETS:
            rows.append(
                [
                    str(imgsz),
                    name,
                    str(c_all[name]),
                    f"{100 * c_all[name] / n_all:.2f}",
                    str(c_test[name]),
                    f"{100 * c_test[name] / n_test:.2f}",
                ]
            )
    print(fmt_table(rows, ["imgsz", "kubelek", "obiekty", "%", "test", "% test"]))
    print(
        "\nUWAGA: w pikselach oryginalu 98.7% obiektow wpada w medium+large, a kubelek"
        "\ntiny ma w CALYM zbiorze 15 obiektow (wszystkie w tescie) — AP-tiny liczone"
        "\nw tej skali jest szumem. Obiekty AFO sa male WZGLEDEM KADRU, nie w pikselach."
    )

    all_sides.sort()

    def pct(p: float) -> float:
        if not all_sides:
            return float("nan")
        return all_sides[min(len(all_sides) - 1, int(p * len(all_sides)))]

    print(
        f"\nBok obiektu (sqrt powierzchni), px: "
        f"p05={pct(0.05):.1f}  mediana={pct(0.50):.1f}  p95={pct(0.95):.1f}  max={all_sides[-1]:.1f}"
    )
    rel_areas.sort()
    print(
        f"Udzial powierzchni obiektu w obrazie: "
        f"mediana={100 * rel_areas[len(rel_areas) // 2]:.4f}%  "
        f"p95={100 * rel_areas[int(0.95 * len(rel_areas))]:.4f}%"
    )

    # --- rozdzielczosci obrazow -------------------------------------------
    res = Counter()
    for rec in ds_all.records:
        res[rec.size()] += 1
    print("\n## Rozdzielczosci obrazow\n")
    rows = [[f"{w}x{h}", str(n), f"{100 * n / total_imgs:.2f}"] for (w, h), n in res.most_common()]
    print(fmt_table(rows, ["rozdzielczosc", "obrazy", "%"]))

    # --- znane nieprawidlowosci -------------------------------------------
    print("\n## Znane nieprawidlowosci\n")
    oor_lines = 0
    oor_files: list[str] = []
    label_dir = ds_all.label_dir
    for rec in ds_all.records:
        if rec.label_path is None:
            continue
        hit = 0
        for raw in rec.label_path.read_text(encoding="utf-8").splitlines():
            parts = raw.split()
            if len(parts) != 5:
                continue
            if any(float(v) < 0.0 or float(v) > 1.0 for v in parts[1:]):
                hit += 1
        if hit:
            oor_lines += hit
            oor_files.append(f"{rec.filename}({hit})")
    print(f"Linii etykiet ze wspolrzedna spoza [0,1]: {oor_lines} w {len(oor_files)} plikach")
    if oor_files:
        print(f"  pliki: {', '.join(oor_files)}")

    dupe_info = None
    if not args.no_dupes:
        by_hash: dict[str, list[str]] = defaultdict(list)
        for rec in ds_all.records:
            by_hash[md5_of(rec.image_path)].append(rec.filename)
        groups = [sorted(v) for v in by_hash.values() if len(v) > 1]
        split_of = {}
        for s in SPLITS:
            for r in datasets[s].records:
                split_of[r.filename] = s
        cross = [g for g in groups if len({split_of.get(n) for n in g}) > 1]
        print(f"\nGrup obrazow identycznych bajtowo: {len(groups)}")
        print(f"  w tym rozlozonych na rozne podzialy (przeciek): {len(cross)}")
        for g in groups:
            tag = "/".join(str(split_of.get(n)) for n in g)
            mark = "  <-- PRZECIEK" if g in cross else ""
            print(f"    {g}  [{tag}]{mark}")
        dupe_info = {"groups": groups, "cross_split": cross}

    if args.json_out:
        out = {
            "root": str(root),
            "images": total_imgs,
            "objects": total_objs,
            "class_names": list(class_names),
            "splits": {
                s: {
                    "images": len(datasets[s]),
                    "objects": datasets[s].num_objects(),
                    "object_share_pct": round(100 * datasets[s].num_objects() / total_objs, 4),
                    "class_histogram": per_split[s],
                    "size_buckets": dict(bucket_counts[s]),
                }
                for s in SPLITS
            },
            "resolutions": {f"{w}x{h}": n for (w, h), n in res.most_common()},
            "out_of_range_label_lines": oor_lines,
            "out_of_range_files": oor_files,
            "images_in_no_split": orphans,
            "duplicates": dupe_info,
        }
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nZapisano JSON: {Path(args.json_out).resolve()}")


if __name__ == "__main__":
    main()
