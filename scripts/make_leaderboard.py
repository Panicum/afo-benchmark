#!/usr/bin/env python
"""Buduje tabele wynikow z ``results/*/metrics.json`` i wstawia ja do README.

Kazda kolumna rozmiarowa niesie liczebnosc GT, a wiersze przy roznych imgsz sa
rozdzielone — kubelki AP-tiny/AP-small licza sie w pikselach wejscia modelu,
wiec miedzy rozdzielczosciami NIE sa porownywalne (patrz BENCHMARK.md).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

START = "<!-- leaderboard:start -->"
END = "<!-- leaderboard:end -->"

HF_URL = "https://huggingface.co/{repo}"


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


def build_table(rows: list[dict]) -> str:
    if not rows:
        return "*Brak wynikow — uruchom `scripts/train_ultralytics.py`.*"

    # Leaderboard jest 6-klasowy. Biegi na innej liczbie klas to inne zadanie
    # i nie wolno ich mieszac w jednej tabeli — raportujemy je osobno
    # (sekcja porownania z praca w README).
    main = [r for r in rows if r.get("categories", "6categories") == "6categories"]
    other = [r for r in rows if r.get("categories", "6categories") != "6categories"]
    if other:
        names = ", ".join(sorted({f"{r['model']}@{r['imgsz']} ({r['categories']})" for r in other}))
        print(f"Poza glowna tabela (inna liczba klas): {names}")
    if not main:
        return "*Brak wynikow 6-klasowych.*"
    rows = main

    by_imgsz: dict[int, list[dict]] = {}
    for r in rows:
        by_imgsz.setdefault(r.get("imgsz") or 0, []).append(r)

    out: list[str] = []
    for imgsz in sorted(by_imgsz):
        group = sorted(by_imgsz[imgsz], key=lambda r: -(r.get("mAP50_95") or -1))
        counts = (group[0].get("afo_buckets") or {})
        n_tiny = counts.get("tiny", {}).get("n_gt")
        n_small = counts.get("small", {}).get("n_gt")

        out.append(f"\n**imgsz = {imgsz}**\n")
        out.append(
            f"| model | mAP50 | mAP50-95 | AP-tiny (n={n_tiny or '?'}) | "
            f"AP-small (n={n_small or '?'}) | params (M) | FPS | wagi |"
        )
        out.append("|---|---|---|---|---|---|---|---|")
        for r in group:
            b = r.get("afo_buckets") or {}
            repo = r.get("weights_hf")
            link = f"[HF]({HF_URL.format(repo=repo)})" if repo else "—"
            fps = (r.get("speed") or {}).get("end_to_end", {}).get("fps")
            out.append(
                f"| `{r.get('model','?')}` | {fmt(r.get('mAP50'))} | "
                f"**{fmt(r.get('mAP50_95'))}** | "
                f"{fmt(b.get('tiny',{}).get('AP50_95'))} | "
                f"{fmt(b.get('small',{}).get('AP50_95'))} | "
                f"{fmt(r.get('params_M'), 2)} | {fmt(fps, 1)} | {link} |"
            )

    out.append(
        "\nSortowanie po mAP50-95 na tescie. Kubelki rozmiaru liczone w pikselach "
        "**wejscia modelu**, wiec AP-tiny przy 640 i przy 1280 opisuja **inne populacje "
        "obiektow** i nie wolno ich porownywac miedzy sekcjami. Jeden bieg na model, "
        "bez pomiaru rozrzutu po ziarnach — roznice ponizej ~1 punktu traktowac jak szum. "
        "FPS: batch 1, FP16, RTX 4090 **Laptop** GPU. Zadna kolumna nie mierzy falszywych "
        "alarmow (test nie ma obrazow tla) — patrz [BENCHMARK.md](BENCHMARK.md)."
    )
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--print-only", action="store_true")
    args = ap.parse_args()

    rows = load(Path(args.results_dir))
    table = build_table(rows)
    print(table)
    if args.print_only:
        return

    readme = Path(args.readme)
    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise SystemExit(f"Brak znacznikow {START} / {END} w {readme}")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    readme.write_text(f"{head}{START}\n{table}\n{END}{tail}", encoding="utf-8")
    print(f"\nZaktualizowano {readme.resolve()} ({len(rows)} wynikow)")


if __name__ == "__main__":
    main()
