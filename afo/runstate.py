"""Stan biegu, ktory musi przezyc przerwanie i wznowienie.

Na razie jedno: skumulowany czas treningu. Bieg wznowiony liczy od zera czas
swojego odcinka, wiec bez kumulacji ``rfdetr-base`` zapisal 0,065 h zamiast
rzeczywistych ~3,65 h — pieciodziesieciokrotne zanizenie w kolumnie, ktora ma
mowic, ile kosztuje wytrenowanie modelu.
"""

from __future__ import annotations

import json
from pathlib import Path

STATE_FILE = "_runstate.json"


def _path(run_dir: Path | str) -> Path:
    return Path(run_dir) / STATE_FILE


def read_state(run_dir: Path | str) -> dict:
    p = _path(run_dir)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def accumulate_hours(run_dir: Path | str, hours: float | None) -> float | None:
    """Dodaje czas tego odcinka do sumy i zwraca sume calego biegu.

    ``hours=None`` (np. ``--post-only``) nic nie dodaje, tylko odczytuje sume —
    dzieki temu domkniecie biegu nie gubi czasu, ktory juz sie policzyl.
    """
    p = _path(run_dir)
    state = read_state(run_dir)
    segments: list[float] = list(state.get("train_hours_segments", []))
    if hours is not None:
        segments.append(round(float(hours), 4))
    total = round(sum(segments), 4) if segments else None

    state["train_hours_segments"] = segments
    state["train_hours_total"] = total
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return total


def set_segments(run_dir: Path | str, segments: list[float]) -> float:
    """Nadpisuje liste odcinkow — do uzupelnienia biegow sprzed kumulacji."""
    state = read_state(run_dir)
    state["train_hours_segments"] = [round(float(x), 4) for x in segments]
    state["train_hours_total"] = round(sum(state["train_hours_segments"]), 4)
    _path(run_dir).parent.mkdir(parents=True, exist_ok=True)
    _path(run_dir).write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state["train_hours_total"]
