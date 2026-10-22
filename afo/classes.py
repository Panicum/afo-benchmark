"""Klasy zbioru AFO — 6 kategorii w kolejnosci indeksow z 6categories.names."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .download import dataset_root, part_dir

# Kolejnosc i PISOWNIA dokladnie jak w pliku 6categories.names z oryginalnego
# wydania. Uwaga: w wydaniu jest literowka — klasa 3 to "bouy", nie "buoy".
# Zostaje tak, jak jest: plik .names jest zrodlem prawdy, a literowka jest
# udokumentowana (README, sekcja "Data notes"), nie poprawiana po cichu.
CLASS_NAMES: tuple[str, ...] = (
    "human",
    "wind/sup-board",
    "boat",
    "bouy",
    "sailboat",
    "kayak",
)

#: pisownia poprawiona — tylko do prezentacji (wykresy, tabele w README).
#: Nigdy nie uzywac jako kluczy do danych ani w eksporcie COCO.
DISPLAY_NAMES: dict[str, str] = {"bouy": "buoy"}

NUM_CLASSES = len(CLASS_NAMES)


def display_name(name: str) -> str:
    """Nazwa klasy do pokazania czlowiekowi (poprawiona literowka)."""
    return DISPLAY_NAMES.get(name, name)


def names_file(root: Path | str | None = None) -> Path:
    """Sciezka do 6categories.names w pobranym zbiorze."""
    r = dataset_root(root)
    return part_dir(r, 1) / "6categories.names"


@lru_cache(maxsize=4)
def load_class_names(root: str | None = None) -> tuple[str, ...]:
    """Czyta nazwy klas z pliku .names (zrodlo prawdy); fallback na CLASS_NAMES."""
    try:
        path = names_file(root)
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
        names = tuple(ln for ln in lines if ln)
        if names:
            return names
    except FileNotFoundError:
        pass
    return CLASS_NAMES
