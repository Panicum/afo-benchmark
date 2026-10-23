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


#: Wersja dwuklasowa z pracy: "human" i "buoy" -> small_obj, reszta -> large_obj.
#: Plik 1categories.names w wydaniu jest PUSTY, stad fallback takze dla niego.
FALLBACK_NAMES: dict[str, tuple[str, ...]] = {
    "6categories": CLASS_NAMES,
    "2categories": ("small_obj", "large_obj"),
    "1category": ("object",),
}

#: Z ktorych klas 6-kategoriowych powstaje kazda klasa wersji dwuklasowej
#: (Gasienica-Jozkowy i in. 2021, sekcja 3.1). Podzial jest SEMANTYCZNY,
#: a nie progiem na powierzchni ramki — mimo nazw "small"/"large".
TWO_CATEGORY_MAP: dict[str, tuple[str, ...]] = {
    "small_obj": ("human", "bouy"),
    "large_obj": ("wind/sup-board", "boat", "sailboat", "kayak"),
}


def names_file(root: Path | str | None = None, categories: str = "6categories") -> Path:
    """Sciezka do pliku .names odpowiadajacego danemu katalogowi etykiet."""
    r = dataset_root(root)
    return part_dir(r, 1) / f"{categories}.names"


@lru_cache(maxsize=8)
def load_class_names(root: str | None = None, categories: str = "6categories") -> tuple[str, ...]:
    """Czyta nazwy klas z pliku .names (zrodlo prawdy); fallback na stale wyzej."""
    try:
        path = names_file(root, categories)
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
        names = tuple(ln for ln in lines if ln)
        if names:
            return names
    except FileNotFoundError:
        pass
    return FALLBACK_NAMES.get(categories, CLASS_NAMES)
