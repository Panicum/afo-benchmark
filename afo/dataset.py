"""Loader zbioru AFO — oficjalne podzialy, etykiety Darknet YOLO.

Zasada: nic nie wypada po cichu. Jesli plik z listy podzialu nie ma obrazu albo
etykiety, loader mowi o tym Z NAZWY (atrybuty `missing_images`, `missing_labels`),
a nie tylko zmniejsza licznik.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterator, NamedTuple, Sequence

from .classes import load_class_names
from .download import dataset_root, part_dir

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

#: nazwa pliku z lista obrazow dla kazdego podzialu
SPLIT_FILES = {
    "train": "train.txt",
    "val": "validation.txt",
    "validation": "validation.txt",
    "test": "test.txt",
}

SPLITS = ("train", "val", "test")


class Annotation(NamedTuple):
    """Jeden obiekt: klasa + znormalizowany srodek i rozmiar (format Darknet)."""

    class_id: int
    cx: float
    cy: float
    w: float
    h: float

    def to_xyxy(self, img_w: int, img_h: int) -> tuple[float, float, float, float]:
        """Rogi w pikselach (x1, y1, x2, y2)."""
        return (
            (self.cx - self.w / 2) * img_w,
            (self.cy - self.h / 2) * img_h,
            (self.cx + self.w / 2) * img_w,
            (self.cy + self.h / 2) * img_h,
        )

    def to_xywh(self, img_w: int, img_h: int) -> tuple[float, float, float, float]:
        """Ramka w pikselach w konwencji COCO (x, y, szerokosc, wysokosc)."""
        x1, y1, x2, y2 = self.to_xyxy(img_w, img_h)
        return (x1, y1, x2 - x1, y2 - y1)


@dataclass
class Sample:
    """Rekord jednego obrazu — metadane bez wczytywania pikseli."""

    filename: str
    image_path: Path
    label_path: Path | None
    annotations: list[Annotation] = field(default_factory=list)
    n_clipped: int = 0

    def size(self) -> tuple[int, int]:
        """(szerokosc, wysokosc) z naglowka pliku, bez dekodowania obrazu."""
        from PIL import Image

        with Image.open(self.image_path) as im:
            return im.size

    def load_image(self):
        """Wczytuje obraz jako PIL.Image w RGB."""
        from PIL import Image

        with Image.open(self.image_path) as im:
            return im.convert("RGB")


def _parse_label_file(path: Path, clip: bool) -> tuple[list[Annotation], int]:
    """Czyta plik .txt Darknet. Zwraca (adnotacje, liczba przycietych wspolrzednych)."""
    anns: list[Annotation] = []
    n_clipped = 0
    if not path.is_file():
        return anns, 0
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            warnings.warn(f"{path}:{lineno}: oczekiwano 5 pol, jest {len(parts)} — linia pominieta")
            continue
        cid = int(float(parts[0]))
        vals = [float(v) for v in parts[1:]]
        out_of_range = sum(1 for v in vals if v < 0.0 or v > 1.0)
        if out_of_range and clip:
            vals = [min(1.0, max(0.0, v)) for v in vals]
        n_clipped += out_of_range
        anns.append(Annotation(cid, *vals))
    return anns, n_clipped


@lru_cache(maxsize=4)
def _image_index(root_str: str) -> dict[str, Path]:
    """Mapa nazwa pliku -> sciezka, zbudowana z PART_1..PART_3.

    Przy kolizji nazw miedzy czesciami wygrywa czesc o nizszym numerze, a kolizja
    jest zglaszana ostrzezeniem (nie jest to cicha podmiana).
    """
    root = Path(root_str)
    index: dict[str, Path] = {}
    for part in (1, 2, 3):
        img_dir = part_dir(root, part) / "images"
        if not img_dir.is_dir():
            warnings.warn(f"Brak katalogu obrazow: {img_dir}")
            continue
        for p in sorted(img_dir.iterdir()):
            if p.suffix not in IMAGE_EXTS:
                continue
            if p.name in index:
                warnings.warn(
                    f"Nazwa obrazu wystepuje w dwoch czesciach: {p.name} "
                    f"({index[p.name]} vs {p}) — uzywam pierwszej"
                )
                continue
            index[p.name] = p
    return index


def all_image_names(root: Path | str | None = None) -> list[str]:
    """Nazwy wszystkich obrazow w zbiorze (PART_1..PART_3), posortowane."""
    r = dataset_root(root)
    return sorted(_image_index(str(r)))


def read_split_list(split: str, root: Path | str | None = None) -> list[str]:
    """Nazwy obrazow z oficjalnej listy podzialu, w kolejnosci z pliku."""
    key = split.lower()
    if key not in SPLIT_FILES:
        raise ValueError(f"Nieznany podzial {split!r}; dozwolone: {sorted(set(SPLIT_FILES))}")
    r = dataset_root(root)
    path = part_dir(r, 1) / SPLIT_FILES[key]
    if not path.is_file():
        raise FileNotFoundError(f"Brak pliku listy podzialu: {path}")
    names: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip().replace("\\", "/")
        if not line:
            continue
        names.append(line.rsplit("/", 1)[-1])
    return names


class AFODataset(Sequence):
    """Zbior AFO dla jednego oficjalnego podzialu.

    Parametry
    ---------
    split : "train" | "val" | "test" | "all"
        "all" to wszystkie obrazy zbioru, takze te spoza list podzialu.
    root : sciezka do pobranego zbioru (domyslnie cache z afo.download)
    clip : przycinac wspolrzedne spoza [0,1] do [0,1] (domyslnie True)
    categories : katalog etykiet — "6categories" (domyslnie), "2categories", "1category"
    download : pobrac zbior, jesli go nie ma

    Iteracja zwraca krotke ``(PIL.Image, list[Annotation])``; metadane bez
    dekodowania pikseli sa w ``.records``.
    """

    def __init__(
        self,
        split: str = "train",
        root: Path | str | None = None,
        clip: bool = True,
        categories: str = "6categories",
        download: bool = False,
    ) -> None:
        if download:
            from .download import ensure_dataset

            ensure_dataset(root)

        self.root = dataset_root(root)
        self.split = split.lower()
        self.clip = clip
        self.categories = categories
        self.class_names = load_class_names(str(self.root), categories)
        self.label_dir = part_dir(self.root, 1) / categories

        index = _image_index(str(self.root))
        if self.split == "all":
            names = sorted(index)
        else:
            names = read_split_list(self.split, self.root)

        self.missing_images: list[str] = []
        self.missing_labels: list[str] = []
        self.records: list[Sample] = []
        self.n_clipped = 0

        for name in names:
            img_path = index.get(name)
            if img_path is None:
                self.missing_images.append(name)
                continue
            lbl_path = self.label_dir / (Path(name).stem + ".txt")
            if not lbl_path.is_file():
                self.missing_labels.append(name)
                lbl_path_or_none = None
                anns, nclip = [], 0
            else:
                lbl_path_or_none = lbl_path
                anns, nclip = _parse_label_file(lbl_path, clip)
            self.n_clipped += nclip
            self.records.append(
                Sample(
                    filename=name,
                    image_path=img_path,
                    label_path=lbl_path_or_none,
                    annotations=anns,
                    n_clipped=nclip,
                )
            )

        # Nic nie znika po cichu: ubytki raportowane z nazwy.
        if self.missing_images:
            warnings.warn(
                f"[AFO/{self.split}] {len(self.missing_images)} pozycji z listy podzialu "
                f"nie ma obrazu: {self.missing_images[:10]}"
                + (" ..." if len(self.missing_images) > 10 else "")
            )
        if self.missing_labels:
            warnings.warn(
                f"[AFO/{self.split}] {len(self.missing_labels)} obrazow bez pliku etykiet w "
                f"{categories}: {self.missing_labels[:10]}"
                + (" ..." if len(self.missing_labels) > 10 else "")
            )
        if self.clip and self.n_clipped:
            warnings.warn(
                f"[AFO/{self.split}] przycieto {self.n_clipped} wspolrzednych spoza [0,1] "
                f"(clip=True). Pliki zrodlowe nie sa zmieniane."
            )

    # --- protokol sekwencji -------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        rec = self.records[i]
        return rec.load_image(), rec.annotations

    def __iter__(self) -> Iterator:
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        return (
            f"AFODataset(split={self.split!r}, images={len(self)}, "
            f"objects={self.num_objects()}, categories={self.categories!r}, clip={self.clip})"
        )

    # --- pomocnicze ---------------------------------------------------------

    def num_objects(self) -> int:
        return sum(len(r.annotations) for r in self.records)

    def class_histogram(self) -> dict[str, int]:
        """Liczba obiektow na klase, w kolejnosci indeksow."""
        hist = {name: 0 for name in self.class_names}
        for rec in self.records:
            for a in rec.annotations:
                hist[self.class_names[a.class_id]] += 1
        return hist

    def out_of_range_count(self) -> int:
        """Liczba LINII etykiet z co najmniej jedna wspolrzedna spoza [0,1]."""
        n = 0
        for rec in self.records:
            if rec.label_path is None:
                continue
            for raw in rec.label_path.read_text(encoding="utf-8").splitlines():
                parts = raw.split()
                if len(parts) != 5:
                    continue
                if any(float(v) < 0.0 or float(v) > 1.0 for v in parts[1:]):
                    n += 1
        return n
