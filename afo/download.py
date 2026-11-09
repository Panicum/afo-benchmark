"""Pobranie zbioru AFO z Hugging Face do lokalnego cache.

Dane nigdy nie trafiaja do repozytorium. Domyslny cache: ~/.cache/afo,
nadpisywalny zmienna srodowiskowa AFO_ROOT.

Korzen zbioru moze byc TYLKO DO ODCZYTU - tak jest na Kaggle, gdzie zbior jest
podmontowany w /kaggle/input/... Dlatego zadna funkcja pakietu nie zapisuje do
korzenia: miejsce na wyniki daje `output_root()` (AFO_OUT, a w razie braku
katalog roboczy), a `ensure_dataset()` nie dotyka dysku, jesli zbior juz jest.
"""

from __future__ import annotations

import os
from pathlib import Path

HF_REPO_ID = "jgasienica/afo-dataset"


def default_root() -> Path:
    """Katalog, w ktorym ma lezec rozpakowany zbior."""
    env = os.environ.get("AFO_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / ".cache" / "afo").resolve()


def looks_like_dataset(path: Path | str) -> bool:
    """Czy pod ta sciezka lezy rozpakowany AFO (uklad z wydania Kaggle)."""
    return (Path(path) / "PART_1" / "PART_1" / "images").is_dir()


def is_writable(path: Path | str) -> bool:
    """Czy da sie zalozyc plik w tym katalogu.

    Probujemy naprawde zapisac, zamiast pytac ``os.access``: na Windowsie
    os.access patrzy na atrybut "tylko do odczytu", a nie na ACL, wiec dla
    katalogu z odmowa zapisu zwraca True.

    Jedna jawna proba, bez ``tempfile``: tempfile ma wlasna petle po nazwach i
    przy PermissionError sprawdza wlasnie ``os.access``, wiec na katalogu z
    odmowa zapisu krecil sie TMP_MAX razy zamiast zwrocic odpowiedz. Wygladalo
    to jak zawieszenie - sprawdzone na symulacji montowania Kaggle.
    """
    p = Path(path)
    if not p.is_dir():
        return False
    probe = p / f".afo-write-probe-{os.getpid()}"
    try:
        fd = os.open(probe, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except OSError:
        return False
    os.close(fd)
    try:
        probe.unlink()
    except OSError:
        pass
    return True


def output_root(root: Path | str | None = None) -> Path:
    """Katalog na pliki, ktore pakiet wytwarza (COCO JSON, dane dla trenerow).

    Kolejnosc: zmienna AFO_OUT, potem korzen zbioru jesli jest zapisywalny,
    a na koncu katalog roboczy. Na Kaggle korzen jest tylko do odczytu, a
    katalogiem roboczym jest /kaggle/working - czyli jedyne miejsce, ktore
    sesja zachowuje jako wynik.
    """
    env = os.environ.get("AFO_OUT")
    if env:
        return Path(env).expanduser().resolve()
    try:
        r = dataset_root(root)
    except FileNotFoundError:
        return default_root()
    return r if is_writable(r) else Path.cwd().resolve()


def ensure_dataset(
    root: Path | str | None = None,
    revision: str | None = None,
    force: bool = False,
) -> Path:
    """Pobiera snapshot zbioru z HF (o ile go nie ma) i zwraca sciezke do korzenia.

    Zwrocony katalog zawiera uklad 1:1 z oryginalnym wydaniem Kaggle:
        PART_1/PART_1/{images, 1category, 2categories, 6categories, *.names, *.txt}
        PART_2/PART_2/images
        PART_3/PART_3/images

    Jesli zbior juz tam lezy, funkcja nie tworzy katalogow i nie siega do sieci.
    Bez tego ``AFODataset(..., download=True)`` wywracal sie na podmontowanym
    zbiorze Kaggle, probujac zapisac do katalogu tylko do odczytu.
    """
    target = Path(root).expanduser().resolve() if root is not None else default_root()
    if not force and looks_like_dataset(target):
        return target

    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        revision=revision,
        local_dir=str(target),
    )
    return Path(path)


def dataset_root(root: Path | str | None = None) -> Path:
    """Sciezka do pobranego zbioru bez pobierania; blad, jesli go nie ma."""
    target = Path(root).expanduser().resolve() if root is not None else default_root()
    if not looks_like_dataset(target):
        raise FileNotFoundError(
            f"Nie znaleziono zbioru AFO w {target}. "
            f"Uruchom: python -c 'from afo.download import ensure_dataset; ensure_dataset()'"
        )
    return target


def part_dir(root: Path, part: int) -> Path:
    """Katalog danej czesci zbioru (PART_1..PART_3)."""
    return root / f"PART_{part}" / f"PART_{part}"


if __name__ == "__main__":
    print(ensure_dataset())
