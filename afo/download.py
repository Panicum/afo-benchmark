"""Pobranie zbioru AFO z Hugging Face do lokalnego cache.

Dane nigdy nie trafiaja do repozytorium. Domyslny cache: ~/.cache/afo,
nadpisywalny zmienna srodowiskowa AFO_ROOT.
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


def ensure_dataset(root: Path | str | None = None, revision: str | None = None) -> Path:
    """Pobiera snapshot zbioru z HF (o ile go nie ma) i zwraca sciezke do korzenia.

    Zwrocony katalog zawiera uklad 1:1 z oryginalnym wydaniem Kaggle:
        PART_1/PART_1/{images, 1category, 2categories, 6categories, *.names, *.txt}
        PART_2/PART_2/images
        PART_3/PART_3/images
    """
    from huggingface_hub import snapshot_download

    target = Path(root).expanduser().resolve() if root is not None else default_root()
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
    if not (target / "PART_1" / "PART_1" / "images").is_dir():
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
