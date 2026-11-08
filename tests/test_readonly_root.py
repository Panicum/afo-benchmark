"""Zbior moze lezec w katalogu, do ktorego nie wolno pisac.

Tak jest na Kaggle: dane sa podmontowane w /kaggle/input/... tylko do odczytu, a
jedynym miejscem na wyniki jest /kaggle/working. Te testy pilnuja, zeby pakiet
niczego nie probowal zapisac do korzenia zbioru.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from afo import download
from afo.coco import export_coco
from afo.download import dataset_root, is_writable, looks_like_dataset, output_root


@pytest.fixture(scope="module")
def root():
    try:
        return dataset_root()
    except FileNotFoundError as e:
        pytest.skip(str(e))


def test_output_root_prefers_env(monkeypatch, tmp_path):
    monkeypatch.setenv("AFO_OUT", str(tmp_path))
    assert output_root() == tmp_path.resolve()


def test_output_root_uses_dataset_root_when_writable(monkeypatch, root):
    monkeypatch.delenv("AFO_OUT", raising=False)
    monkeypatch.setattr(download, "is_writable", lambda p: True)
    assert output_root(root) == Path(root)


def test_output_root_avoids_read_only_dataset_root(monkeypatch, root):
    """Korzen tylko do odczytu -> wyniki ida do katalogu roboczego, nie do zbioru."""
    monkeypatch.delenv("AFO_OUT", raising=False)
    monkeypatch.setattr(download, "is_writable", lambda p: False)
    out = output_root(root)
    assert out != Path(root)
    assert out == Path.cwd().resolve()


def test_is_writable_true_for_tmp(tmp_path):
    assert is_writable(tmp_path) is True


def test_is_writable_false_for_missing_dir(tmp_path):
    assert is_writable(tmp_path / "nie-ma-mnie") is False


def test_is_writable_gives_up_after_one_attempt(monkeypatch, tmp_path):
    """Odmowa zapisu ma dac odpowiedz, a nie petle.

    Pierwsza wersja uzywala ``tempfile``, ktory na Windowsie przy PermissionError
    sprawdza ``os.access`` (ignorujacy ACL) i probuje kolejnych nazw TMP_MAX razy.
    Na katalogu z odmowa zapisu wygladalo to na zawieszenie procesu.
    """
    import os as _os

    calls = []

    def deny(*a, **k):
        calls.append(a)
        raise PermissionError(13, "odmowa")

    monkeypatch.setattr(_os, "open", deny)
    assert is_writable(tmp_path) is False
    assert len(calls) == 1, f"probowano {len(calls)} razy zamiast raz"


def test_ensure_dataset_does_not_touch_disk_when_present(monkeypatch, root):
    """Zbior juz jest -> zadnego mkdir ani pobierania.

    Bez tego ``AFODataset(..., download=True)`` wywracal sie na podmontowanym
    zbiorze Kaggle: ensure_dataset robilo mkdir w katalogu tylko do odczytu.
    """

    def explode(*a, **k):
        raise AssertionError("snapshot_download nie powinno zostac wywolane")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", explode)
    monkeypatch.setattr(Path, "mkdir", lambda *a, **k: explode())
    assert download.ensure_dataset(root) == Path(root)


def test_looks_like_dataset(root, tmp_path):
    assert looks_like_dataset(root) is True
    assert looks_like_dataset(tmp_path) is False


def test_export_coco_writes_outside_the_dataset_root(monkeypatch, root, tmp_path):
    monkeypatch.setenv("AFO_OUT", str(tmp_path))
    before = {p.name for p in Path(root).iterdir()}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = export_coco("test", root=root, progress=False)

    assert out == tmp_path.resolve() / "coco" / "test.json"
    assert out.is_file()
    assert {p.name for p in Path(root).iterdir()} == before, "cos powstalo w korzeniu zbioru"


def test_export_coco_honours_explicit_out_path(root, tmp_path):
    target = tmp_path / "gdzie-indziej" / "test.json"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = export_coco("test", out_path=target, root=root, progress=False)
    assert out == target and target.is_file()
