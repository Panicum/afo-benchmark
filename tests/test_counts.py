"""Testy spojnosci zbioru AFO — liczby z README musza sie zgadzac z danymi.

Wymagaja pobranego zbioru (``python -m afo.download`` albo AFO_ROOT).
"""

from __future__ import annotations

import warnings

import pytest

from afo.classes import CLASS_NAMES
from afo.dataset import SPLITS, AFODataset, all_image_names
from afo.download import dataset_root

EXPECTED_IMAGES = 3641
EXPECTED_OBJECTS = 39991
EXPECTED_SHARES = {"train": 67.42, "val": 13.48, "test": 19.10}
SHARE_TOL_PP = 0.1
EXPECTED_OUT_OF_RANGE_LINES = 29
UNLISTED_IMAGE = "i3_200.jpg"


@pytest.fixture(scope="module")
def root():
    try:
        return dataset_root()
    except FileNotFoundError as e:
        pytest.skip(str(e))


@pytest.fixture(scope="module")
def ds_all(root):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return AFODataset("all", root=root, clip=True)


@pytest.fixture(scope="module")
def splits(root):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {s: AFODataset(s, root=root, clip=True) for s in SPLITS}


def test_image_count(ds_all):
    assert len(ds_all) == EXPECTED_IMAGES


def test_object_count(ds_all):
    assert ds_all.num_objects() == EXPECTED_OBJECTS


def test_every_image_has_labels(ds_all):
    assert ds_all.missing_labels == [], f"obrazy bez etykiet: {ds_all.missing_labels}"


def test_split_lists_resolve_to_images(splits):
    for name, d in splits.items():
        assert d.missing_images == [], f"{name}: brak obrazow dla {d.missing_images}"


def test_split_object_shares(splits):
    total = sum(d.num_objects() for d in splits.values())
    for name, expected in EXPECTED_SHARES.items():
        share = 100.0 * splits[name].num_objects() / total
        assert abs(share - expected) <= SHARE_TOL_PP, (
            f"{name}: udzial obiektow {share:.2f}% != {expected}% (+-{SHARE_TOL_PP} pp)"
        )


def test_splits_cover_all_but_one_image(splits, ds_all):
    listed = set()
    for d in splits.values():
        listed.update(r.filename for r in d.records)
    missing = sorted(set(all_image_names(ds_all.root)) - listed)
    assert missing == [UNLISTED_IMAGE], f"obrazy spoza list podzialu: {missing}"


def test_splits_are_disjoint_by_filename(splits):
    names = [s for s in SPLITS]
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            sa = {r.filename for r in splits[a].records}
            sb = {r.filename for r in splits[b].records}
            assert not (sa & sb), f"{a} i {b} dziela nazwy: {sorted(sa & sb)[:10]}"


def test_unlisted_image_absent_from_all_splits(splits):
    for name, d in splits.items():
        assert UNLISTED_IMAGE not in {r.filename for r in d.records}, (
            f"{UNLISTED_IMAGE} jest w podziale {name}"
        )


def test_background_image_counts(splits):
    """Obrazy tla: 720 w train, 8 w val, 0 w test — test nie mierzy falszywych alarmow."""
    expected = {"train": 720, "val": 8, "test": 0}
    actual = {s: sum(1 for r in d.records if not r.annotations) for s, d in splits.items()}
    assert actual == expected


def test_total_background_images(ds_all):
    assert sum(1 for r in ds_all.records if not r.annotations) == 729


def test_out_of_range_coordinate_lines(ds_all):
    assert ds_all.out_of_range_count() == EXPECTED_OUT_OF_RANGE_LINES


def test_clipping_keeps_coords_in_unit_range(ds_all):
    for rec in ds_all.records:
        for a in rec.annotations:
            for v in (a.cx, a.cy, a.w, a.h):
                assert 0.0 <= v <= 1.0, f"{rec.filename}: {a}"


def test_no_clip_leaves_coords_untouched(root):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = AFODataset("all", root=root, clip=False)
    bad = sum(
        1
        for rec in raw.records
        for a in rec.annotations
        if any(v < 0.0 or v > 1.0 for v in (a.cx, a.cy, a.w, a.h))
    )
    assert bad == EXPECTED_OUT_OF_RANGE_LINES


def test_class_names_match_names_file(ds_all):
    assert tuple(ds_all.class_names) == CLASS_NAMES


def test_class_ids_within_range(ds_all):
    n = len(ds_all.class_names)
    for rec in ds_all.records:
        for a in rec.annotations:
            assert 0 <= a.class_id < n, f"{rec.filename}: class_id {a.class_id}"
