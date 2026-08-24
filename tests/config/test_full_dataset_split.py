from datetime import date
from importlib.resources import files
from typing import cast

import yaml


def _load_split() -> dict[str, object]:
    path = files("icenet_mp.config") / "data" / "split" / "full_dataset.yaml"
    return cast("dict[str, object]", yaml.safe_load(path.read_text()))


def _ranges(split: dict[str, object], name: str) -> list[dict[str, object]]:
    return cast("list[dict[str, object]]", split[name])


def _contains_year(ranges: list[dict[str, object]], year: int) -> bool:
    """Return whether a calendar year intersects any configured date range."""
    for date_range in ranges:
        start = date_range["start"]
        end = date_range["end"]
        start_year = -10**9 if start is None else int(str(start)[:4])
        end_year = 10**9 if end is None else int(str(end)[:4])
        if start_year <= year <= end_year:
            return True
    return False


def test_full_dataset_uses_recorded_paper_holdout_years() -> None:
    """Reserve 2013 and 2023 exclusively for final paper evaluation."""
    split = _load_split()

    assert _ranges(split, "test") == [
        {"start": date(2013, 1, 1), "end": date(2013, 12, 31)},
        {"start": date(2023, 1, 1), "end": date(2023, 12, 31)},
    ]


def test_full_dataset_uses_2019_for_validation_and_2025_for_prediction() -> None:
    """Keep the development comparison year separate from future prediction data."""
    split = _load_split()

    assert _ranges(split, "validate") == [
        {"start": date(2019, 1, 1), "end": date(2019, 12, 31)}
    ]
    assert _ranges(split, "predict") == [
        {"start": date(2025, 1, 1), "end": date(2025, 12, 31)}
    ]


def test_full_dataset_training_ranges_match_recorded_partition() -> None:
    """Train on the remaining full-record periods identified in issue 184."""
    split = _load_split()

    assert _ranges(split, "train") == [
        {"start": None, "end": date(2012, 12, 31)},
        {"start": date(2014, 1, 1), "end": date(2018, 12, 31)},
        {"start": date(2020, 1, 1), "end": date(2022, 12, 31)},
        {"start": date(2024, 1, 1), "end": date(2024, 12, 31)},
    ]


def test_full_dataset_reserved_years_do_not_leak_between_splits() -> None:
    """Keep test, validation and future prediction years out of training."""
    split = _load_split()
    train = _ranges(split, "train")
    test = _ranges(split, "test")
    validate = _ranges(split, "validate")
    predict = _ranges(split, "predict")

    for year in (2013, 2019, 2023, 2025):
        assert not _contains_year(train, year)

    for year in range(1979, 2026):
        memberships = sum(
            _contains_year(ranges, year)
            for ranges in (train, test, validate, predict)
        )
        assert memberships == 1
