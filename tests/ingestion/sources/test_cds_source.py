import datetime
from unittest.mock import MagicMock

import earthkit.data as ekd
import pytest
from earthkit.data.core.fieldlist import FieldList, MultiFieldList

from icenet_mp.ingestion.sources.cds import CDSSource


def _source(request: dict | None = None) -> CDSSource:
    return CDSSource(
        MagicMock(),
        dataset="reanalysis-pan-carra",
        request=request
        or {
            "level_type": "single_levels",
            "variable": ["sea_ice_area_fraction"],
            "product_type": "analysis",
            "data_format": "grib",
            "download_format": "unarchived",
            "area": [81, 15, 76, 35],
        },
    )


class TestCDSSource:
    """Tests for generic CDS date grouping and ROI handling."""

    def test_builds_exact_monthly_requests(self) -> None:
        """Group by month without requesting unconfigured date combinations."""
        source = _source()
        requests = source.requests_for_dates(
            [
                datetime.datetime(2021, 1, 1, 12),
                datetime.datetime(2021, 1, 3, 12),
                datetime.datetime(2021, 2, 2, 12),
            ]
        )

        assert len(requests) == 2
        assert requests[0]["year"] == ["2021"]
        assert requests[0]["month"] == ["01"]
        assert requests[0]["day"] == ["01", "03"]
        assert requests[0]["time"] == ["12:00"]
        assert requests[1]["month"] == ["02"]
        assert requests[1]["day"] == ["02"]

    def test_rejects_explicit_date_fields(self) -> None:
        """Recipe dates are the single source of truth for CDS valid times."""
        with pytest.raises(ValueError, match="derived from recipe dates"):
            _source({"variable": ["sea_ice_area_fraction"], "year": ["2021"]})

    @pytest.mark.parametrize(
        "area",
        [
            [81, 15, 76],
            [70, 15, 81, 35],
            [81, 35, 76, 15],
        ],
    )
    def test_rejects_invalid_area(self, area: list[float]) -> None:
        """Reject malformed or inverted bounding boxes."""
        with pytest.raises(ValueError, match="area|bounds"):
            _source({"variable": ["sea_ice_area_fraction"], "area": area})

    def test_execute_fetches_each_month(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Execute one CDS request per month and combine returned fields."""
        source = _source()
        empty = ekd.from_source("empty")
        calls: list[tuple[str, str, dict]] = []

        def fake_from_source(name: str, dataset: str, request: dict) -> FieldList:
            calls.append((name, dataset, request))
            return empty

        monkeypatch.setattr(ekd, "from_source", fake_from_source)

        result = source.execute(
            [
                datetime.datetime(2021, 1, 1, 12),
                datetime.datetime(2021, 2, 1, 12),
            ]
        )

        assert isinstance(result, MultiFieldList)
        assert len(calls) == 2
        assert all(call[0] == "cds" for call in calls)
        assert all(call[1] == "reanalysis-pan-carra" for call in calls)
