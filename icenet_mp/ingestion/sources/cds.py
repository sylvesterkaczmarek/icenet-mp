import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, ClassVar

import earthkit.data as ekd
from anemoi.datasets.create.input.context import Context
from anemoi.datasets.create.source import Source
from anemoi.datasets.create.sources import source_registry
from anemoi.datasets.dates.groups import GroupOfDates
from earthkit.data.core.fieldlist import FieldList, MultiFieldList
from typing_extensions import override

logger = logging.getLogger(__name__)


@source_registry.register("cds")
class CDSSource(Source):
    """Load date-indexed fields from a Climate Data Store dataset."""

    date_keys: ClassVar[frozenset[str]] = frozenset(
        {"year", "month", "day", "time"}
    )
    area_size: ClassVar[int] = 4
    latitude_limit: ClassVar[float] = 90.0
    longitude_limit: ClassVar[float] = 180.0

    def __init__(
        self,
        context: Context,
        *,
        dataset: str,
        request: dict[str, Any],
    ) -> None:
        """Initialise the CDS source with a dataset name and base request."""
        self.context = context
        self.dataset = dataset
        self.request = dict(request)

        reserved = self.date_keys.intersection(self.request)
        if reserved:
            keys = ", ".join(sorted(reserved))
            msg = (
                f"CDS request date fields ({keys}) are derived from recipe dates "
                "and must not be configured explicitly."
            )
            raise ValueError(msg)

        area = self.request.get("area")
        if area is not None:
            self._validate_area(area)

    @classmethod
    def _validate_area(cls, area: Any) -> None:
        """Validate a CDS [north, west, south, east] bounding box."""
        if not isinstance(area, (list, tuple)) or len(area) != cls.area_size:
            msg = "CDS area must be [north, west, south, east]."
            raise ValueError(msg)
        north, west, south, east = map(float, area)
        if not (-cls.latitude_limit <= south <= north <= cls.latitude_limit):
            msg = f"Invalid north/south bounds in CDS area: {area}."
            raise ValueError(msg)
        if not (
            -cls.longitude_limit <= west < east <= cls.longitude_limit
        ):
            msg = f"Invalid west/east bounds in CDS area: {area}."
            raise ValueError(msg)

    def requests_for_dates(
        self, dates: list[datetime]
    ) -> list[dict[str, Any]]:
        """Build exact monthly CDS requests for the requested valid times."""
        grouped: dict[tuple[int, int], list[datetime]] = defaultdict(list)
        for date in sorted(dates):
            grouped[(date.year, date.month)].append(date)

        requests: list[dict[str, Any]] = []
        for (year, month), month_dates in grouped.items():
            request = dict(self.request)
            request.update(
                {
                    "year": [f"{year:04d}"],
                    "month": [f"{month:02d}"],
                    "day": sorted({date.strftime("%d") for date in month_dates}),
                    "time": sorted({date.strftime("%H:%M") for date in month_dates}),
                }
            )
            requests.append(request)
        return requests

    @override
    def execute(self, argument: list[datetime] | GroupOfDates) -> FieldList:
        """Retrieve requested dates from CDS and return them as one field list."""
        requests = self.requests_for_dates(list(argument))
        if not requests:
            return ekd.from_source("empty")

        field_lists: list[FieldList] = []
        for request in requests:
            logger.info(
                "Requesting %s for %s-%s (%d day(s)).",
                self.dataset,
                request["year"][0],
                request["month"][0],
                len(request["day"]),
            )
            fields = ekd.from_source("cds", self.dataset, request)
            if not isinstance(fields, FieldList):
                msg = (
                    f"CDS dataset {self.dataset!r} returned "
                    f"{type(fields).__name__}, expected gridded fields."
                )
                raise TypeError(msg)
            field_lists.append(fields)

        return MultiFieldList(field_lists)
