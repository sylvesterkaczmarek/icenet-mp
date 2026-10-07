#!/usr/bin/env python3
# Verification assertions and progress output are intentional; see README.md.
# ruff: noqa: PLC0415, PLR0915, PT018, S101, T201
"""Exhaustive read-only verification of Anemoi's native legacy-calendar adapter."""

import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path


def main(data_root: Path, output_file: Path) -> None:
    """Compare every observed date, value and stored statistic with the reader."""
    import numpy as np
    import zarr
    from anemoi.datasets import __version__ as anemoi_version
    from anemoi.datasets import open_dataset

    assert os.environ.get("ANEMOI_DATASETS_MISSING_DATES_FIX_EXPERIMENTAL") == "1"
    filenames = [
        "samp-sicsouth-osisaf-25p0km-2020-2024-24h-v1.zarr",
        "samp-floatsouth-argo-25p0km-2020-2024-24h-v3.zarr",
        "samp-weathersouth-era5-25p0km-2020-2024-24h-v4.zarr",
    ]
    manifest = [{"path": str(data_root / name)} for name in filenames]
    analytics_path = Path.home() / ".config/anemoi/analytics.json"
    options = json.loads(analytics_path.read_text()) if analytics_path.exists() else {}
    assert not options.get("enabled", False), (
        "Anemoi analytics must be disabled for the local verification."
    )
    results = []
    started = time.perf_counter()
    for record in manifest:
        path = Path(record["path"])
        attrs_hash = hashlib.sha256((path / ".zattrs").read_bytes()).hexdigest()
        raw = zarr.open_group(str(path), mode="r")
        stored_dates = raw["dates"][:].astype("datetime64[s]")
        assert len(np.unique(stored_dates)) == len(stored_dates)
        assert np.all(np.diff(stored_dates) > np.timedelta64(0, "s"))
        assert raw.attrs["frequency"] == "1d"
        calendar = np.arange(
            stored_dates[0],
            stored_dates[-1] + np.timedelta64(1, "D"),
            np.timedelta64(1, "D"),
        )
        missing = np.array(raw.attrs["missing_dates"], dtype="datetime64[s]")
        expected_missing = np.setdiff1d(calendar, stored_dates)
        np.testing.assert_array_equal(np.sort(missing), expected_missing)
        indices = np.searchsorted(calendar, stored_dates)
        np.testing.assert_array_equal(calendar[indices], stored_dates)
        reader = open_dataset(str(path))
        np.testing.assert_array_equal(reader.dates, calendar)
        assert set(reader.missing) == set(
            np.searchsorted(calendar, expected_missing).tolist()
        )
        assert reader.shape == (len(calendar), *raw["data"].shape[1:])
        assert list(reader.variables) == list(raw.attrs["variables"])
        stored_digest = hashlib.sha256()
        adapted_digest = hashlib.sha256()
        values_checked = 0
        for stored_index, calendar_index in enumerate(indices):
            original = raw["data"][stored_index]
            adapted = reader[int(calendar_index)]
            assert np.array_equal(original, adapted, equal_nan=True), (
                path.name,
                stored_index,
            )
            assert original.dtype == adapted.dtype and original.shape == adapted.shape
            stored_digest.update(memoryview(np.ascontiguousarray(original)))
            adapted_digest.update(memoryview(np.ascontiguousarray(adapted)))
            values_checked += original.size
            if stored_index % 128 == 0:
                print("VERIFY", path.name, stored_index, "/", len(indices), flush=True)
        assert stored_digest.hexdigest() == adapted_digest.hexdigest()
        statistics_hashes = {}
        for name in ["minimum", "maximum", "mean", "stdev"]:
            original = raw[name][:]
            adapted = reader.statistics[name]
            np.testing.assert_array_equal(original, adapted)
            statistics_hashes[name] = hashlib.sha256(
                np.ascontiguousarray(original).tobytes()
            ).hexdigest()
        assert hashlib.sha256((path / ".zattrs").read_bytes()).hexdigest() == attrs_hash
        result = {
            "dataset": path.name,
            "stored_dates": len(stored_dates),
            "calendar_dates": len(calendar),
            "missing_dates": len(expected_missing),
            "all_observed_dates_values_equal": True,
            "values_checked": values_checked,
            "data_sha256": stored_digest.hexdigest(),
            "adapted_data_sha256": adapted_digest.hexdigest(),
            "attrs_sha256_before_after": attrs_hash,
            "statistics_sha256": statistics_hashes,
            "statistics_start_date": raw.attrs["statistics_start_date"],
            "statistics_end_date": raw.attrs["statistics_end_date"],
        }
        results.append(result)
        print("STORE_VERIFIED", json.dumps(result), flush=True)
    result = {
        "anemoi_datasets": anemoi_version,
        "setting": {"ANEMOI_DATASETS_MISSING_DATES_FIX_EXPERIMENTAL": "1"},
        "mode": "Native reader in-memory calendar adapter; original stores opened read-only; no conversion or data copy.",
        "stores": results,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "complete": True,
    }
    output_file.write_text(json.dumps(result, indent=2) + "\n")
    print(
        "ALL_STORES_VERIFIED",
        json.dumps({k: v for k, v in result.items() if k != "stores"}),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "calendar-verification.json",
    )
    args = parser.parse_args()
    main(args.data, args.output)
