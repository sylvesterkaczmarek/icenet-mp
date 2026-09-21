from pathlib import Path

import pytest
import yaml

DATASETS_DIR = Path(__file__).parents[2] / "icenet_mp" / "config" / "data" / "datasets"
YAML_FILES = sorted(DATASETS_DIR.glob("*.yaml"))


class TestConfigFiles:
    """Tests for dataset config files."""

    @pytest.mark.parametrize(
        "config_file", YAML_FILES, ids=[f.name for f in YAML_FILES]
    )
    def test_filename_matches_key_and_name(self, config_file: Path) -> None:
        expected_key = config_file.stem.replace("_", "-")
        with config_file.open() as f:
            data = yaml.safe_load(f)

        assert list(data.keys()) == [expected_key], (
            f"Top-level key {list(data.keys())!r} does not match filename-derived key {expected_key!r}"
        )
        assert data[expected_key]["name"] == expected_key, (
            f"name attribute {data[expected_key]['name']!r} does not match top-level key {expected_key!r}"
        )

    @pytest.mark.parametrize(
        "config_file", YAML_FILES, ids=[f.name for f in YAML_FILES]
    )
    def test_dataset_body_has_anemoi_recipe_structure(self, config_file: Path) -> None:
        with config_file.open() as f:
            data = yaml.safe_load(f)
        body = next(iter(data.values()))

        assert body.get("group_as"), "missing or empty 'group_as'"
        dates = body.get("dates") or {}
        assert dates.get("start"), "missing 'dates.start'"
        assert dates.get("end"), "missing 'dates.end'"
        assert dates.get("frequency"), "missing 'dates.frequency'"
        # anemoi's 'input' recipe shape varies (a single source, or a pipe/join/concat
        # combinator), so only its presence as a non-empty mapping is source-agnostic.
        assert isinstance(body.get("input"), dict), "missing 'input' recipe"
        assert body["input"], "empty 'input' recipe"

    def test_carra2_svalbard_roi_recipe(self) -> None:
        """Keep the initial downscaling ROI and CARRA2 target request explicit."""
        config_file = (
            DATASETS_DIR / "samp_sicnorth_carra2_2p5km_2020_2024_24h_v1.yaml"
        )
        with config_file.open() as f:
            data = yaml.safe_load(f)

        body = next(iter(data.values()))
        source = body["input"]["cds"]
        request = source["request"]

        assert source["dataset"] == "reanalysis-pan-carra"
        assert request["area"] == [81, 15, 76, 35]
        assert request["variable"] == ["sea_ice_area_fraction"]
        assert request["product_type"] == "analysis"
        assert body["statistics"]["end"] == "2021-12-31T12:00:00"
