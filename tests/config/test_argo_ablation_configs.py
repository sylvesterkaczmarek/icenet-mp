from importlib.resources import files

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

CONFIG_DIR = str(files("icenet_mp.config"))
PAIRS = (
    ("sample_north", "sample_north_no_argo"),
    ("sample_south", "sample_south_no_argo"),
    ("full_north", "full_north_no_argo"),
    ("full_south", "full_south_no_argo"),
)


def _compose_data(data_config: str) -> DictConfig:
    """Compose the base config with one data-group override."""
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="base", overrides=[f"data={data_config}"])


def _datasets(cfg: DictConfig) -> dict[str, dict[str, object]]:
    """Return configured dataset entries as ordinary dictionaries."""
    return OmegaConf.to_container(cfg.data.datasets, resolve=True)  # type: ignore[return-value]


def _non_argo_dataset_names(cfg: DictConfig) -> set[str]:
    return {
        str(dataset["name"])
        for dataset in _datasets(cfg).values()
        if dataset["group_as"] != "float-argo"
    }


def test_no_argo_configs_remove_only_argo_dataset() -> None:
    """Keep SIC, ERA5 and split settings identical in each ablation pair."""
    for with_argo_name, without_argo_name in PAIRS:
        with_argo = _compose_data(with_argo_name)
        without_argo = _compose_data(without_argo_name)

        with_groups = {str(ds["group_as"]) for ds in _datasets(with_argo).values()}
        without_groups = {
            str(ds["group_as"]) for ds in _datasets(without_argo).values()
        }

        assert "float-argo" in with_groups
        assert "float-argo" not in without_groups
        assert without_groups == with_groups - {"float-argo"}
        assert _non_argo_dataset_names(with_argo) == _non_argo_dataset_names(without_argo)
        assert OmegaConf.to_container(with_argo.data.split, resolve=True) == OmegaConf.to_container(
            without_argo.data.split, resolve=True
        )


def test_no_argo_configs_preserve_prediction_target_group() -> None:
    """Removing Argo must not alter the configured SIC prediction target."""
    for _, without_argo_name in PAIRS:
        cfg = _compose_data(without_argo_name)
        target_group = str(cfg.predict.target.group_name)
        groups = {str(ds["group_as"]) for ds in _datasets(cfg).values()}

        assert target_group in groups
