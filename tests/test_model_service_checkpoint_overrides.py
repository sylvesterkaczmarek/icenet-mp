from pathlib import Path
from unittest.mock import MagicMock

import pytest
from omegaconf import DictConfig, OmegaConf

from icenet_mp.model_service import ModelService


class FakeCommonDataModule:
    def __init__(self, _config: DictConfig) -> None:
        self.mask_directory = Path("nonexistent")
        self.latitudes: dict[str, list[float]] = {}
        self.longitudes: dict[str, list[float]] = {}


class FakeModel:
    @classmethod
    def load_from_checkpoint(
        cls, *_args: object, **_kwargs: object
    ) -> "FakeModel":
        return cls()


def test_checkpoint_model_overrides_reach_loader(
    cfg_model_service: DictConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    checkpoint_path = checkpoints_dir / "model.ckpt"
    checkpoint_path.write_text("checkpoint")

    files_dir = tmp_path / "files"
    files_dir.mkdir(parents=True)
    OmegaConf.save(cfg_model_service, files_dir / "model_config.yaml")

    processor_override = DictConfig(
        {"_target_": "example.DDIMProcessor", "num_inference_steps": 10}
    )
    load_from_checkpoint = MagicMock(return_value=FakeModel())
    monkeypatch.setattr("icenet_mp.model_service.CommonDataModule", FakeCommonDataModule)
    monkeypatch.setattr(
        "icenet_mp.model_service.hydra.utils.get_class", lambda _target: FakeModel
    )
    monkeypatch.setattr(FakeModel, "load_from_checkpoint", load_from_checkpoint)

    service = ModelService.from_checkpoint(
        DictConfig({"model": {"processor": processor_override}}), checkpoint_path
    )

    assert service.config["model"]["processor"] == processor_override
    loader_kwargs = load_from_checkpoint.call_args.kwargs
    assert loader_kwargs["processor"] == processor_override
    assert "_target_" not in loader_kwargs
