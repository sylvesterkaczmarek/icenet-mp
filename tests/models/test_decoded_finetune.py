"""Contracts for the optional physical-output loss on latent-loss finetuning."""

import copy
from dataclasses import replace
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
import torch
from lightning import Trainer
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cryocast.models import EncodeProcessDecode
from cryocast.models.decoders import NaiveLinearDecoder
from cryocast.models.encoders import BaseEncoder
from cryocast.models.multistage import DecoderStage, EncoderStage, ProcessorStage
from cryocast.models.processors import BaseProcessor, DiffusionProcessor, NullProcessor
from cryocast.types import DataSpace, ProcessorOutput, TensorNCHW


class _ScaleEncoder(BaseEncoder):
    def __init__(self, data_space_in: DataSpace) -> None:
        super().__init__(
            data_space_in=data_space_in,
            latent_space=data_space_in.shape,
            output_channels=data_space_in.channels,
        )
        self.scale = nn.Parameter(torch.tensor(0.8))

    def forward(self, x: TensorNCHW) -> TensorNCHW:
        return self.scale * x


class _Velocity(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels + 6, channels, 1)

    def forward(
        self, y: torch.Tensor, t: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        return self.conv(torch.cat([y, cond], dim=1)) + 0 * t[:, None, None, None]


def make_model(
    weight: float = 0.0, *, autoregressive: bool = True, real_unet: bool = False
) -> EncodeProcessDecode:
    """Use two conditioning sources, with a one-channel subset of the target source."""
    shape = (16, 16)
    spaces = [
        DataSpace(name="weather", channels=1, shape=shape),
        DataSpace(name="sic", channels=2, shape=shape),
    ]
    output = DataSpace(name="sic", channels=1, shape=shape)
    combined = DataSpace(name="combined", channels=3, shape=shape)
    processor = DiffusionProcessor(
        data_space=combined,
        data_space_target=spaces[1],
        target_channel_offset=1,
        n_history_steps=2,
        n_forecast_steps=3,
        timesteps=8,
        start_out_channels=8,
        time_embed_dim=256,
        dropout_rate=0,
        use_autoregressive=autoregressive,
        loss=nn.MSELoss(),
    )
    if not real_unet:
        processor.add_module("model", _Velocity(processor.diffused_channels))
    return EncodeProcessDecode(
        name="decoded-finetune",
        hemisphere="north",
        input_spaces=[s.to_dict() for s in spaces],
        output_space=output.to_dict(),
        encoders=cast(
            "list[BaseEncoder]",
            [_ScaleEncoder(s) for s in spaces]
            + [_ScaleEncoder(DataSpace(name="target", channels=1, shape=shape))],
        ),
        processor=processor,
        decoder=NaiveLinearDecoder(data_space_in=combined, data_space_out=output),
        target_variable_indices=[1],
        n_history_steps=2,
        n_forecast_steps=3,
        decoded_loss_weight=weight,
        metrics=[],
        loss=DictConfig({"_target_": "torch.nn.MSELoss"}),
        optimizer=DictConfig({"_target_": "torch.optim.AdamW", "lr": 1e-3}),
        scheduler=DictConfig({}),
        lr_scheduler=DictConfig({}),
    )


def make_batch() -> dict[str, torch.Tensor]:
    """Return distinct future targets without sharing tensors with conditioning."""
    generator = torch.Generator().manual_seed(587)
    return {
        name: torch.rand(shape, generator=generator)
        for name, shape in {
            "weather": (2, 2, 1, 16, 16),
            "sic": (2, 2, 2, 16, 16),
            "target": (2, 3, 1, 16, 16),
        }.items()
    }


def gradients(model: nn.Module) -> dict[str, torch.Tensor | None]:
    """Snapshot gradient values including explicitly absent gradients."""
    return {
        n: None if p.grad is None else p.grad.clone()
        for n, p in model.named_parameters()
    }


@pytest.mark.parametrize(
    "weight", [-1, float("nan"), float("inf"), -float("inf"), True, "0.1"]
)
def test_rejects_invalid_weight(weight: object) -> None:
    """Reject coefficients that would silently change or poison the objective."""
    with pytest.raises(ValueError, match="finite nonnegative"):
        make_model(cast("float", weight))


@pytest.mark.parametrize("autoregressive", [True, False])
@pytest.mark.parametrize("real_unet", [False, True])
def test_enabled_loss_reaches_all_used_modules(
    *, autoregressive: bool, real_unet: bool
) -> None:
    """Check real and probe denoisers propagate finite gradients to all used modules."""
    model = make_model(0.3, autoregressive=autoregressive, real_unet=real_unet)
    batch = make_batch()
    output = model.training_step(batch, 0)
    assert output.loss is not None
    assert torch.isfinite(output.loss)
    output.loss.backward()
    for module in [*model.encoders, model.processor, model.decoder]:
        grads = [p.grad for p in module.parameters() if p.requires_grad]
        assert grads
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads if g is not None) > 0
    assert all(
        p.grad is None and not p.requires_grad
        for p in model.target_encoder.parameters()
    )
    assert not output.prediction.requires_grad
    assert output.prediction.shape == batch["target"].shape


@pytest.mark.parametrize("autoregressive", [True, False])
def test_zero_weight_matches_legacy_training_equations(*, autoregressive: bool) -> None:
    """Keep predictions, scalar loss and gradients bitwise identical with weight zero."""
    model = make_model(autoregressive=autoregressive)
    reference = copy.deepcopy(model)
    batch = make_batch()
    torch.manual_seed(19)
    actual = model.training_step(batch, 0)
    torch.manual_seed(19)
    history = reference._encode_inputs(batch)
    target = reference._encode_target_latent(batch, batch["target"].clone().detach())
    expected = reference.processor.rollout(history, target)
    with torch.no_grad():
        prediction = reference.decoder.rollout(expected.prediction, None)
    assert actual.loss is not None
    assert expected.loss is not None
    torch.testing.assert_close(actual.loss, expected.loss, rtol=0, atol=0)
    torch.testing.assert_close(actual.prediction, prediction, rtol=0, atol=0)
    actual.loss.backward()
    expected.loss.backward()
    for name, gradient in gradients(model).items():
        other = gradients(reference)[name]
        if gradient is None:
            assert other is None
        else:
            torch.testing.assert_close(gradient, other, rtol=0, atol=0)
    assert not model.decoder.training
    assert all(not p.requires_grad for p in model.decoder.parameters())


@pytest.mark.parametrize("autoregressive", [True, False])
def test_processor_exposes_only_trained_steps_with_exact_snr_weight(
    *, autoregressive: bool
) -> None:
    """Expose trained frames only and retain the original latent objective."""
    model = make_model(autoregressive=autoregressive)
    processor = model.processor
    assert isinstance(processor, DiffusionProcessor)
    batch = make_batch()
    x = model._encode_inputs(batch)
    y = model._encode_target_latent(batch, batch["target"])
    times = torch.tensor([0, 7])
    with patch("torch.randint", return_value=times):
        torch.manual_seed(91)
        default = processor.rollout(x, y)
        torch.manual_seed(91)
        enabled = processor.rollout_with_decoded_loss(x, y)
    assert default.decoded_prediction is None
    assert not default.prediction.requires_grad
    assert enabled.decoded_prediction is not None
    assert enabled.decoded_prediction.requires_grad
    assert enabled.decoded_prediction.shape[1] == (1 if autoregressive else 3)
    assert not enabled.prediction.requires_grad
    torch.testing.assert_close(default.prediction, enabled.prediction, rtol=0, atol=0)
    torch.testing.assert_close(default.loss, enabled.loss, rtol=0, atol=0)
    assert enabled.decoded_loss_weights is not None
    torch.testing.assert_close(
        enabled.decoded_loss_weights, processor.diffusion.alphas_cumprod[times]
    )
    assert enabled.decoded_loss_weights[0] > enabled.decoded_loss_weights[1] >= 0
    with torch.no_grad():
        estimate = processor.rollout_with_decoded_loss(x, y).decoded_prediction
        assert estimate is not None
        assert not estimate.requires_grad


def test_autoregressive_loss_ignores_later_ground_truth() -> None:
    """Later targets must not supervise repetitions of the first-step estimate."""
    model = make_model(0.5)
    other = copy.deepcopy(model)
    batch = make_batch()
    changed = {k: v.clone() for k, v in batch.items()}
    changed["target"][:, 1:] += 1000
    torch.manual_seed(12)
    first = model.training_step(batch, 0)
    torch.manual_seed(12)
    second = other.training_step(changed, 0)
    assert first.loss is not None
    assert second.loss is not None
    torch.testing.assert_close(first.loss, second.loss, rtol=0, atol=0)
    first.loss.backward()
    second.loss.backward()
    for name, grad in gradients(model).items():
        if grad is not None:
            torch.testing.assert_close(grad, gradients(other)[name], rtol=0, atol=0)


@pytest.mark.parametrize("autoregressive", [True, False])
def test_decoded_term_matches_weighted_per_sample_oracle(
    *, autoregressive: bool
) -> None:
    """Compare the reduction with direct per-sample arithmetic."""
    model = make_model(0.7, autoregressive=autoregressive)
    batch = make_batch()
    n_steps = 1 if autoregressive else 3
    latent = torch.randn(2, n_steps, 3, 16, 16, requires_grad=True)
    weights = torch.tensor([0.2, 0.8])
    raw = ProcessorOutput(
        prediction=latent.detach(),
        decoded_prediction=latent,
        decoded_loss_weights=weights,
    )
    prediction, result = model._decoded_training_loss(raw, batch["target"], None)
    decoded = model.decoder.rollout(latent, None)
    expected = (
        (decoded - batch["target"][:, :n_steps]).square().mean(dim=(1, 2, 3, 4))
        * weights
    ).mean()
    torch.testing.assert_close(result, expected)
    assert prediction.shape == batch["target"].shape
    result.backward()
    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()


@pytest.mark.parametrize("weight", [0.0, 0.4])
def test_train_eval_cycles_and_optimizer_membership(weight: float) -> None:
    """Keep unused target encoders frozen and include enabled decoder parameters."""
    model = make_model(weight)
    for mode in [False, True, False, True]:
        model.train(mode=mode)
        assert model.decoder.training == (mode and weight > 0)
        assert not model.target_encoder.training
    optimizer = get_optimizer(model)
    included = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert all((id(p) in included) == (weight > 0) for p in model.decoder.parameters())
    assert all(id(p) not in included for p in model.target_encoder.parameters())


@pytest.mark.parametrize("weight", [0.0, 0.4])
@pytest.mark.parametrize("autoregressive", [True, False])
def test_safe_checkpoint_and_adam_resume(
    tmp_path: Path, weight: float, *, autoregressive: bool
) -> None:
    """Restore portable state and reproduce the next AdamW update exactly."""
    model = make_model(weight, autoregressive=autoregressive)
    optimizer = get_optimizer(model)
    batch = make_batch()
    model.training_step(batch, 0).loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )
    saved = torch.load(path, weights_only=True)
    restored = make_model(
        saved["hyper_parameters"]["decoded_loss_weight"], autoregressive=autoregressive
    )
    restored.load_state_dict(saved["state_dict"], strict=True)
    resumed = get_optimizer(restored)
    resumed.load_state_dict(saved["optimizer"])
    for current, opt in [(model, optimizer), (restored, resumed)]:
        torch.manual_seed(921)
        current.training_step(batch, 0).loss.backward()
        opt.step()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)


def test_unsupported_processor_and_processor_stage_reject_opt_in() -> None:
    """Prevent silent opt-in on unsupported processors or the isolated processor stage."""
    model = make_model()
    args = dict(model.hparams)
    args.update(
        encoders=[*model.encoders, model.target_encoder],
        decoder=model.decoder,
        processor=NullProcessor(
            data_space=model.processor.data_space, n_forecast_steps=3, n_history_steps=2
        ),
        metrics=[],
        decoded_loss_weight=0.1,
    )
    with pytest.raises(ValueError, match="supporting decoded-output"):
        EncodeProcessDecode(**args)
    with pytest.raises(ValueError, match="full finetune model"):
        ProcessorStage(
            processor=DictConfig({}),
            decoder_model=cast("DecoderStage", None),
            target_encoder=cast("EncoderStage", None),
            decoded_loss_weight=0.1,
        )
    with pytest.raises(NotImplementedError, match="does not support"):
        BaseProcessor(
            data_space=model.processor.data_space, n_forecast_steps=3, n_history_steps=2
        ).rollout_with_decoded_loss(torch.empty(0), torch.empty(0))


def get_optimizer(model: EncodeProcessDecode) -> torch.optim.Optimizer:
    """Narrow the unscheduled optimizer configuration used by these fixtures."""
    config = model.configure_optimizers()
    assert isinstance(config, dict)
    optimizer = config["optimizer"]
    assert isinstance(optimizer, torch.optim.Optimizer)
    return optimizer


@pytest.mark.parametrize("autoregressive", [True, False])
def test_auxiliary_loss_alone_reaches_conditioning_and_decoder(
    *, autoregressive: bool
) -> None:
    """Exclude the latent objective when proving the new gradient path works."""
    model = make_model(0.2, autoregressive=autoregressive).double()
    batch = {k: v.double().requires_grad_() for k, v in make_batch().items()}
    out = model.processor.rollout_with_decoded_loss(
        model._encode_inputs(batch),
        model._encode_target_latent(batch, batch["target"]).detach(),
    )
    _, auxiliary = model._decoded_training_loss(out, batch["target"].detach(), None)
    auxiliary.backward()
    for module in [*model.encoders, model.processor, model.decoder]:
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()
        )
    assert batch["target"].grad is None
    for name in ["sic", "weather"]:
        assert batch[name].grad is not None
    assert all(p.grad is None for p in model.target_encoder.parameters())


@pytest.mark.parametrize("autoregressive", [True, False])
def test_enabling_auxiliary_loss_does_not_change_inference(
    *, autoregressive: bool
) -> None:
    """Matching weights and random state produce identical reverse-diffusion samples."""
    original = make_model(autoregressive=autoregressive).eval()
    enabled = make_model(0.4, autoregressive=autoregressive).eval()
    enabled.load_state_dict(original.state_dict(), strict=True)
    with torch.no_grad():
        torch.manual_seed(752)
        first = original(make_batch())
        torch.manual_seed(752)
        second = enabled(make_batch())
    torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize(
    "case", ["missing", "batch", "steps", "nan", "negative", "above-one"]
)
def test_rejects_invalid_decoded_loss_outputs(case: str) -> None:
    """Malformed auxiliary metadata must fail rather than broadcast silently."""
    model = make_model(0.2)
    latent = torch.ones(2, 1, 3, 16, 16)
    output = ProcessorOutput(
        prediction=latent, decoded_prediction=latent, decoded_loss_weights=torch.ones(2)
    )
    if case == "missing":
        output = replace(output, decoded_prediction=None)
    elif case == "batch":
        output = replace(output, decoded_loss_weights=torch.ones(2, 1))
    elif case == "steps":
        output = replace(output, decoded_prediction=latent.expand(-1, 2, -1, -1, -1))
    else:
        value = {"nan": float("nan"), "negative": -0.1, "above-one": 1.1}[case]
        output = replace(output, decoded_loss_weights=torch.tensor([value, 1.0]))
    with pytest.raises(ValueError, match="decoded-loss"):
        model._decoded_training_loss(output, make_batch()["target"], None)


@pytest.mark.parametrize("weight", [0.0, 0.4])
def test_lightning_fit_and_safe_reload(tmp_path: Path, weight: float) -> None:
    """Exercise a real optimizer loop and Lightning's portable checkpoint loader."""
    model = make_model(weight)
    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        default_root_dir=tmp_path,
    )
    # A list of batches is a supported one-batch Lightning iterable.
    loader = DataLoader(_OneBatch(), batch_size=None)
    trainer.fit(model, train_dataloaders=loader)
    path = tmp_path / "lightning.ckpt"
    trainer.save_checkpoint(path)
    saved = torch.load(path, weights_only=True)
    assert saved["hyper_parameters"]["decoded_loss_weight"] == weight
    template = make_model(weight)
    restored = EncodeProcessDecode.load_from_checkpoint(
        path,
        weights_only=True,
        encoders=[*template.encoders, template.target_encoder],
        processor=template.processor,
        decoder=template.decoder,
        metrics=[],
    )
    assert restored.decoded_loss_weight == weight
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)


class _OneBatch(Dataset[dict[str, torch.Tensor]]):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index != 0:
            raise IndexError(index)
        return make_batch()


@pytest.mark.parametrize("autoregressive", [True, False])
@pytest.mark.parametrize("loss_name", ["huber", "amse", "lead-weighted"])
def test_real_physical_loss_variants(loss_name: str, *, autoregressive: bool) -> None:
    """The configured physical criterion must keep finite gradients in both modes."""
    model = make_model(0.3, autoregressive=autoregressive)
    configs = {
        "huber": {"_target_": "torch.nn.HuberLoss", "delta": 0.5},
        "amse": {"_target_": "cryocast.losses.amse_loss.AMSELoss"},
        "lead-weighted": {
            "_target_": "torch.nn.MSELoss",
            "lead_time_exponent": 2.0,
        },
    }
    model.loss_cfg = DictConfig(configs[loss_name])
    output = model.training_step(make_batch(), 0)
    assert output.loss is not None
    assert torch.isfinite(output.loss)
    output.loss.backward()
    for module in [*model.encoders, model.processor, model.decoder]:
        grads = [p.grad for p in module.parameters() if p.requires_grad]
        assert grads
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_rejects_nonscalar_physical_loss() -> None:
    """Reject reduction='none' rather than broadcasting per-pixel confidence."""
    model = make_model(0.2)
    model.loss_cfg = DictConfig({"_target_": "torch.nn.MSELoss", "reduction": "none"})
    with pytest.raises(ValueError, match="one scalar per sample"):
        model.training_step(make_batch(), 0)


def test_zero_confidence_has_zero_auxiliary_gradient() -> None:
    """A zero confidence weight excludes a sample without renormalising other weights."""
    model = make_model(0.2)
    latent = torch.randn(2, 1, 3, 16, 16, requires_grad=True)
    raw = ProcessorOutput(
        prediction=latent.detach(),
        decoded_prediction=latent,
        decoded_loss_weights=torch.tensor([0.0, 0.5], requires_grad=True),
    )
    _, loss = model._decoded_training_loss(raw, make_batch()["target"], None)
    loss.backward()
    assert latent.grad is not None
    assert torch.count_nonzero(latent.grad[0]) == 0
    assert torch.count_nonzero(latent.grad[1]) > 0
    assert raw.decoded_loss_weights is not None
    assert raw.decoded_loss_weights.grad is None
