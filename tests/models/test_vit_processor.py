import pytest
import torch

from icenet_mp.models.processors import VitProcessor
from icenet_mp.types import DataSpace


def make_vit_processor(*, dropout: float = 0.0) -> VitProcessor:
    return VitProcessor(
        data_space=DataSpace(name="latent", channels=3, shape=(16, 16)),
        n_forecast_steps=2,
        n_history_steps=2,
        depth=1,
        dropout=dropout,
        emb_dim=16,
        heads=4,
        mlp_dim=32,
        patch_size=4,
    )


def test_rejects_patch_size_that_does_not_divide_image() -> None:
    with pytest.raises(ValueError, match="must be divisible by patch_size"):
        VitProcessor(
            data_space=DataSpace(name="latent", channels=3, shape=(18, 18)),
            n_forecast_steps=1,
            n_history_steps=1,
            depth=1,
            emb_dim=16,
            heads=4,
            mlp_dim=32,
            patch_size=4,
        )


def test_forward_backpropagates_through_vit() -> None:
    processor = make_vit_processor()
    inputs = torch.randn(2, 6, 16, 16, requires_grad=True)

    output = processor(inputs)
    output.mean().backward()

    assert output.shape == (2, 3, 16, 16)
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert processor.pos_embed.grad is not None
    assert processor.decoder[0].weight.grad is not None
    assert processor.smooth.weight.grad is not None


def test_eval_mode_is_deterministic_with_dropout() -> None:
    processor = make_vit_processor(dropout=0.5)
    processor.eval()
    inputs = torch.randn(1, 6, 16, 16)

    with torch.no_grad():
        first = processor(inputs)
        second = processor(inputs)

    torch.testing.assert_close(first, second)
