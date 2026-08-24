import pytest
import torch

from icenet_mp.models.processors import VitProcessor
from icenet_mp.types import DataSpace


def _processor(*, patch_size: int = 4, dropout: float = 0.0) -> VitProcessor:
    """Build a small ViT processor for unit tests."""
    return VitProcessor(
        data_space=DataSpace(name="latent", channels=3, shape=(8, 8)),
        n_forecast_steps=2,
        n_history_steps=3,
        depth=2,
        dropout=dropout,
        emb_dim=16,
        heads=4,
        mlp_dim=32,
        patch_size=patch_size,
    )


def test_vit_rollout_preserves_processor_shape() -> None:
    """Return one latent frame for each requested forecast step."""
    processor = _processor()
    inputs = torch.randn(2, 3, 3, 8, 8)

    output = processor.rollout(inputs)

    assert output.prediction.shape == (2, 2, 3, 8, 8)


def test_vit_forward_uses_full_history_channel_window() -> None:
    """Accept the history window concatenated along the channel dimension."""
    processor = _processor()
    inputs = torch.randn(2, 9, 8, 8)

    output = processor(inputs)

    assert output.shape == (2, 3, 8, 8)
    assert processor.patch_embed.proj.in_channels == 9


def test_vit_backpropagates_through_embeddings_and_decoder() -> None:
    """Keep positional embeddings and decoder parameters trainable end to end."""
    processor = _processor()
    inputs = torch.randn(2, 3, 3, 8, 8, requires_grad=True)

    processor.rollout(inputs).prediction.square().mean().backward()

    assert inputs.grad is not None
    assert processor.pos_embed.grad is not None
    assert processor.decoder[0].weight.grad is not None
    assert processor.smooth.weight.grad is not None


def test_vit_positional_embedding_matches_patch_grid() -> None:
    """Create one learned positional token for every spatial patch."""
    processor = _processor(patch_size=2)

    assert processor.pos_embed.shape == (1, 16, 16)


def test_vit_rejects_non_square_latent_space() -> None:
    """Reject latent spaces that cannot use the square ViT patch grid."""
    with pytest.raises(ValueError, match="height and width"):
        VitProcessor(
            data_space=DataSpace(name="latent", channels=3, shape=(8, 12)),
            n_forecast_steps=1,
            n_history_steps=1,
            patch_size=4,
        )


def test_vit_rejects_patch_size_that_does_not_tile_image() -> None:
    """Reject patch sizes that do not evenly tile the configured image."""
    with pytest.raises(ValueError, match="must be divisible by patch_size"):
        _processor(patch_size=3)


def test_vit_eval_mode_is_deterministic_when_dropout_is_disabled() -> None:
    """Return stable predictions for identical inputs in evaluation mode."""
    processor = _processor().eval()
    inputs = torch.randn(1, 9, 8, 8)

    first = processor(inputs)
    second = processor(inputs)

    torch.testing.assert_close(first, second)
