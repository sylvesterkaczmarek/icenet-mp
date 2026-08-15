from icenet_mp.models.encoders import (
    CNNEncoder,
    DeepCompressionEncoder,
    PiecewiseEncoder,
)
from icenet_mp.types import DataSpace


def test_cnn_sets_output_channels_at_construction() -> None:
    encoder = CNNEncoder(
        data_space_in=DataSpace(name="input", channels=3, shape=(16, 16)),
        latent_space=(4, 4),
        n_layers=2,
        scale_factor=2,
    )
    assert encoder.data_space_out.channels == 12


def test_piecewise_sets_output_channels_at_construction() -> None:
    encoder = PiecewiseEncoder(
        data_space_in=DataSpace(name="input", channels=2, shape=(8, 8)),
        latent_space=(4, 4),
        conv_subblocks_initial=0,
        conv_subblocks_final=0,
    )
    assert encoder.data_space_out.channels == 50


def test_deep_compression_sets_output_channels_at_construction() -> None:
    encoder = DeepCompressionEncoder(
        data_space_in=DataSpace(name="input", channels=3, shape=(8, 8)),
        latent_space=(4, 4),
        hid_blocks=(1, 1),
        hid_channels=(4, 8),
        latent_channels=5,
        stride=2,
    )
    assert encoder.data_space_out.channels == 5
