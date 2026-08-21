import torch

from icenet_mp.metrics import IceNetAccuracy, SeaIceExtentErrorPerForecastDay


def test_icenet_accuracy_is_reported_per_forecast_day() -> None:
    metric = IceNetAccuracy()
    predictions = torch.tensor(
        [[[[[0.20, 0.10]]], [[[0.20, 0.10]]]]],
    )
    targets = torch.tensor(
        [[[[[0.20, 0.10]]], [[[0.10, 0.10]]]]],
    )

    metric.update(predictions, targets)

    torch.testing.assert_close(metric.compute(), torch.tensor([100.0, 50.0]))


def test_icenet_accuracy_respects_sample_weights() -> None:
    metric = IceNetAccuracy()
    predictions = torch.tensor([[[[[0.20, 0.20]]]]])
    targets = torch.tensor([[[[[0.20, 0.10]]]]])
    sample_weight = torch.tensor([[[[[1, 0]]]]])

    metric.update(predictions, targets, sample_weight)

    torch.testing.assert_close(metric.compute(), torch.tensor([100.0]))


def test_icenet_accuracy_accumulates_batches() -> None:
    metric = IceNetAccuracy()

    metric.update(
        torch.tensor([[[[[0.20, 0.10]]]]]),
        torch.tensor([[[[[0.20, 0.10]]]]]),
    )
    metric.update(
        torch.tensor([[[[[0.20, 0.20]]]]]),
        torch.tensor([[[[[0.10, 0.10]]]]]),
    )

    torch.testing.assert_close(metric.compute(), torch.tensor([50.0]))


def test_sea_ice_extent_error_is_mean_absolute_error_per_day() -> None:
    metric = SeaIceExtentErrorPerForecastDay(pixel_size=10)
    predictions = torch.tensor(
        [
            [[[[0.20, 0.20]]], [[[0.10, 0.10]]]],
            [[[[0.10, 0.10]]], [[[0.20, 0.20]]]],
        ]
    )
    targets = torch.tensor(
        [
            [[[[0.20, 0.10]]], [[[0.20, 0.10]]]],
            [[[[0.10, 0.10]]], [[[0.10, 0.10]]]],
        ]
    )

    metric.update(predictions, targets)

    # Mean pixel-count errors are 0.5 and 1.5; each 10 km pixel is 100 km².
    torch.testing.assert_close(metric.compute(), torch.tensor([50.0, 150.0]))


def test_sea_ice_extent_error_is_zero_for_identical_fields() -> None:
    metric = SeaIceExtentErrorPerForecastDay(pixel_size=25)
    field = torch.tensor([[[[[0.20, 0.10], [0.30, 0.00]]]]])

    metric.update(field, field.clone())

    torch.testing.assert_close(metric.compute(), torch.tensor([0.0]))
