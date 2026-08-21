# Notebooks overview

The notebooks in this directory cover several generations of IceNet-MP development. Some are current exploratory examples, while others predate the present pipeline and are retained for reference.

## Current notebook status

| Notebook | Purpose | Status |
| --- | --- | --- |
| `ARGO_data.ipynb` | Downloading and gridding Argo float observations | Working |
| `case_study_whale_corridors.ipynb` | Whale and ship corridor demonstrations | Working, requires the associated data |
| `degrid_and_visualise.ipynb` | Create and visualise synthetic non-gridded data | Working |
| `extract_anomalies.ipynb` | Calculate climate anomalies used by the de-gridding example | Working |
| `layer_diagnostics.ipynb` | Inspect activation layers of a U-Net model | Needs review |
| `persistence.ipynb` | Early persistence-model exploration | Needs review; the persistence model now exists in the pipeline |
| `0_notebook_tf.ipynb` | Legacy TensorFlow U-Net workflow derived from the original IceNet pipeline | Currently broken |
| `1_icenet_forecast_unet.ipynb` | Legacy PyTorch U-Net workflow derived from the original IceNet pipeline | Currently broken |
| `2_icenet_forecast_cgan.ipynb` | Legacy PyTorch CGAN workflow derived from the original IceNet pipeline | Currently broken |
| `demo_pipeline.ipynb` | Earlier end-to-end pipeline demonstration | Currently broken |

The status above tracks the notebook review in [issue #417](https://github.com/alan-turing-institute/icenet-mp/issues/417). Until `demo_pipeline.ipynb` is refreshed, use the maintained command-line workflow and project documentation for an end-to-end IceNet-MP run.

## Running notebooks

Install the notebook dependencies and start Jupyter with:

```bash
uv run --group notebooks jupyter notebook
```

For the maintained pipeline workflow, see the main [README](../README.md) and the project documentation linked there.

## Legacy IceNet notebooks

Notebooks 0, 1, and 2 include TensorFlow U-Net, PyTorch U-Net, and PyTorch CGAN versions of the earlier IceNet workflow. They were adapted from the following projects:

- [eds-book-gallery](https://github.com/eds-book-gallery/67a1e320-7c47-4ea9-8df8-e868326bc90b/tree/main)
- [icenet-notebooks](https://github.com/icenet-ai/icenet-notebooks)

They are retained as historical references rather than recommended entry points to the current IceNet-MP pipeline.
