# Run multistage training

Multistage training is an alternative to [normal training](./train.md) for `EncodeProcessDecode` models.
Instead of training the full model in one pass, each component is trained in isolation before the weights are loaded into the full model for a final finetuning step.
This can be useful when the model is large, when encoder inputs have very different characteristics, or when you want finer control over the training of individual components.

## The four stages

**Stage 1 - Train encoders**

Each encoder is trained independently as a standalone autoencoder (encoder + disposable decoder). One training run per encoder.

![Stage 1 diagram](../assets/staged-training-stage1.png)

**Stage 2 - Train decoder**

The decoder is trained on the combined frozen encoder latents from stage 1.

![Stage 2 diagram](../assets/staged-training-stage2.png)

**Stage 3 - Train processor**

The processor is trained with frozen encoders and decoder from stages 1–2.

![Stage 3 diagram](../assets/staged-training-stage3.png)

**Stage 4 - Finetune**

Pretrained weights are loaded into the full `EncodeProcessDecode` model. Models with a physical-output loss train end-to-end. Latent-loss processors keep the decoder frozen by default; the optional decoded-output loss below enables joint finetuning.

![Stage 4 diagram](../assets/staged-training-stage4.png)

## Running staged training

```bash
uv run cryocast train --multistage
```

A checkpoint is saved at the end of each stage. To resume a partially completed run, pass `--checkpoint-dir` pointing at the checkpoint directory from the original run - any stage whose checkpoint already exists there will be skipped:

```bash
uv run cryocast train --multistage --checkpoint-dir ${BASE_DIR}/training/wandb/run-<date>-<id>/checkpoints
```

## Per-stage config overrides

Each stage inherits the top-level `train` config.
To override settings for a specific stage, add a block under the `multistage` key:

```yaml
train:
  trainer:
    max_epochs: 20
  multistage:
    encoders:
      trainer:
        max_epochs: 10
    decoder:
      trainer:
        max_epochs: 10
    processor:
      trainer:
        max_epochs: 10
    finetune:
      optimizer:
        lr: 1e-4
      trainer:
        max_epochs: 5
```

Any key valid under `train` can be overridden per stage: `optimizer`, `scheduler`, `trainer`, and `callbacks`.
Keys not present in a stage block fall back to the top-level `train` values.
For example, the `lr` for the `encoders`, `decoder` and `processor` stages would be the default `5e-3` if not overridden elsewhere.


## Optional decoded-output loss for diffusion finetuning

For `EncodeProcessDecode` with a `DiffusionProcessor`, set a positive
`model.decoded_loss_weight` to train the decoder alongside the processor and input
encoders in the final stage. The default is `0.0`, preserving the existing loss,
frozen decoder and checkpoint parameter layout. No shipped preset enables it.

For example, with your existing diffusion baseline and local data configuration:

```bash
uv run cryocast train --multistage --config-name my-diffusion.local \
  +model.decoded_loss_weight=0.1
```

The coefficient is an experimental choice, not a recommended optimum. It belongs
under `model`, not `train.multistage.finetune`. The separately constructed
`ProcessorStage` continues to use a frozen decoder and rejects a nonzero weight.
The unused target encoder also remains frozen.

The objective is:

```text
latent_loss + decoded_loss_weight * mean_b(alpha_bar[t_b] * output_loss_b)
```

Here `alpha_bar[t] = SNR(t) / (1 + SNR(t))` is the bounded signal-to-noise weight
from the trained diffusion schedule. It suppresses unreliable high-noise
reconstructions without a singular division at an endpoint. Each sample's output
loss uses the configured physical loss, including its spatial and lead-time
reductions; weights are averaged over the whole batch, not normalised by their sum.
For losses with batch-dependent reductions, this per-sample definition is
intentional. Masks, range restrictions and skip connections use the existing decoder.

Autoregressive diffusion trains only forecast step zero, so only that step enters
the auxiliary loss. The repeated estimates used for legacy training metrics are
not extra supervised forecasts. Parallel diffusion supervises every trained step.
Validation and inference retain their existing sampling paths.

Old checkpoints without the new option use weight zero. Enabling the option for a
new finetune run adds decoder parameters to its optimizer: initialise that
optimizer afresh rather than loading optimizer state from a weight-zero run.
The existing multistage flow already creates a fresh optimizer for finetuning.
Matching settings can resume normally. Improved forecasting skill still requires
matched experiments; gradient and checkpoint tests alone do not establish it.
