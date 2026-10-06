"""Build standalone, source-backed comparison figures and the PR evidence note."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).parent
result=json.loads((ROOT/'results.json').read_text())
assert result['complete'] and len(result['runs'])==4
with np.load(ROOT/'input_slice.npz',allow_pickle=False) as f:
    sic=f['sic'];mask=f['mask'];dates=f['dates']
with np.load(ROOT/'predictions.npz',allow_pickle=False) as f:
    predictions={k:f[k].copy() for k in f.files}
assert hashlib.sha256((ROOT/'input_slice.npz').read_bytes()).hexdigest()==result['design']['input_sha256']
for seed in [477,478]:
    rows=[r for r in result['runs'] if r['seed']==seed]
    assert len({r['initial_state_sha256'] for r in rows})==1
    assert all(r['training_steps']==120 for r in rows)
    assert next(r for r in rows if r['scale']==1)['default_equals_base_training_and_sampling']
    for r in rows:
        a=predictions[f'seed_{seed}_scale_{r["scale"]:g}'][1:]
        error=a[...,mask]-sic[7:10,...][...,mask]
        assert np.isclose(np.abs(error).mean(),r['test_raw']['mae'])
        assert np.isclose(np.sqrt(np.square(error).mean()),r['test_raw']['rmse'])
figures=ROOT/'figures';figures.mkdir(exist_ok=True)
# Predeclared first test target and first training seed. All three use the same scale.
fields=[('observed-20240108.png','Observed sea ice',sic[7]),
        ('scale1-20240108.png','One-day forecast with x0_scale = 1',predictions['seed_477_scale_1'][1]),
        ('scale2-20240108.png','One-day forecast with x0_scale = 2',predictions['seed_477_scale_2'][1])]
for filename,title,field in fields:
    fig,ax=plt.subplots(figsize=(7.1,6.3),layout='constrained')
    displayed=np.ma.array(np.clip(field,0,1)*100,mask=~mask)
    im=ax.imshow(displayed,vmin=0,vmax=100,interpolation='nearest')
    ax.set_title(title+'\n8 January 2024, seed 477',fontsize=14)
    ax.set_xlabel('Column in fixed 25 km grid crop')
    ax.set_ylabel('Row in fixed 25 km grid crop')
    fig.colorbar(im,ax=ax,label='Sea-ice concentration (%)',shrink=0.8)
    fig.savefig(figures/filename,dpi=150)
    plt.close(fig)
# Discrete date comparisons, not a long time-series claim. Raw predictions, seed mean.
fig,ax=plt.subplots(figsize=(8.0,4.8),layout='constrained')
x=np.arange(3)
for scale,marker in [(1,'o'),(2,'s')]:
    values=np.array([[r['per_date'][k]['mae']*100 for k in [1,2,3]]
                     for r in result['runs'] if r['scale']==scale])
    ax.plot(x,values.mean(axis=0),marker=marker,label=f'x0_scale = {scale}, mean of two seeds')
persistence=[np.abs(sic[i,mask]-sic[i-1,mask]).mean()*100 for i in [7,8,9]]
ax.plot(x,persistence,marker='^',linestyle='--',label='Previous-day persistence')
ax.set_xticks(x,['8 Jan','9 Jan','10 Jan'])
ax.set_ylim(bottom=0)
ax.set_xlabel('Forecast target date in 2024, one-day lead')
ax.set_ylabel('Raw MAE (percentage points of sea-ice concentration)')
ax.set_title('Held-out one-day forecast errors',fontsize=14)
ax.legend(fontsize=9)
fig.savefig(figures/'daily-mae.png',dpi=150)
plt.close(fig)
rows=[]
for r in result['runs']:
    rows.append({'seed':r['seed'],'scale':r['scale'],
                 **{'raw_'+k+'_percentage_points':v*100 for k,v in r['test_raw'].items()},
                 **{'clipped_'+k+'_percentage_points':v*100 for k,v in r['test_clipped'].items()}})
with (ROOT/'metrics.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)
means={scale:{k:float(np.mean([r['test_raw'][k] for r in result['runs'] if r['scale']==scale])) for k in ['mae','rmse','bias']} for scale in [1,2]}
table='| Seed | x0_scale | Raw MAE | Raw RMSE | Raw bias | Clipped MAE | Clipped RMSE |\n|---|---|---|---|---|---|---|\n'
for r in rows:
    table+=f'| {r["seed"]} | {r["scale"]:g} | {r["raw_mae_percentage_points"]:.3f} | {r["raw_rmse_percentage_points"]:.3f} | {r["raw_bias_percentage_points"]:.3f} | {r["clipped_mae_percentage_points"]:.3f} | {r["clipped_rmse_percentage_points"]:.3f} |\n'
persistence=result['persistence']
text=f'''# Matched x0 scaling forecast pilot

This records actual training and forecast evaluation for IceNet PR #478. **The pilot does not show a benefit from `x0_scale=2.0`.** Raw MAE and RMSE worsen for both paired training seeds. Across the two seeds, mean raw MAE changes from **{means[1]['mae']*100:.3f}** to **{means[2]['mae']*100:.3f}** percentage points, and mean raw RMSE from **{means[1]['rmse']*100:.3f}** to **{means[2]['rmse']*100:.3f}**. The default should remain `1.0`.

## Results

All metrics below are percentage points of sea-ice concentration. Each row evaluates the same three one-day forecasts over the same {int(mask.sum()):,} fixed active-ocean pixels. Raw predictions are the primary comparison. Clipped metrics are reported separately and do not replace the raw result.

{table}

Previous-day persistence has raw MAE **{persistence['mae']*100:.3f}** and RMSE **{persistence['rmse']*100:.3f}** percentage points, substantially better than either short-trained model. Mean training-field climatology has MAE **{result['training_mean_field']['mae']*100:.3f}** and RMSE **{result['training_mean_field']['rmse']*100:.3f}**. This is a limited experiment, not evidence of an operationally useful forecast model. The two-seed means above are arithmetic means of the per-seed metrics, not significance estimates or pooled confidence intervals.

## Controlled comparison

The experiment uses the repository's actual `DiffusionProcessor`, conditional U-Net and DDIM sampler on an observed OSI SAF sea-ice concentration slice. Each seed's two arms start with identical hashed state dictionaries. Training timestep draws, Gaussian noise and sampling noise are paired. Only `x0_scale` changes between the two arms.

The original processor from base `{result['baseline_ref']}` is loaded separately. At scale 1.0, the PR matches that original code bit-for-bit for training loss, training prediction and sampling with identical weights and random seeds. Both scale arms are then trained from the same initial checkpoint for 120 updates; the final checkpoint is evaluated without selecting on validation or test scores.

- Observations cover 1–10 January 2024 at 12:00 UTC, on the native 25 km grid. A 64×64 crop is centred on the grid point nearest 78°N, 25°E, fixed before either run.
- Five training pairs forecast target dates 2–6 January from the preceding day's observation. Mean, standard deviation and active/ocean mask are derived only from observations on 1–6 January.
- Validation target is 7 January. Held-out target dates are 8, 9 and 10 January, each with a one-day lead and its own observed previous-day input. These are three rolling-origin hindcasts, not a three-day autoregressive forecast from one origin.
- The encoder and decoder are fixed affine transforms using training-only mean and standard deviation. A fixed ocean-mask channel accompanies the SIC history. This is not an experiment with a pretrained production encoder.
- Two seeds, 477 and 478; 901,217 trainable parameters; AdamW with learning rate 0.001 and weight decay 0.0001; full training batch; gradient clipping at 1.0; identical masked velocity loss.
- 32 training diffusion steps and 32 deterministic DDIM sampling steps, eta 0. Four matched-noise ensemble members are averaged per forecast.

## Prediction figures

The first held-out date and first predeclared seed are shown, not the most favourable case. These are native-grid array views, not projected geographic maps. Pixels excluded by the fixed mask are blank. All three figures share a 0–100% display scale. Forecast values are clipped only for these figures; raw metrics retain out-of-range values.

![Observed SIC on 8 January 2024](figures/observed-20240108.png)

![Scale 1 forecast on 8 January 2024](figures/scale1-20240108.png)

![Scale 2 forecast on 8 January 2024](figures/scale2-20240108.png)

![Raw held-out one-day errors with persistence baseline](figures/daily-mae.png)

## Reproduction

The branch retains the exact PR source at `{result['source_head']}` and adds only this evidence directory. Install the repository development environment, then from the repository root run:

```sh
python validation/x0scale-478/run_comparison.py \\
  --input validation/x0scale-478/input_slice.npz \\
  --design validation/x0scale-478/design.json \\
  --output /path/to/local/results \\
  --baseline-ref {result['baseline_ref']}
```

Use a complete clone containing the baseline commit. `run_comparison.py` writes local checkpoints and metrics to the supplied output directory and performs the base-equivalence checks. No external service, experiment-tracking endpoint or remote training job is required. The checked-in input slice has SHA-256 `{result['design']['input_sha256']}`. Reproduction across Torch versions or hardware may differ despite deterministic settings; the recorded run used Torch {result['torch']}, NumPy {result['numpy']} and macOS CPU.

`design.json`, `metrics.csv`, `results.json` and `predictions.npz` contain inspectable configuration, state hashes, numerical results and forecast arrays. Large model checkpoints are deliberately excluded from Git. The small input slice is derived from the cached public EUMETSAT/OSI SAF dataset, attributed under CC-BY-4.0 as recorded in the repository ingestion configuration. The source filename template and processing details are retained in `design.json`. No private observational dataset or credentials are included.

## Limits

Ten winter observation dates, five training targets, three correlated test dates, one spatial crop, two seeds and a short training budget do not establish a reliable effect size or generalisation across seasons, regions or operational checkpoints. The existing default matches the old code; changing the experimental scaling knob is not justified as a forecast-skill improvement by this pilot. A multi-year benchmark with the intended learned encoder would be needed for that claim.
'''
(ROOT/'README.md').write_text(text)
(ROOT/'summary.json').write_text(json.dumps({'means':means,'persistence':persistence,'n_mask_pixels':int(mask.sum()),'raw_mae_worse_both_seeds':all(next(r for r in result['runs'] if r['seed']==s and r['scale']==2)['test_raw']['mae']>next(r for r in result['runs'] if r['seed']==s and r['scale']==1)['test_raw']['mae'] for s in [477,478])},indent=2))
print(json.dumps({'means':means,'persistence':persistence,'figures':[x.name for x in figures.iterdir()]},indent=2))
