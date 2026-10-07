"""Generate inspectable results for the paired missing-conditioning pilot."""
from pathlib import Path
import csv
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).parent
r=json.loads((ROOT/'results.json').read_text());a=json.loads((ROOT/'first-clean-run.json').read_text())
assert r['complete'] and len(r['runs'])==4
for x,y in zip(a['runs'],r['runs'],strict=True):
    for k in ['seed','policy','initial_hash','final_hash','complete_test','missing_test_fill']:
        assert x[k]==y[k],(x['seed'],k)
with np.load(ROOT/'input_observations.npz',allow_pickle=False) as z:
    truth=z['target'];mask=z['mask'];dates=z['dates']
with np.load(ROOT/'predictions.npz',allow_pickle=False) as z:predictions={k:z[k].copy() for k in z.files}
for row in r['runs']:
    for scenario,key in [('complete','complete_test'),('missing','missing_test_fill')]:
        p=predictions[f'{row["seed"]}_{row["policy"]}_{scenario}'];d=p[:,mask]-truth[7:10,mask]
        assert np.isclose(np.abs(d).mean(),row[key]['mae'])
        assert np.isclose(np.sqrt((d*d).mean()),row[key]['rmse'])
means={p:{s:{m:float(np.mean([x[s][m] for x in r['runs'] if x['policy']==p]))*100 for m in ['mae','rmse']} for s in ['complete_test','missing_test_fill']} for p in ['strict','fill']}
rows=[]
for x in r['runs']:
    rows.append({'seed':x['seed'],'training_policy':x['policy'],'train_seconds':x['training_seconds'],'complete_mae_pp':100*x['complete_test']['mae'],'complete_rmse_pp':100*x['complete_test']['rmse'],'missing_mae_pp':100*x['missing_test_fill']['mae'],'missing_rmse_pp':100*x['missing_test_fill']['rmse']})
with (ROOT/'metrics.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
figures=ROOT/'figures';figures.mkdir(exist_ok=True)
fig,ax=plt.subplots(figsize=(8,4.8),layout='constrained')
for policy,marker in [('strict','o'),('fill','s')]:
    data=np.array([[d['mae']*100 for d in x['missing_test_fill']['per_date']] for x in r['runs'] if x['policy']==policy])
    ax.plot(range(3),data.mean(axis=0),marker=marker,label=policy.capitalize()+' training, fill-enabled test loader')
persist=[100*np.abs(truth[i,mask]-truth[i-1,mask]).mean() for i in [7,8,9]]
ax.plot(range(3),persist,marker='^',linestyle='--',label='Previous-day persistence')
ax.set_xticks(range(3),['8 Jan','9 Jan\nconditioning absent','10 Jan']);ax.set_ylim(bottom=0)
ax.set_ylabel('Masked MAE (percentage points of sea-ice concentration)');ax.set_xlabel('One-day forecast target date in 2024');ax.set_title('Held-out forecasts with one conditioning date withheld');ax.legend(fontsize=8)
fig.savefig(figures/'missing-condition-mae.png',dpi=150);plt.close(fig)
for filename,title,field in [('observed-20240109.png','Observed sea-ice concentration',truth[8]),('strict-trained-missing-20240109.png','Strict-trained model with absent conditioning',predictions['123_strict_missing'][1]),('fill-trained-missing-20240109.png','Fill-trained model with absent conditioning',predictions['123_fill_missing'][1])]:
    fig,ax=plt.subplots(figsize=(6,5.2),layout='constrained');im=ax.imshow(np.ma.array(field*100,mask=~mask),vmin=0,vmax=100,interpolation='nearest');ax.set_title(title+'\n9 January 2024, seed 123',fontsize=12);ax.set_xlabel('Column in 25 km target-grid crop');ax.set_ylabel('Row in 25 km target-grid crop');fig.colorbar(im,ax=ax,label='Sea-ice concentration (%)',shrink=0.82);fig.savefig(figures/filename,dpi=150);plt.close(fig)
table='| Seed | Training policy | Train seconds | Complete-input MAE | Complete-input RMSE | Missing-input MAE | Missing-input RMSE |\n|---|---|---:|---:|---:|---:|---:|\n'
for x in rows:table+=f'| {x["seed"]} | {x["training_policy"]} | {x["train_seconds"]:.3f} | {x["complete_mae_pp"]:.3f} | {x["complete_rmse_pp"]:.3f} | {x["missing_mae_pp"]:.3f} | {x["missing_rmse_pp"]:.3f} |\n'
time_table='| Measurement | Strict path | Fill-enabled path |\n|---|---:|---:|\n'
for key,label in [('model_only_ms_per_sample','Model only, median ms/sample'),('loader_only_ms_per_sample','Loader only, median ms/sample'),('loader_plus_model_ms_per_sample','Loader plus model, median ms/sample')]:time_table+=f'| {label} | {r[key]["strict"]:.3f} | {r[key]["fill"]:.3f} |\n'
text=f'''# Missing-conditioning training and forecast pilot

This adds a fresh, reproducible evaluation of CryoCast PR #452 with an explicit ocean mask and a held-out missing-conditioning case. It supplements the earlier South-pole pilot in the PR discussion; its smaller northern dataset and scores must not be pooled with that earlier result.

## Main result

The fill-enabled loader retains **5 training windows instead of 3** under the fixed two-date conditioning outage. With one held-out conditioning date removed, it retains **all 3 test windows instead of 2**. Mandatory target history and target observations are not removed or imputed.

Mean masked MAE across two seeds is **{means['strict']['complete_test']['mae']:.3f}** percentage points for strict training versus **{means['fill']['complete_test']['mae']:.3f}** for fill-enabled training on complete test inputs. With one conditioning date absent and the same three test forecasts produced through the fill-enabled loader, the corresponding values are **{means['strict']['missing_test_fill']['mae']:.3f}** and **{means['fill']['missing_test_fill']['mae']:.3f}**. The short-trained models remain substantially worse than previous-day persistence, at **{r['persistence']['mae']*100:.3f} MAE** and **{r['persistence']['rmse']*100:.3f} RMSE**.

These are pilot observations, not evidence of production forecast improvement. Only three correlated held-out dates and two training seeds are evaluated. The default remains strict; this evidence supports keeping the capability opt-in while establishing that the real loader/model path runs with missing conditioning.

## Exact scores

Errors are percentage points of sea-ice concentration, evaluated over the same fixed {int(mask.sum())} active-ocean pixels on 8, 9 and 10 January 2024. No clipping was used for metrics; the model decoder already enforces its configured sigmoid range.

{table}

For the missing-input score, **both trained models use the fill-enabled loader** so the evaluation covers exactly the same three targets. An ordinary strict loader would drop the 9 January target when its 8 January conditioning observation is absent. Its reduced two-window average is deliberately not compared against the three-window average. This distinguishes training-policy effects from unequal test coverage.

## Timing

{time_table}

Both policies receive the same 200 optimizer steps, batch size two, and identical initial weights per seed. Training order is reversed for the second seed. Model-only and loader-plus-model timing use the same fixed trained model and complete-input windows; all tensors from the strict and fill-enabled complete-input paths are verified bit-identical. The model-only measurement uses five warmups and 40 interleaved calls; loader measurements use 12 alternating passes. The unmodified upstream strict loader also produces exactly the same dates and tensor values.

These are local macOS CPU measurements with two PyTorch threads and other development activity on the machine, not isolated benchmark guarantees. A repeat run reproduced parameter hashes and forecast scores exactly but had substantially different elapsed times. The timing evidence therefore does not establish a precise percentage speedup or slowdown. Raw per-seed times and both timing runs are retained in the JSON files.

## Data and controls

- Ten days, 1–10 January 2024 at noon, from the existing public EUMETSAT/OSI SAF sea-ice concentration cache and C3S CARRA2 sea-ice concentration reanalysis cache.
- Fixed 24×24 target-grid crop around 78°N, 25°E. CARRA2 is mapped to these target points by nearest neighbour on the unit sphere. The retained points are at most {r['design']['max_included_distance_km']:.3f} km from their source neighbour.
- Mask derives only from 1–6 January target status flags and finite training-source values, intersected with fixed geographic coverage. Excluded pixels are zero in both input stores. There is no future-data-derived mask selection.
- Physical concentration fractions already lie in [0,1]; `SingleDataset(normalise=False)` avoids using any fitted normalization statistics.
- Train targets are 2–6 January. Conditioning observations on 2 and 4 January are withheld before constructing date-to-index caches. Target history and targets remain complete. The validation date 7 January is not used for checkpoint selection.
- Test targets are 8–10 January, each with one day of observed history and one-day lead. Conditioning on 8 January is withheld for the missing-test scenario. These are rolling-origin hindcasts, not an autoregressive three-day trajectory.
- Actual `CombinedDataset`, `SingleDataset`, `EncodeProcessDecode`, `NaiveLinearEncoder`, `UNetProcessor` and `NaiveLinearDecoder` are used. Latent grid 32×32, eight starting U-Net channels, Adam 0.001. Each seed's two treatments share initial weights and receive exactly 200 updates. Seeds are 123 and 456.

CARRA2 is reanalysis, not an independent real-time observation stream. This experiment does not validate operational data-arrival delays or independence of data sources. The earlier 2020–2021/2024 South-pole pilot remains separately documented at https://github.com/alan-turing-institute/cryocast/pull/452#issuecomment-5622298939.

## Forecast figures

The missing-conditioning target date and first predetermined seed are shown. These are array-grid plots, not a georeferenced map. All panels use the same 0–100% display scale, and masked pixels are blank.

![Errors by forecast date](figures/missing-condition-mae.png)

![Observed field](figures/observed-20240109.png)

![Strict-trained model with missing conditioning](figures/strict-trained-missing-20240109.png)

![Fill-trained model with missing conditioning](figures/fill-trained-missing-20240109.png)

## Reproduction

The evidence branch retains PR source `{r['source_head']}`. With its development dependencies available, from the repository root:

```sh
python validation/missing-452/prepare_reader_stores.py
PYTHONPATH=. WANDB_MODE=disabled python validation/missing-452/run_comparison.py
python validation/missing-452/make_evidence.py
```

A complete clone containing upstream commit `{r['upstream_reference']}` is needed for the strict-path equality check. The scripts create only local derived reader stores and result files. No credentials, external downloads, model registry or remote experiment tracker are used. `input_observations.npz` has SHA-256 `{r['design']['input_sha256']}`. Its small derived crop is attributed to EUMETSAT/OSI SAF (CC-BY-4.0) and Copernicus Climate Change Service (C3S) / ECMWF CARRA2 (CC-BY-4.0 as recorded by the source ingestion configuration); source filenames and transformations are recorded in `design.json`. This is a modified sample, not an official reissued data product. No private datasets or trained model checkpoints are included.

The first clean run and the self-contained reader-store rebuild have identical initial/final state hashes, complete/missing test scores and daily metrics for all four arms. The runtime was Torch {r['torch']} on CPU. Cross-platform bitwise equality is not promised.
'''
(ROOT/'README.md').write_text(text)
summary={'means_percentage_points':means,'persistence_percentage_points':{k:100*v for k,v in r['persistence'].items()},'same_results_after_self_contained_rebuild':True,'same_initial_and_final_parameter_hashes':True,'timing_is_not_stable_across_runs':True,'training_windows':r['training_windows'],'missing_test_windows':r['missing_test_windows']}
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
