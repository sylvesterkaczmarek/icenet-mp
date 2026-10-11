"""Summarize all predeclared experiments, with no selection on test performance."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path("/Volumes/T7/Research-Data/alan-turing-institute/cryocast/x0scale-478-expanded-20261011")
SEEDS = (477, 478, 479)
TIMESTEPS = (16, 32, 64)


def load():
    runs={}
    for p in sorted((BASE/"training_results").glob("*/metrics.json")):
        result=json.loads(p.read_text())
        key=(result["seed"],result["training_timesteps"],result["x0_scale"])
        assert key not in runs
        runs[key]=result
    assert len(runs)==18, ("Expected 3 seeds x 3 training T x 2 scales",len(runs))
    enc=json.loads((BASE/"encoder_results.json").read_text())
    scale=enc["training_target_latent_std"]
    for seed in SEEDS:
        for t in TIMESTEPS:
            a,b=runs[(seed,t,1.0)],runs[(seed,t,scale)]
            for attr in ("initial_state_sha256","encoder_sha256","driver_sha256","design_sha256"):
                assert a[attr]==b[attr],(seed,t,attr)
            assert a["training_examples"]==119 and a["validation_examples"]==30 and a["testing_examples"]==28
    return runs,enc


def aggregate(runs,enc):
    scale=enc["training_target_latent_std"]
    with np.load(BASE/"observed_sic_paired.npz",allow_pickle=False) as data:
        raw_test=data["test"]
        mask=data["mask"]
        test_dates=data["test_dates"].tolist()
    assert len(test_dates)==29 and mask.sum()>=2500
    difference=(raw_test[1:,mask]-raw_test[:-1,mask]).astype(np.float64)
    persistence_mae=float(np.abs(difference).mean()*100)
    persistence_rmse=float(np.sqrt(np.square(difference).mean())*100)
    rows=[]
    for seed in SEEDS:
        for t in TIMESTEPS:
            for treatment,sigma in (("raw",1.0),("training-std",scale)):
                run=runs[(seed,t,sigma)]
                for s,forecast in sorted(run["samplers"].items(),key=lambda x:int(x[0])):
                    rows.append({
                        "seed":seed,"training_timesteps":t,"treatment":treatment,
                        "x0_scale":sigma,"ddim_sampling_steps":int(s),
                        "test_dates":28,
                        "mae_pp":forecast["mae_percentage_points"],
                        "rmse_pp":forecast["rmse_percentage_points"],
                        "bias_pp":forecast["bias_percentage_points"],
                        "cropped_binary_ice_area_error_cells":forecast["binary_ice_area_abs_error_cells"],
                        "inference_seconds_for_28_dates":forecast["sampling_and_decoding_wall_seconds"],
                        "training_seconds":run["training_wall_seconds"],
                        "trained_updates":run["trained_updates"],
                        "best_validation_update":run["best_validation_update"],
                        "best_validation_v_mse":run["best_validation_v_mse"],
                        "reached_validation_plateau":run["converged_by_plateau"]
                    })
    with (BASE/"metrics_all.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
        w.writeheader();w.writerows(rows)

    comparison=[]
    for t in TIMESTEPS:
        for s in sorted({int(k) for k in runs[(SEEDS[0],t,1.0)]["samplers"]}):
            samples=[]
            for seed in SEEDS:
                control=runs[(seed,t,1.0)]["samplers"][str(s)]
                scaled=runs[(seed,t,scale)]["samplers"][str(s)]
                per_date=np.asarray(scaled["per_date_mae_percentage_points"])-np.asarray(control["per_date_mae_percentage_points"])
                assert len(per_date)==28
                samples.append({
                    "seed":seed,"raw_mae_pp":control["mae_percentage_points"],
                    "scaled_mae_pp":scaled["mae_percentage_points"],
                    "delta_mae_pp":float(per_date.mean()),
                    "delta_rmse_pp":scaled["rmse_percentage_points"]-control["rmse_percentage_points"],
                    "dates_scaled_better":int((per_date<0).sum()),
                })
            comparison.append({
                "training_timesteps":t,"sampling_steps":s,
                "raw_mae_pp":float(np.mean([v["raw_mae_pp"] for v in samples])),
                "scaled_mae_pp":float(np.mean([v["scaled_mae_pp"] for v in samples])),
                "scaled_minus_raw_mae_pp":float(np.mean([v["delta_mae_pp"] for v in samples])),
                "scaled_minus_raw_rmse_pp":float(np.mean([v["delta_rmse_pp"] for v in samples])),
                "seeds":samples
            })
    summary={
        "experiment":"CryoCast x0_scale on public OSI SAF daily sea-ice data",
        "runs":len(runs),"seeds":list(SEEDS),"training_timesteps":list(TIMESTEPS),
        "train_pairs":119,"validation_pairs":30,"test_pairs":28,
        "test_dates":test_dates[1:],"training_year":2023,"validation_year":2024,
        "training_only_target_latent_std":scale,
        "training_target_latent_mean":enc["training_target_latent_mean"],
        "train_derived_ocean_mask_cells":int(mask.sum()),
        "persistence_mae_pp":persistence_mae,"persistence_rmse_pp":persistence_rmse,
        "encoder_reconstruction_mae_pp":enc["ae_oracle_test_mae"]*100,
        "encoder_pretraining_plateau":enc["stopped_by_plateau"],
        "encoder_pretraining_best_step":enc["best_step"],
        "diffusion_plateau_arms":sum(bool(x["converged_by_plateau"]) for x in runs.values()),
        "diffusion_budget_limited_arms":sum(not x["converged_by_plateau"] for x in runs.values()),
        "paired_comparisons":comparison,
        "limitations":["One 64x64 SIC crop, not hemispheric or multimodal operational forecasts.",
                       "One-day rolling hindcasts on correlated held-out dates.",
                       "Small CNN encoder and decoder; not a pretrained production encoder.",
                       "Three seeds are descriptive, not conclusive statistical evidence.",
                       "Raw and scaled validation v-losses are not directly comparable.",
                       "Concurrent CPU training times are descriptive, not controlled throughput comparisons."]
    }
    (BASE/"summary.json").write_text(json.dumps(summary,indent=2))
    return summary


def make_plots(runs,summary):
    figures=BASE/"figures"
    figures.mkdir(exist_ok=True)
    scale=summary["training_only_target_latent_std"]
    fig,ax=plt.subplots(figsize=(9,5))
    for t in TIMESTEPS:
        pts=[r for r in summary["paired_comparisons"] if r["training_timesteps"]==t]
        x=[p["sampling_steps"] for p in pts]
        ax.plot(x,[p["raw_mae_pp"] for p in pts],marker="o",label=f"T={t}, raw")
        ax.plot(x,[p["scaled_mae_pp"] for p in pts],marker="x",linestyle="--",label=f"T={t}, x0-scaled")
    ax.axhline(summary["persistence_mae_pp"],color="black",linestyle=":",label="Persistence")
    ax.set(xlabel="DDIM inference steps (eta=0)",ylabel="Held-out ocean-mask MAE (percentage points)",title="February 2024 OSI SAF: 3-seed mean")
    ax.set_xscale("log",base=2);ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2)
    fig.tight_layout();fig.savefig(figures/"skill-vs-sampling-steps.png",dpi=170);plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(13,3.6))
    for ax,t in zip(axes,TIMESTEPS,strict=True):
        for seed in SEEDS:
            for name,xscale in (("raw",1.0),("scaled",scale)):
                r=runs[(seed,t,xscale)]
                ax.plot([v["step"] for v in r["history"]],[v["validation_v_mse"] for v in r["history"]],label=f"{seed} {name}",alpha=.7)
        ax.set_xlabel("Optimizer updates");ax.set_title(f"Training T={t}")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Fixed-noise v-loss (assess within each run only)")
    axes[-1].legend(fontsize=7,ncol=2)
    fig.tight_layout();fig.savefig(figures/"validation-curves.png",dpi=170);plt.close(fig)

    # Preselected representative date/configuration, not best cherry-picked test.
    with np.load(BASE/"observed_sic_paired.npz",allow_pickle=False) as data:
        obs=data["test"];mask=data["mask"];dates=data["test_dates"].tolist()
    day=14;t=32;sampler=16;seed=477
    panels=[("Observed",obs[day]),("Persistence",obs[day-1])]
    for title,sigma in (("Raw",1.0),("Training-std scale",scale)):
        folder=BASE/"training_results"/f"seed-{seed}-T-{t}-scale-{sigma:.8g}"
        with np.load(folder/"test_predictions.npz",allow_pickle=False) as data:
            forecast=data[str(sampler)][day-1].copy()
        panels.append((title,forecast))
    fig,axes=plt.subplots(1,4,figsize=(13,3.4))
    for ax,(title,panel) in zip(axes,panels,strict=True):
        ax.imshow(np.ma.masked_where(~mask,panel),vmin=0,vmax=1,cmap="Blues")
        ax.set_title(title);ax.set_axis_off()
    fig.suptitle(f"{dates[day]}: fixed illustrative configuration, seed 477, T=32, DDIM=16")
    fig.tight_layout();fig.savefig(figures/"illustrative-february-forecast.png",dpi=175);plt.close(fig)


def main():
    runs,encoder=load()
    summary=aggregate(runs,encoder)
    make_plots(runs,summary)
    print("REPORT_DONE",len(runs),"runs",len(summary["paired_comparisons"]),"timesteps-sampler comparisons",flush=True)
    print("PERSISTENCE_MAE_PP",summary["persistence_mae_pp"],flush=True)
    print("CONVERGED_ARMS",summary["diffusion_plateau_arms"],"/",len(runs),flush=True)
    for p in summary["paired_comparisons"]:
        print("COMPARE",p["training_timesteps"],p["sampling_steps"],
              "raw",round(p["raw_mae_pp"],4),"scaled",round(p["scaled_mae_pp"],4),
              "delta",round(p["scaled_minus_raw_mae_pp"],4),flush=True)

if __name__=="__main__":
    main()
