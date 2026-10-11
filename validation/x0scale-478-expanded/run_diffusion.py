"""Paired observed-data comparisons of CryoCast diffusion x0 scaling.

Uses an independently pretrained, frozen CryoCast CNN encoder/decoder. The
normalizing scale is the empirical standard deviation of TRAIN TARGET LATENTS.
Training/validation/test years do not overlap; all 2024 targets are held out.
Every scale pair shares exact starting weights, training minibatches, timesteps,
diffusion noise and sampling RNGs. No PR implementation code is modified.
"""

import argparse
import copy
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cryocast.models.processors import DiffusionProcessor
from cryocast.types import DataSpace

BASE = Path("/Volumes/T7/Research-Data/alan-turing-institute/cryocast/x0scale-478-expanded-20261011")
SEEDS = (477, 478)
TRAIN_TIMESTEPS = (16, 32, 64)
MAX_UPDATES = 5000
CHECK_INTERVAL = 50
PATIENCE = 10
MIN_UPDATES = 800
BATCH = 8
LR = 0.001
SCALE_VARIANTS = ("raw", "empirical")


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def state_hash(state):
    digest=hashlib.sha256()
    for name, weight in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str((weight.dtype, tuple(weight.shape))).encode())
        digest.update(weight.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def make_processor(training_steps, scale):
    latent = DataSpace(channels=2, name="target_latent", shape=(32, 32))
    return DiffusionProcessor(
        data_space=latent,
        data_space_target=latent,
        target_channel_offset=0,
        n_history_steps=1,
        n_forecast_steps=1,
        timesteps=training_steps,
        ddim_steps=training_steps,
        eta=0.0,
        x0_scale=scale,
        start_out_channels=8,
        time_embed_dim=256,
        dropout_rate=0.0,
        use_autoregressive=True,
        loss=nn.MSELoss(),
    )


def count_pair_metrics(pred, truth, mask):
    """Compute physical SIC skill on the exact same training-derived ocean cells."""
    assert pred.shape == truth.shape and pred.ndim == 3
    diff=(pred-truth)[:,mask].astype(np.float64)
    return {
        "mae_percentage_points":float(np.abs(diff).mean()*100),
        "rmse_percentage_points":float(np.sqrt(np.square(diff).mean())*100),
        "bias_percentage_points":float(diff.mean()*100),
        "binary_ice_area_abs_error_cells":float(np.abs((pred[:,mask]>=0.15).sum(1)-(truth[:,mask]>=0.15).sum(1)).mean()),
    }


def validate_model(model, val_in, val_out, seed):
    """Evaluate validation v-loss with fixed timestep/noise samples."""
    model.eval()
    values=[]
    with torch.inference_mode():
        for index in range(0,len(val_in),BATCH):
            torch.manual_seed(606060+seed*100+index)
            output=model.rollout(val_in[index:index+BATCH],val_out[index:index+BATCH])
            assert output.loss is not None and torch.isfinite(output.loss)
            values.append((float(output.loss),len(val_in[index:index+BATCH])))
    return sum(loss*n for loss,n in values)/len(val_in)


def evaluate_forecast(model, decoder, test_in, test_raw, mask, seed, timestep):
    """Measure each permitted deterministic DDIM sampler with paired noise."""
    steps = sorted({4, 8, 16, min(32,timestep), timestep})
    steps = [s for s in steps if s<=timestep]
    forecasts={}
    metrics={}
    with torch.inference_mode():
        for num_steps in steps:
            model.set_sampler(ddim_steps=num_steps,eta=0.0)
            model.eval()
            preds=[]
            start=time.perf_counter()
            for index in range(0,len(test_in),8):
                torch.manual_seed(818181+seed*10000+timestep*100+index+num_steps*1000)
                latents=model.rollout(test_in[index:index+8]).prediction[:,0]
                phys=decoder.finalise(decoder(latents),None).cpu().numpy()[:,0]
                preds.append(phys)
            predictions=np.concatenate(preds,axis=0)
            elapsed=time.perf_counter()-start
            assert np.isfinite(predictions).all()
            assert predictions.min()>=0 and predictions.max()<=1
            assert len(predictions)==len(test_raw)-1
            forecasts[str(num_steps)]=predictions
            met=count_pair_metrics(predictions,test_raw[1:],mask)
            met["sampling_and_decoding_wall_seconds"]=elapsed
            met["test_target_dates"]=int(len(predictions))
            met["per_date_mae_percentage_points"]=[
                float(abs(p[mask]-y[mask]).mean()*100)
                for p,y in zip(predictions,test_raw[1:],strict=True)
            ]
            metrics[str(num_steps)]=met
    return forecasts,metrics


def train_one(seed,timestep,scale,train_in,train_out,val_in,val_out,
              decoder,test_in,test_raw,mask,initial,initial_hash,design_sha,source_sha,encoder_sha,driver_sha):
    name=f"seed-{seed}-T-{timestep}-scale-{scale:.8g}"
    output=BASE/"training_results"/name
    output.mkdir(parents=True,exist_ok=True)
    finished=output/"metrics.json"
    if finished.exists():
        record=json.loads(finished.read_text())
        assert record["design_sha256"]==design_sha
        assert record["processor_sha256"]==source_sha
        assert record["encoder_sha256"]==encoder_sha
        assert record["driver_sha256"]==driver_sha
        assert record["initial_state_sha256"]==initial_hash
        print("REUSE_VERIFIED",name,flush=True)
        return record

    torch.manual_seed(seed*100+timestep)
    model=make_processor(timestep,scale)
    model.load_state_dict(copy.deepcopy(initial))
    assert state_hash(model.state_dict())==initial_hash
    optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
    best_loss=float('inf');best_state=None;best_step=0;stagnant=0
    history=[]
    start=time.perf_counter()
    for step in range(1,MAX_UPDATES+1):
        model.train()
        # Identical minibatch draws for both scale arms.
        indices=torch.randint(len(train_in),(BATCH,),generator=torch.Generator().manual_seed(seed*100000+step))
        xb=train_in[indices];yb=train_out[indices]
        # Identical diffusion random draws and timesteps for both scale arms.
        torch.manual_seed(seed*500000+timestep*10000+step)
        optimizer.zero_grad(set_to_none=True)
        rollout_result=model.rollout(xb,yb)
        loss=rollout_result.loss
        assert loss is not None and torch.isfinite(loss),(name,step)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()

        if step % CHECK_INTERVAL==0:
            v=validate_model(model,val_in,val_out,seed+timestep)
            entry={"step":step,"training_v_mse":float(loss.detach()),"validation_v_mse":v,"elapsed_seconds":time.perf_counter()-start}
            history.append(entry)
            if v < best_loss*0.995:
                best_loss=v
                best_step=step
                best_state=copy.deepcopy(model.state_dict())
                stagnant=0
            else:
                stagnant+=1
            print("DIFFUSION_CHECK",name,json.dumps(entry),"best",round(best_loss,7),"stagnant",stagnant,flush=True)
            # Require sufficient learning before early stopping. All scale arms
            # are allowed their own validation-based stop, with explicit budgets.
            if step>=MIN_UPDATES and stagnant>=PATIENCE:
                break
    assert best_state is not None
    train_wall=time.perf_counter()-start
    did_stop=stagnant>=PATIENCE
    model.load_state_dict(best_state)
    model.eval()
    final_hash=state_hash(best_state)
    torch.save(best_state,output/"best_processor.pt")
    forecast,metrics=evaluate_forecast(model,decoder,test_in,test_raw,mask,seed,timestep)
    np.savez_compressed(output/"test_predictions.npz",**forecast)
    record={
        "seed":seed,"training_timesteps":timestep,"x0_scale":scale,
        "initial_state_sha256":initial_hash,"best_checkpoint_sha256":final_hash,
        "design_sha256":design_sha,"processor_sha256":source_sha,
        "encoder_sha256":encoder_sha,"driver_sha256":driver_sha,
        "training_examples":len(train_in),"validation_examples":len(val_in),
        "testing_examples":len(test_in),
        "trained_updates":step,"best_validation_update":best_step,
        "best_validation_v_mse":best_loss,
        "converged_by_plateau":did_stop,
        "stopping_reason":"validation_plateau" if did_stop else "max_updates_reached",
        "plateau_check_count":stagnant,"training_wall_seconds":train_wall,
        "gradient_clip_norm":1.0,"optimizer":"AdamW","learning_rate":LR,
        "batch_size":BATCH,
        "validation_evaluated_on":"2024 January, time-separated, fixed stochastic noise",
        "history":history,"samplers":metrics,
    }
    finished.write_text(json.dumps(record,indent=2))
    print("DIFFUSION_FINISHED",name,"best_step",best_step,"run_steps",step,
          "status",record["stopping_reason"],"mean_mae",metrics[str(timestep)]["mae_percentage_points"],flush=True)
    return record


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seeds",type=int,nargs="*",default=list(SEEDS))
    ap.add_argument("--timesteps",type=int,nargs="*",default=list(TRAIN_TIMESTEPS))
    args=ap.parse_args()
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    source_sha=sha_bytes((Path(__file__).parents[2]/"cryocast/models/processors/diffusion.py").read_bytes())
    driver_sha=sha_bytes(Path(__file__).read_bytes())
    data_design_sha=sha_bytes((BASE/"dataset_design.json").read_bytes())
    ae_result=json.loads((BASE/"encoder_results.json").read_text())
    assert ae_result["data_design_sha256"]==data_design_sha
    assert ae_result["latents_sha256"]==sha_bytes((BASE/"latent_sequences.npz").read_bytes())
    with np.load(BASE/"observed_sic_paired.npz",allow_pickle=False) as f:
        assert sha_bytes((BASE/"observed_sic_paired.npz").read_bytes())==json.loads((BASE/"dataset_design.json").read_text())["prepared_sha256"]
        physical_test=f["test"].copy()
        mask=f["mask"].copy()
        test_dates=f["test_dates"].tolist()
    with np.load(BASE/"latent_sequences.npz",allow_pickle=False) as f:
        train=f["train"].copy()
        val=f["val"].copy()
        test=f["test"].copy()
    train_in=torch.from_numpy(train[:-1,None]);train_out=torch.from_numpy(train[1:,None])
    val_in=torch.from_numpy(val[:-1,None]);val_out=torch.from_numpy(val[1:,None])
    test_in=torch.from_numpy(test[:-1,None])
    assert train_in.shape==(119,1,2,32,32) and val_in.shape==(30,1,2,32,32)
    assert test_in.shape==(28,1,2,32,32)
    scale=ae_result["training_target_latent_std"]
    assert 0.02<scale<5 and scale!=1.0

    # The decoder and encoder weights are trained on 2023 only and remain frozen.
    from train_encoder import make_encoder_decoder
    encoder,decoder=make_encoder_decoder()
    checkpoint=torch.load(BASE/"trained_cnn_encoder_decoder.pt",map_location="cpu",weights_only=True)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    encoder.eval();decoder.eval()
    for p in decoder.parameters():p.requires_grad_(False)

    physical_persistence=count_pair_metrics(physical_test[:-1],physical_test[1:],mask)
    results=[]
    for timestep in args.timesteps:
        assert timestep in TRAIN_TIMESTEPS
        for seed in args.seeds:
            torch.manual_seed(seed*100+timestep)
            template=make_processor(timestep,1.0)
            init=copy.deepcopy(template.state_dict())
            initial_hash=state_hash(init)
            # Interleaving treatments controls for machine/clock changes and
            # reports exact training and sampled inference runtime.
            for treatment in SCALE_VARIANTS:
                actual_scale=1.0 if treatment=="raw" else scale
                run=train_one(seed,timestep,actual_scale,train_in,train_out,
                      val_in,val_out,decoder,test_in,physical_test,mask,
                      init,initial_hash,data_design_sha,source_sha,
                      ae_result["encoder_final_sha256"],driver_sha)
                run["treatment"]=treatment
                results.append(run)
                (BASE/f"diffusion_progress_T{timestep}.json").write_text(json.dumps({
                    "observations":180,
                    "training_pairs":119,"validation_pairs":30,"testing_pairs":28,
                    "latent_std":scale,"persistence":physical_persistence,
                    "runs_completed":len(results),
                    "runs":[{"seed":r["seed"],"T":r["training_timesteps"],"scale":r["x0_scale"],"best_step":r["best_validation_update"],"converged":r["converged_by_plateau"],"best_validation_v_mse":r["best_validation_v_mse"]} for r in results]
                },indent=2))
    summary={
        "source_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "source_diffusion_sha256":source_sha,"driver_sha256":driver_sha,
        "design_sha256":data_design_sha,
        "reference_pilot":"PR 478 pilot was 5 training pairs and 3 test pairs; this is 119 train, 30 validation, 28 test pairs",
        "encoder_sha256":ae_result["encoder_final_sha256"],
        "encoder_latent_std":scale,
        "encoder_trained_on_training_year_only":True,
        "training_timesteps":list(args.timesteps),"seeds":list(args.seeds),
        "training_parameters":{"max_steps":MAX_UPDATES,"check_every":CHECK_INTERVAL,"patience":PATIENCE,"min_steps":MIN_UPDATES,"batch":BATCH,"lr":LR},
        "test_dates":test_dates[1:],
        "test_mask_pixels":int(mask.sum()),
        "persistence":physical_persistence,
        "runs":results
    }
    suffix="T"+"-".join(str(t) for t in args.timesteps)+"-seed"+"-".join(str(s) for s in args.seeds)
    (BASE/f"diffusion_results_{suffix}.json").write_text(json.dumps(summary,indent=2))
    print("DIFFUSION_ALL_COMPLETE",len(results),"runs",flush=True)


if __name__=="__main__":
    main()
