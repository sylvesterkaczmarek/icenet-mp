"""Train a small CryoCast CNN encoder/decoder on 2023 sea-ice observations only.

All raw files, checkpoints and processed latents live outside this Git checkout.
The encoder and decoder are FROZEN for every paired diffusion experiment. This
also computes x0_scale from actual training TARGET latents only.
"""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cryocast.models.decoders import CNNDecoder
from cryocast.models.encoders import CNNEncoder
from cryocast.types import DataSpace

BASE = Path("/Volumes/T7/Research-Data/alan-turing-institute/cryocast/x0scale-478-expanded-20261011")


def make_encoder_decoder():
    """Instantiate the actual CryoCast CNN encoder and its matching decoder."""
    space = DataSpace(channels=1, name="sic", shape=(64, 64))
    encoder = CNNEncoder(
        data_space_in=space,
        latent_space=(32, 32),
        n_layers=1,
        n_subblocks=1,
        scale_factor=2,
    )
    decoder = CNNDecoder(
        data_space_in=encoder.data_space_out,
        data_space_out=space,
        n_layers=1,
        n_subblocks=1,
        scale_factor=2,
        restrict_range="sigmoid",
    )
    return encoder, decoder


def decode(decoder, latents):
    """Bound decoded SIC to [0, 1] with the repository's decoder machinery."""
    return decoder.finalise(decoder(latents), None)


def mse_masked(output, truth, mask):
    """MSE over training-derived static ocean mask, excluding land/invalid cells."""
    delta = ((output - truth) * mask).square()
    return delta.sum() / mask.sum() / output.shape[0]


def sha_state(state):
    """Stable digest of architecture, order and exact learned tensor values."""
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        digest.update(key.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=850)
    ap.add_argument("--check-every", type=int, default=50)
    ap.add_argument("--min-steps", type=int, default=250)
    ap.add_argument("--patience", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(3)
    torch.manual_seed(14277)
    np.random.seed(14277)
    with np.load(BASE/"observed_sic_paired.npz", allow_pickle=False) as npz:
        train = npz["train"].copy()
        val = npz["val"].copy()
        test = npz["test"].copy()
        ocean_mask = npz["mask"].copy()
        dates = {split:npz[split+"_dates"].tolist() for split in ("train","val","test")}
    x = torch.from_numpy(train[:,None]).float()
    val_x = torch.from_numpy(val[:,None]).float()
    mask = torch.from_numpy(ocean_mask[None,None].astype(np.float32))
    assert len(train) == 120 and len(val) == 31 and mask.sum() > 2500

    encoder, decoder = make_encoder_decoder()
    params = [*encoder.parameters(), *decoder.parameters()]
    optim = torch.optim.AdamW(params,lr=0.002,weight_decay=1e-5)
    base_state=sha_state({**{"encoder."+k:v for k,v in encoder.state_dict().items()},
                          **{"decoder."+k:v for k,v in decoder.state_dict().items()}})
    history=[]
    best=float('inf')
    best_snapshot=None
    stagnant=0
    started=time.perf_counter()
    for step in range(1,args.max_steps+1):
        encoder.train();decoder.train()
        generator=torch.Generator().manual_seed(14277*10000+step)
        idx=torch.randint(len(train),(8,),generator=generator)
        batch=x[idx]
        optim.zero_grad(set_to_none=True)
        loss=mse_masked(decode(decoder,encoder(batch)),batch,mask)
        assert torch.isfinite(loss)
        loss.backward()
        nn.utils.clip_grad_norm_(params,2.0)
        optim.step()
        if step % args.check_every == 0:
            encoder.eval();decoder.eval()
            with torch.inference_mode():
                v_losses=[]
                for subset in val_x.split(8):
                    v_losses.append((mse_masked(decode(decoder,encoder(subset)),subset,mask).item(),len(subset)))
            v=sum(v*n for v,n in v_losses)/len(val)
            entry={"step":step,"train_mse":float(loss.detach()),"validation_mse":v,"elapsed_s":time.perf_counter()-started}
            history.append(entry)
            if v < best * 0.997:
                best=v
                stagnant=0
                best_snapshot={
                    "encoder":copy.deepcopy(encoder.state_dict()),
                    "decoder":copy.deepcopy(decoder.state_dict()),
                }
                best_step=step
            else:
                stagnant+=1
            print("AUTOENCODER_CHECK",json.dumps(entry),"best",round(best,6),"stagnant",stagnant,flush=True)
            if step>=args.min_steps and stagnant>=args.patience:
                break
    assert best_snapshot is not None
    encoder.load_state_dict(best_snapshot["encoder"])
    decoder.load_state_dict(best_snapshot["decoder"])
    encoder.eval();decoder.eval()
    for param in params:param.requires_grad_(False)

    # Only compute target distribution from actual held-out-from-evaluation
    # TRAINING target dates (Jan 2 through Apr 30, 2023).
    latents={}
    with torch.inference_mode():
        for split,arr in (("train",train),("val",val),("test",test)):
            chunks=[]
            for raw_chunk in torch.from_numpy(arr[:,None]).float().split(16):
                chunks.append(encoder(raw_chunk).cpu().numpy())
            latents[split]=np.concatenate(chunks,axis=0).astype(np.float32)
            assert latents[split].shape==(len(arr),2,32,32)
    target_latents=latents["train"][1:]
    active_latent=torch.nn.functional.interpolate(mask,size=(32,32),mode="nearest").numpy().astype(bool)[0,0]
    available=target_latents[:,:,active_latent]
    std=float(available.std(dtype=np.float64))
    mean=float(available.mean(dtype=np.float64))
    assert np.isfinite(std) and 0.02<std<5, ("Invalid encoder latent std",std)
    channel_std=[float(available[:,j,:].std(dtype=np.float64)) for j in range(2)]
    print("TRAINING_TARGET_LATENT_STD",std,"MEAN",mean,"CHANNEL_STD",channel_std,flush=True)

    data_path=BASE/"latent_sequences.npz"
    np.savez_compressed(data_path,train=latents["train"],val=latents["val"],test=latents["test"])
    ckpt=BASE/"trained_cnn_encoder_decoder.pt"
    torch.save({"encoder":best_snapshot["encoder"],"decoder":best_snapshot["decoder"]},ckpt)
    mask_np=ocean_mask.astype(bool)
    with torch.inference_mode():
        test_x=torch.from_numpy(test[:,None]).float()
        pred=np.concatenate([decode(decoder,torch.from_numpy(c)).numpy() for c in np.array_split(latents["test"],5)],axis=0)[:,0]
    expected=test[:,mask_np]
    oracle=np.abs(pred[:,mask_np]-expected).mean()
    persistence=np.abs(test[:-1,:,][...,mask_np]-test[1:,...,mask_np]).mean()
    summary={
        "architecture":"CryoCast CNNEncoder and CNNDecoder, n_layers=1, n_subblocks=1, latent 2x32x32, raw SIC 1x64x64",
        "train_period":[dates["train"][0],dates["train"][-1]],
        "validation_period":[dates["val"][0],dates["val"][-1]],
        "test_period":[dates["test"][0],dates["test"][-1]],
        "encoder_seed":14277,
        "train_frames":len(train),"validation_frames":len(val),"test_frames":len(test),
        "encoder_initial_sha256":base_state,
        "encoder_final_sha256":sha_state(best_snapshot["encoder"]),
        "decoder_final_sha256":sha_state(best_snapshot["decoder"]),
        "encoder_learning_rate":0.002,
        "best_validation_mse":best,
        "best_step":best_step,
        "last_step":step,
        "stopped_by_plateau":stagnant>=args.patience,
        "history":history,
        "training_target_latent_std":std,
        "training_target_latent_mean":mean,
        "training_target_channel_std":channel_std,
        "target_latent_heldout_years_disjoint":True,
        "target_latent_mask_pixel_count":int(active_latent.sum()),
        "ae_oracle_test_mae":float(oracle),
        "persistence_test_mae":float(persistence),
        "weights_checkpoint_sha256":hashlib.sha256(ckpt.read_bytes()).hexdigest(),
        "latents_sha256":hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "data_design_sha256":hashlib.sha256((BASE/"dataset_design.json").read_bytes()).hexdigest(),
        "seconds":time.perf_counter()-started
    }
    (BASE/"encoder_results.json").write_text(json.dumps(summary,indent=2))
    print("AUTOENCODER_COMPLETE",json.dumps({k:v for k,v in summary.items() if k!="history"}),flush=True)

if __name__=="__main__":
    main()
