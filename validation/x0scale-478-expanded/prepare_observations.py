"""Prepare observed SIC for year-separated, one-day forecast comparisons.

Reads public OSI SAF NetCDF archives downloaded into the research-data directory.
Keeps source files read-only, retains temporal order, records provenance and
takes the same 64x64 north-Atlantic crop as the October 6 small pilot.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr

BASE = Path("/Volumes/T7/Research-Data/alan-turing-institute/cryocast/x0scale-478-expanded-20261011")
EVIDENCE = Path("/Volumes/T7/GitHub-Contributions/alan-turing-institute/icenet-mp/evidence-478-20261006-2213/validation/x0scale-478")
ROWS, COLS = slice(232, 296), slice(206, 270)


def load_split(manifest: dict, split: str):
    records = sorted((x for x in manifest["files"] if x["split"] == split), key=lambda r: r["date"])
    fields = []
    valid = []
    for i, record in enumerate(records, 1):
        raw = BASE / record["file"]
        if hashlib.sha256(raw.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Checksum changed: {raw}")
        with xr.open_dataset(raw, engine="netcdf4") as ds:
            sea_ice = np.asarray(ds["ice_conc"].values[0, ROWS, COLS], dtype=np.float32)
            assert sea_ice.shape == (64, 64), (raw,sea_ice.shape)
            if "status_flag" not in ds:
                raise RuntimeError(f"Missing OSI SAF status metadata in {raw}")
            status = np.asarray(ds["status_flag"].values[0, ROWS, COLS])
            # OSI SAF bit 7 (128) flags inactive/invalid grid cells; treating
            # a finite status value as valid would include 244 inactive cells.
            status_int = np.nan_to_num(status, nan=128).astype(np.int32)
            valid_day = (
                np.isfinite(sea_ice)
                & np.isfinite(status)
                & ((status_int & 128) == 0)
            )
            # The product stores percent sea ice, not fraction.
            assert not np.any((sea_ice[valid_day] < 0) | (sea_ice[valid_day] > 100))
            sea_ice = np.where(valid_day, sea_ice / 100.0, 0.0)
            fields.append(sea_ice.astype(np.float32))
            valid.append(valid_day)
    dates = [x["date"] for x in records]
    print(split,len(dates),"first",dates[0],"last",dates[-1],flush=True)
    return np.stack(fields), np.stack(valid), np.asarray(dates)


def main():
    manifest=json.loads((BASE/"download_manifest.json").read_text())
    train, train_valid, train_dates = load_split(manifest,"training")
    val, val_valid, val_dates = load_split(manifest,"validation")
    test, test_valid, test_dates = load_split(manifest,"testing")
    # Ocean/valid observation selection uses TRAINING ONLY, no validation/test.
    train_availability = np.mean(train_valid, axis=0)
    mask = train_availability >= 0.95
    assert mask.sum() >= 2500, ("Insufficient observed sea-ice cells",mask.sum())
    # Training-only normalization is deliberately NOT applied here: scaling should
    # use empirical encoder latents, not artificially unit-standardised inputs.
    for split, a, v in [('training',train,train_valid),('validation',val,val_valid),('testing',test,test_valid)]:
        available = v & mask[None]
        frac=float(available.sum())/(len(a)*mask.sum())
        print(split,'observed_mask_cells',int(mask.sum()),'valid_fraction',round(frac,5),'minmax',float(a.min()),float(a.max()),flush=True)
        assert frac>0.9
        assert np.isfinite(a).all()

    # Confirm the same observed Jan 2024 source and spatial crop as the prior pilot.
    with np.load(EVIDENCE/'input_slice.npz',allow_pickle=False) as prior:
        old=prior['sic'][0]
        old_mask=prior['mask'].astype(bool)
        shared=old_mask & val_valid[0]
        # Different years have different active-status histories. The mask is
        # deliberately derived from 2023 only; do not select it on 2024.
        print("MASK_VS_OLD_PILOT", "training_active", int(mask.sum()),
              "prior_jan2024_active", int(old_mask.sum()),
              "train_only_cells", int((mask & ~old_mask).sum()),
              "old_only_cells", int((old_mask & ~mask).sum()), flush=True)
        reference_diff=float(np.max(np.abs(old[shared]-val[0][shared])))
        print('OLD_PILOT_COMMON_JAN01_PIXELS',int(shared.sum()),'MAX_ABS_DIFF',reference_diff,flush=True)
        assert reference_diff<0.0001, 'Data source/crop unexpectedly changed from prior experiment'

    prepared = BASE/"observed_sic_paired.npz"
    np.savez_compressed(prepared,train=train,val=val,test=test,
        train_valid=train_valid,val_valid=val_valid,test_valid=test_valid,
        train_dates=train_dates,val_dates=val_dates,test_dates=test_dates,mask=mask)
    design = {
        'source':'OSI SAF OSI-438 AMSR v3p0 public NetCDF, 25km northern EASE2 grid',
        'raw_manifest_sha256':hashlib.sha256((BASE/'download_manifest.json').read_bytes()).hexdigest(),
        'prepared_sha256':hashlib.sha256(prepared.read_bytes()).hexdigest(),
        'observations':{'train':int(len(train)),'validation':int(len(val)),'test':int(len(test))},
        'pairs':{'train':int(len(train)-1),'validation':int(len(val)-1),'test':int(len(test)-1)},
        'years_disjoint':True, 'train_cutoff':str(train_dates[-1]),
        'crop_rows':[232,296],'crop_cols':[206,270],
        'ocean_mask_from_training_only':True,'excluded_status_flag_bit_128':True,
        'ocean_mask_pixels':int(mask.sum()),'reference_jan2024_max_abs_error':reference_diff,
        'normalization':'raw fractional SIC [0,1] with training-only target encoder latent std computed in training script',
        'license':'OSI SAF CC-BY-4.0', 'raw_archive_retained_outside_repository':True
    }
    (BASE/'dataset_design.json').write_text(json.dumps(design,indent=2))
    print('PREPARED',prepared,'sha256',design['prepared_sha256'],'shape',train.shape,val.shape,test.shape,flush=True)

if __name__=='__main__':
    main()
