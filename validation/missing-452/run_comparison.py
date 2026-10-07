"""Controlled observed-data loader/training/inference pilot for CryoCast PR 452."""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import statistics
import subprocess
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from cryocast.data import CombinedDataset, SingleDataset
from cryocast.models import EncodeProcessDecode
from cryocast.types import Hemisphere


def datasets(allow,phase,missing):
    limits={'train':('2024-01-01','2024-01-06'),'test':('2024-01-07','2024-01-10')}
    start,end=limits[phase]
    single=[SingleDataset(name,[ROOT/filename],date_ranges=[{'start':start,'end':end}],variables=['ice_conc'],normalise=False) for name,filename in [('target-sic','target-north.zarr'),('conditioning','conditioning-north.zarr')]]
    if missing:
        dropped=D['withheld_training_conditioning_dates' if phase=='train' else 'withheld_test_conditioning_dates']
        single[1].dates=[day for day in single[1].dates if str(day.astype('datetime64[D]')) not in dropped]
    combined=CombinedDataset(single,target_group_name='target-sic',target_variables=['ice_conc'],n_history_steps=1,n_forecast_steps=1,allow_missing_inputs=allow)
    return combined

def model_for(ds):
    encoders={'latent_space':[32,32],**{s.name:{'_target_':'cryocast.models.encoders.NaiveLinearEncoder'} for s in ds.inputs}}
    return EncodeProcessDecode(encoders=DictConfig(encoders),processor=DictConfig({'_target_':'cryocast.models.processors.UNetProcessor','start_out_channels':8}),decoder=DictConfig({'_target_':'cryocast.models.decoders.NaiveLinearDecoder','restrict_range':'sigmoid','mask_type':None}),target_variable_indices=[0],hemisphere=Hemisphere.NORTH,input_spaces=[s.space.to_dict() for s in ds.inputs],output_space=ds.target.space.to_dict(),n_history_steps=1,n_forecast_steps=1,name='missing-input-observed-pilot',metrics=[],loss=DictConfig({'_target_':'torch.nn.MSELoss'}),optimizer=DictConfig({'_target_':'torch.optim.Adam','lr':0.001}),scheduler=DictConfig({}),lr_scheduler=DictConfig({}))

def unpack(batch):
    inputs={k:v.float() for k,v in batch.items() if k!='target'}
    target=batch['target'].float()
    assert torch.isfinite(target).all() and all(torch.isfinite(v).all() for v in inputs.values())
    return inputs,target

def digest(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        h.update(name.encode());h.update(str((value.dtype,tuple(value.shape))).encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def scores(model,ds):
    output=[];truth=[]
    with torch.inference_mode():
        for b in DataLoader(ds,batch_size=1,num_workers=0):
            x,y=unpack(b);output.append(model(x).numpy()[0,0,0]);truth.append(y.numpy()[0,0,0])
    p=np.stack(output);y=np.stack(truth);diff=p[:,mask]-y[:,mask]
    daily=[{'date':str(ds.get_forecast_steps(day)[0]),'mae':float(np.abs(a[mask]-b[mask]).mean()),'rmse':float(np.sqrt(np.square(a[mask]-b[mask]).mean()))} for day,a,b in zip(ds.dates,p,y,strict=True)]
    return {'mae':float(np.abs(diff).mean()),'rmse':float(np.sqrt(np.square(diff).mean())),'per_date':daily},p


def main():
    global ROOT, D, observed, mask, dates
    ROOT=Path(__file__).parent
    D=json.loads((ROOT/'design.json').read_text())
    assert hashlib.sha256((ROOT/'input_observations.npz').read_bytes()).hexdigest()==D['input_sha256']
    with np.load(ROOT/'input_observations.npz',allow_pickle=False) as z:
        observed=z['target'];mask=z['mask'];dates=z['dates']
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


    train={k:datasets(v,'train',True) for k,v in [('strict',False),('fill',True)]}
    complete={k:datasets(v,'test',False) for k,v in [('strict',False),('fill',True)]}
    missing_test={k:datasets(v,'test',True) for k,v in [('strict',False),('fill',True)]}
    assert complete['strict'].dates==complete['fill'].dates
    for i in range(len(complete['strict'])):
        a,b=complete['strict'][i],complete['fill'][i]
        for key in a:np.testing.assert_array_equal(a[key],b[key])
    # Compare strict behavior with the unchanged upstream CombinedDataset class.
    repo=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],text=True).strip())
    base='57057d009c485911eb99b59fc99708ae53af8870'
    reference_source=subprocess.check_output(['git','show',base+':cryocast/data/combined_dataset.py'],cwd=repo)
    ref_file=ROOT/'upstream_combined_dataset.py';ref_file.write_bytes(reference_source)
    spec=importlib.util.spec_from_file_location('cryocast.data._missing_reference',ref_file);reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
    ref=reference.CombinedDataset(complete['strict'].inputs,target_group_name='target-sic',target_variables=['ice_conc'],n_history_steps=1,n_forecast_steps=1)
    assert ref.dates==complete['strict'].dates
    for i in range(len(ref)):
        for key,value in ref[i].items():np.testing.assert_array_equal(value,complete['strict'][i][key])
    result={'design':D,'source_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'upstream_reference':base,'complete_strict_equals_base':True,'complete_strict_fill_bit_identical':True,'training_windows':{k:len(v) for k,v in train.items()},'complete_test_windows':len(complete['strict']),'missing_test_windows':{k:len(v) for k,v in missing_test.items()},'torch':torch.__version__,'device':'CPU','threads':2,'runs':[]}
    assert result['training_windows']=={'strict':3,'fill':5},result['training_windows']
    assert result['missing_test_windows']=={'strict':2,'fill':3},result['missing_test_windows']
    mask_t=torch.from_numpy(mask)[None,None,None]
    predictions={}
    print('DATA_READY',json.dumps({k:v for k,v in result.items() if k!='design'}),flush=True)
    for seed in D['seeds']:
        torch.manual_seed(seed);initial=copy.deepcopy(model_for(train['fill']).state_dict());initial_hash=digest(initial)
        for policy in (['strict','fill'] if seed==123 else ['fill','strict']):
            torch.manual_seed(seed);model=model_for(train[policy]);model.load_state_dict(initial);assert digest(model.state_dict())==initial_hash
            model.train();opt=torch.optim.Adam(model.parameters(),lr=0.001)
            loader=DataLoader(train[policy],batch_size=2,shuffle=True,generator=torch.Generator().manual_seed(seed),num_workers=0,drop_last=True)
            it=iter(loader);losses=[];start=time.perf_counter()
            for step in range(D['optimizer_steps']):
                try:b=next(it)
                except StopIteration:it=iter(loader);b=next(it)
                x,y=unpack(b);opt.zero_grad(set_to_none=True);prediction=model(x);weights=mask_t.expand_as(prediction);loss=(prediction-y).square()[weights].mean();assert torch.isfinite(loss)
                loss.backward();opt.step();losses.append(float(loss.detach()))
            duration=time.perf_counter()-start;model.eval()
            a,pa=scores(model,complete['strict']);b,pb=scores(model,missing_test['fill'])
            predictions[f'{seed}_{policy}_complete']=pa;predictions[f'{seed}_{policy}_missing']=pb
            row={'seed':seed,'policy':policy,'initial_hash':initial_hash,'final_hash':digest(model.state_dict()),'steps':len(losses),'training_seconds':duration,'training_ms_per_sample':1000*duration/(2*len(losses)),'complete_test':a,'missing_test_fill':b,'last_loss':losses[-1]}
            result['runs'].append(row);(ROOT/'results.json').write_text(json.dumps(result,indent=2));print('RUN',json.dumps(row),flush=True)
            if seed==123 and policy=='fill':timing_model=model
    # Warm, interleaved microbenchmarks use identical fixed model weights and exact complete inputs.
    timing_model.eval();cached={k:unpack(next(iter(DataLoader(ds,batch_size=1))))[0] for k,ds in complete.items()}
    measurements={'strict':[],'fill':[]}
    with torch.inference_mode():
        for _ in range(5):timing_model(cached['strict'])
        for repeat in range(40):
            for key in (['strict','fill'] if repeat%2==0 else ['fill','strict']):
                t=time.perf_counter();timing_model(cached[key]);measurements[key].append((time.perf_counter()-t)*1000)
    result['model_only_ms_per_sample']={k:statistics.median(v) for k,v in measurements.items()}
    for mode in ['loader_only','loader_plus_model']:
        measurements={'strict':[],'fill':[]}
        for repeat in range(12):
            for key in (['strict','fill'] if repeat%2==0 else ['fill','strict']):
                start=time.perf_counter()
                with torch.inference_mode():
                    for batch in DataLoader(complete[key],batch_size=1,num_workers=0):
                        x,y=unpack(batch)
                        if mode=='loader_plus_model':timing_model(x)
                measurements[key].append((time.perf_counter()-start)*1000/len(complete[key]))
        result[mode+'_ms_per_sample']={k:statistics.median(v) for k,v in measurements.items()}
    diff=observed[6:9,mask]-observed[7:10,mask]
    result['persistence']={'mae':float(np.abs(diff).mean()),'rmse':float(np.sqrt(np.square(diff).mean()))}
    result['complete']=True
    (ROOT/'results.json').write_text(json.dumps(result,indent=2));np.savez_compressed(ROOT/'predictions.npz',**predictions)
    print('COMPLETE',json.dumps({k:v for k,v in result.items() if k not in ['design','runs']}),flush=True)


if __name__ == "__main__":
    main()
