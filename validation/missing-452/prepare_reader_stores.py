"""Create local Anemoi reader stores from the attributed public-data crop."""
from pathlib import Path
import hashlib
import json
import numpy as np
import zarr

def main():
    root=Path(__file__).parent
    design=json.loads((root/'design.json').read_text())
    data_file=root/'input_observations.npz'
    assert hashlib.sha256(data_file.read_bytes()).hexdigest()==design['input_sha256']
    with np.load(data_file,allow_pickle=False) as data:
        dates=data['dates'];lat=data['latitude'];lon=data['longitude']
        for name,key in [('target','target'),('conditioning','conditioning')]:
            values=data[key].astype(np.float32)
            n,h,w=values.shape
            out=zarr.open_group(str(root/(name+'-north.zarr')),mode='w',zarr_version=2)
            out.attrs.update({'version':'0.21','variables':['ice_conc'],'field_shape':[h,w],'shape':[n,1,1,h*w],'frequency':'1d','start_date':str(dates[0]),'end_date':str(dates[-1]),'statistics_start_date':str(dates[0]),'statistics_end_date':str(dates[5]),'missing_dates':[],'resolution':'25km-target-grid','order_by':['valid_datetime','param_level','number']})
            arrays={'data':values.reshape(n,1,1,h*w),'dates':dates,'latitudes':lat.ravel(),'longitudes':lon.ravel(),'mean':np.array([values[:6].mean()]),'stdev':np.array([values[:6].std()]),'minimum':np.array([values[:6].min()]),'maximum':np.array([values[:6].max()])}
            for key,array in arrays.items():out.create_dataset(key,data=array,shape=array.shape,dtype=array.dtype,overwrite=True)
            print('Created',name,values.shape)

if __name__=='__main__':
    main()
