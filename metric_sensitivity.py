import os
import glob
import pandas as pd
import numpy as np
import xarray as xr

from RainySeason import *
from utils import *

###################
#### load data ####
###################

# load calibrated parameters of rainy season metrics
pS = pd.read_csv(r'./params/RSO_params_afterOptim.csv',index_col=0)
pE = pd.read_csv(r'./params/RSE_params_afterOptim.csv',index_col=0)

# remove init and AWS
pS = pS.loc[:,~pS.columns.str.contains('init')]
pS = pS.loc[:,~pS.columns.str.contains('AWS')]
pE = pE.loc[:,~pE.columns.str.contains('init')]
pE = pE.loc[:,~pE.columns.str.contains('AWS')]

# load CMIP5 models
cmip_85 = glob.glob(r'S:/EDDY/Lorenz/07_rainyseason/code/CMIP5_cleaned/future_rcp85/*.nc')
cmip_45 = glob.glob(r'S:/EDDY/Lorenz/07_rainyseason/code/CMIP5_cleaned/future_rcp45/*.nc')

cmip_names, data = [],[]
for file in cmip_85[:]:
    # get model names
    nn = file.split('\\')[-1].split('_')[1]
    print(f'doing {nn}',end='\r')
    cmip_names.append(nn +'_85')
    
    # open cmip5 file
    da = xr.open_dataarray(file)
    data.append(da.rename({nn}))

for file in cmip_45[:]:
    # get model names
    nn = file.split('\\')[-1].split('_')[1]
    print(f'doing {nn}',end='\r')
    cmip_names.append(nn + '_45')
    
    # open cmip5 file
    da = xr.open_dataarray(file)
    data.append(da.rename({nn}))
    

# merge and clean
cmip_ds = xr.concat(data,dim='name')
cmip_ds.name = 'pr'
cmip_ds = cmip_ds.rename({'Time':'time'})

cmip_ds.coords["name"] = cmip_names




# load and preprocess ETCCDI indices
files85 = glob.glob(r'S:\EDDY\Lorenz\07_rainyseason\annual_september_start/*_future_*_annual*.nc')
files45 = glob.glob(r'S:\EDDY\Lorenz\07_rainyseason\ETCCDI_for_Lorenz_rcp45/*_future_*_annual*.nc')


_ds8,_ds4 = [],[]
names8,names4 = [],[]
for f8,f4 in zip(files85,files45):
    n8,n4 = os.path.basename(f8).split('_')[-4] , os.path.basename(f4).split('_')[-4]
    names8.append(n8)
    names4.append(n4)
    print(f' doing {n8,n4}')

    _8,_4 = xr.open_dataset(f8),xr.open_dataset(f4)
    if n8.split('_')[0] != n4.split('_')[0]:
        print(n8,n4,'error')
        break
    _8 = penalty_free_rename_dim(_8,new_dim=n8)
    _4 = penalty_free_rename_dim(_4,new_dim=n4)
    _8 = penalty_free_rename_dim(_8,old_dim='time2',new_dim='time')
    _4 = penalty_free_rename_dim(_4,old_dim='time2',new_dim='time')
    
    _ds8.append(_8.isel(time=slice(1,-1)))
    _ds4.append(_4.isel(time=slice(1,-1)))
    
    

ds8 = xr.merge(_ds8)
ds8 = ds8.assign_coords(name=[f"{name}_85" for name in ds8.coords['name'].values])
ds8 = ds8.drop('quantile')


ds4 = xr.merge(_ds4)
ds4 = ds4.assign_coords(name=[f"{name}_45" for name in ds4.coords['name'].values])
ds4 = ds4.drop('quantile')
ds4 = ds4.drop_sel(name='GFDL-CM3_45') 

print('shapes: ',ds8[list(ds8.data_vars)[0]].shape,ds4[list(ds4.data_vars)[0]].shape)
ds = xr.merge([ds8,ds4])

## Calculate sea/anu precipitation sums

# reindex the cmip data to match the indices data
cmip_ds = cmip_ds.reindex_like(ds['name'])

# for easy calc of annual sum we shift the timeseries 8 month back and forth
ds_shifted = cmip_ds.copy()
ds_shifted['time'] = pd.DatetimeIndex(ds_shifted['time'].values) - pd.DateOffset(months=8)
annual_sum = ds_shifted.resample(time='YS').sum()
annual_sum['time'] = pd.DatetimeIndex(annual_sum['time'].values) + pd.DateOffset(months=8)

# get seasonal sums
cmip_ds['season_year'] = ('time', [get_season_label(t) for t in cmip_ds['time'].to_index()]) 
seasonal_sum = cmip_ds.groupby('season_year').sum(dim='time')

# get seasonal string vector
seasonal_sum['season'] = seasonal_sum['season_year']*0
seasonal_sum['season'].values = [s.split('-')[1] for s in seasonal_sum['season_year'].values.astype(str)]

#add to main ds (indices)
ds['pr_ANU'] = annual_sum
ds['pr_SON'] = (cmip_ds.dims,seasonal_sum.where(seasonal_sum['season'] == 'SON', drop=True).values)
ds['pr_DJF'] = (cmip_ds.dims,seasonal_sum.where(seasonal_sum['season'] == 'DJF', drop=True).values)
ds['pr_MAM'] = (cmip_ds.dims,seasonal_sum.where(seasonal_sum['season'] == 'MAM', drop=True).values)
ds['pr_JJA'] = (cmip_ds.dims,seasonal_sum.where(seasonal_sum['season'] == 'JJA', drop=True).values)


## Calculate RSO metrics
Gu,Cl,Ga,FP,JD,Bu,LM,CB = ([] for _ in range(8))

for modelname in cmip_ds['name']:
    Gu.append(gurgiser_RSO( params=pS.iloc[:,0].dropna(how='all'),ts=cmip_ds.sel(name=modelname).values,return_onset=True))
    Cl.append(climandes_RSO(params=pS.iloc[:,2].dropna(how='all'),ts=cmip_ds.sel(name=modelname).values,return_onset=True))
    Ga.append(garcia_RSO(   params=pS.iloc[:,4].dropna(how='all'),ts=cmip_ds.sel(name=modelname).values,return_onset=True))
    FP.append(FP_RSO(       params=pS.iloc[:,6].dropna(how='all'),ts=cmip_ds.sel(name=modelname).values,return_onset=True))
    JD.append(JD_RSO(       params=pS.iloc[:,8].dropna(how='all'),ts=cmip_ds.sel(name=modelname).values,return_onset=True))
    Bu.append(bucket(       params=pS.iloc[:,10][:7],ts=cmip_ds.sel(name=modelname).values,optim=None)[0])
    LM.append(Liebmann(ts=cmip_ds.sel(name=modelname).values)[0])
    CB.append(CookBuckley_2phase(ts=cmip_ds.sel(name=modelname))[0])
    print(f'RSO of {modelname.values} done' + 100*'  ',end='\r')
    
print('  RSO done, writing data to array')
cols = ['S_Gurgiser','S_Climandes','S_Garcia','S_FP','S_JD','S_Bucket','S_Liebmann','S_CookBuckley']
ons_data = [Gu,Cl,Ga,FP,JD,Bu,LM,CB]
    
for col, vals in zip(cols,ons_data):
    try:
        ds[col] = ds['DD']*0
        ds[col].values = np.array(vals)
    except:
        print(f'{col} not written')

## Calculate RSE metrics
Gu,Cl,Ga,FP,JD,Bu,LM,CB = ([] for _ in range(8))
for modelname in cmip_ds['name']:
    
    Gu.append(gurgiser_RSE( params=pE.iloc[:,0].dropna(how='all'),ts=cmip_ds.sel(name=modelname).values,
                           onsets=ds['S_Gurgiser'].sel(name=modelname).values, return_end=True))
    Cl.append(climandes_RSE(params=pE.iloc[:,2].dropna(how='all'),ts=cmip_ds.sel(name=modelname).values,
                            onsets=ds['S_Climandes'].sel(name=modelname).values ,return_end=True))            
    Ga.append(garcia_RSE(   params=pE.iloc[:,4].dropna(how='all'),ts=cmip_ds.sel(name=modelname).values,
                            onsets=ds['S_Garcia'].sel(name=modelname).values,return_end=True))
    Bu.append(bucket(       params=pE.iloc[:,6][:7],ts=cmip_ds.sel(name=modelname).values,optim=None)[1])   
    LM.append(Liebmann(ts=cmip_ds.sel(name=modelname).values)[1])   
    CB.append(CookBuckley_2phase(ts=cmip_ds.sel(name=modelname))[1])
    print(f'RSE of {modelname.values} done' + 100*'  ',end='\r')
 

print('  RSE done, writing data to array')
cols = ['E_Gurgiser','E_Climandes','E_Garcia','E_Bucket','E_Liebmann','E_CookBuckley']
end_data = [Gu,Cl,Ga,Bu,LM,CB]

for col, vals in zip(cols,end_data):
    try:
        ds[col] = ds['DD']*0
        ds[col].values = np.array(vals)
    except:
        print(f'{col} not written')


 ## float casting / no data handling
for var in ds.data_vars:
    if np.issubdtype(ds[var].dtype, np.integer):
        ds[var] = ds[var].where(ds[var] != -9999, np.nan)      


ds.to_netcdf(r'./outputs/Fig4_data.nc')
print('all done, data saved')