import pandas as pd
import numpy as np
import time
import scipy.optimize as optimize

from RainySeason import *

# parameter table to fill
RSE_params = pd.read_csv(r'./params/RSE_params_b4Optim.csv',index_col=0)

# target LSP data
lsp = pd.read_csv(r'./inputs/LSP_targets.csv',index_col=0,parse_dates=True)

EOS_w = lsp['EOS_no_lag'] - lsp['lag_WRF']
EOS_c = lsp['EOS_no_lag'] - lsp['lag_CHIRPS']
EOS_a = lsp['EOS_no_lag'] - lsp['lag_AWS']
EOS_b = lsp['EOS_no_lag'] # for bucket model only

# calibrated RSO params
RSO_params = pd.read_csv(r'./params/RSO_params_afterOptim.csv',index_col=0)

# load precip timeseries 
data = pd.read_csv('./inputs/Calibration_precip_ts.csv',index_col=0,parse_dates=True,dayfirst=True)
ts_w = data['WRF']
ts_c = data['CHIRPS']
ts_a = data['AWS']



#### Threshold RSE calibration ####

# creating onsets on the fly
param_idx = ['Gu','Cl','Ga']
timeseries = [ts_w,ts_c,ts_a]
models = [gurgiser_RSO,climandes_RSO,garcia_RSO]

# get onsets
onsets = []
for i,model in enumerate(models):
    intm = []
    for j,ts in enumerate(timeseries):
        params = list(RSO_params.loc[:, RSO_params.columns.str.startswith(param_idx[i])].dropna(how='all').iloc[:,j+1])
        intm.append(list(model(params=params,ts=ts,return_onset=True)))
    onsets.append(intm)
    
# redefining lists for EOS        
timeseries = [ts_w,ts_c,ts_a]
validation = [EOS_w,EOS_c,EOS_a]
models = [gurgiser_RSE,climandes_RSE,garcia_RSE]
boundaries = [((0., 1), (2, 15),(15,75)), # Gurgiser [0,10,45]
              ((0., 1), (2, 15),(15,75)), # Climandes [1,16,29]
              ((0, 5), (5,45))]          # Garcia [0,19]

sel = 0 # adjust to skip certain models
c=0
for model,bnds,ons_per_model in zip(models[sel:],boundaries[sel:],onsets[sel:]): #iterate each rainy season metric
    print(f'doing {str(model).split()[1]} \n\n')
    c+=1
    for ts,val,ons in zip(timeseries,validation,ons_per_model): #iterate for each timeseries and corresponding validation
        print(f'...doing {ts.name}')
        s = time.time()
        # optimize and print progress
        res = optimize.differential_evolution(model,bounds=bnds,seed=42,args=(ts,ons,val,True,11))    
        print(300*'   ',end='\r')
        print(f'...execution time: {np.round((time.time() - s)/3600,2)} hours \n')
        print(res)
        RSE_params.iloc[:len(res.x),sel*4+c] = res.x

        for x in res.x: print(x)
        print('\n')
        c+=1

    print('\n\n')

# RSE Bucket
bnds = ((.08,.16),(.3,.7),(.02,.08),(9999,9999),(.2,.7),(.3,.7),(1.1,1.9))
timeseries = [ts_w,ts_c,ts_a]

c=0
for ts in timeseries: #iterate for each timeseries
    print(f'...doing {ts.name}')
    s = time.time()
    # optimize and print progress
    res = optimize.differential_evolution(bucket,bounds=bnds,seed=42,args=(ts,EOS_b,True,15,'RSE'))   
    print(200*'   ',end='\r')
    print(f'...execution time: {np.round((time.time() - s)/3600,2)} hours \n')
    print(res)
    for x in res.x: print(x)
    RSE_params.iloc[:,-3+c] = res.x
    print('\n')
    c+=1

RSE_params.to_csv(r'./params/RSE_params_afterOptim.csv')



#### Figures ####





