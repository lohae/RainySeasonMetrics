import pandas as pd
import numpy as np
import xarray as xr
import scipy
import glob

from RainySeason import *
from utils import *
### load data

## load and prepare data

# RS metric parameters
pS = pd.read_csv(r'./params/RSO_params_afterOptim.csv',index_col=0)
pE = pd.read_csv(r'./params/RSE_params_afterOptim.csv',index_col=0)

# remove init and AWS
pS = pS.loc[:,~pS.columns.str.contains('init')]
pS = pS.loc[:,~pS.columns.str.contains('AWS')]
pE = pE.loc[:,~pE.columns.str.contains('init')]
pE = pE.loc[:,~pE.columns.str.contains('AWS')]

# validation LSP data
lsp = pd.read_csv(r'./inputs/LSP_targets.csv',index_col=0,parse_dates=True)
SOS_w = lsp['SOS_no_lag'] - lsp['lag_WRF']
SOS_c = lsp['SOS_no_lag'] - lsp['lag_CHIRPS']
SOS_a = lsp['SOS_no_lag'] - lsp['lag_AWS']
SOS_b = lsp['SOS_no_lag'] 

EOS_w = lsp['EOS_no_lag'] - lsp['lag_WRF']
EOS_c = lsp['EOS_no_lag'] - lsp['lag_CHIRPS']
EOS_a = lsp['EOS_no_lag'] - lsp['lag_AWS']
EOS_b = lsp['EOS_no_lag'] 

# Past
wrf_past = prepare_ts(xr.open_dataarray('./inputs/WRF_preprocessed_1981_2018.nc').to_pandas(),year_s=1981,year_e=2018)
chi_past = prepare_ts(xr.open_dataarray('./inputs/CHIRPS_preprocessed_1981_2018.nc').to_pandas(),year_s=1981,year_e=2018)

# Future
cmip_85 = glob.glob(r'./inputs/future_rcp85/*.nc')
cmip_45 = glob.glob(r'./inputs/future_rcp45/*.nc')

# x_axes extents
x_p = np.arange(1981,2018)
x_c = np.arange(2000,2018)
x_f = np.arange(2019,2100)


### compute

# past metrics RSO
print('doing RSO past')
smodels_w = [gurgiser_RSO( params=pS.iloc[:,0].dropna(how='all').values,ts=wrf_past.values,return_onset=True),
            climandes_RSO(params=pS.iloc[:,2].dropna(how='all').values,ts=wrf_past.values,return_onset=True),
            garcia_RSO(   params=pS.iloc[:,4].dropna(how='all').values,ts=wrf_past.values,return_onset=True),
            FP_RSO(       params=pS.iloc[:,6].dropna(how='all').values,ts=wrf_past.values,return_onset=True),
            JD_RSO(       params=pS.iloc[:,8].dropna(how='all').values,ts=wrf_past.values,return_onset=True),
            bucket(       params=pS.iloc[:,10][:7],ts=wrf_past.values)[0],
            Liebmann(ts=wrf_past.values)[0],
            CookBuckley_2phase(ts=wrf_past.values)[0]]
print('...WRF done')
smodels_c = [gurgiser_RSO( params=pS.iloc[:,1].dropna(how='all').values,ts=chi_past.values,return_onset=True),
            climandes_RSO(params=pS.iloc[:,3].dropna(how='all').values,ts=chi_past.values,return_onset=True),
            garcia_RSO(   params=pS.iloc[:,5].dropna(how='all').values,ts=chi_past.values,return_onset=True),
            FP_RSO(       params=pS.iloc[:,7].dropna(how='all').values,ts=chi_past.values,return_onset=True),
            JD_RSO(       params=pS.iloc[:,9].dropna(how='all').values,ts=chi_past.values,return_onset=True),
            bucket(       params=pS.iloc[:,11][:7],ts=chi_past.values)[0],
            Liebmann(ts=chi_past.values)[0],
            CookBuckley_2phase(ts=chi_past.values)[0]]
print('...CHIRPS done')

sdf_w = pd.DataFrame(np.array(smodels_w).T,columns= ['Gurgiser','Climandes','Garcia','FP','JD','Bucket','Liebmann','CB'])
sdf_c = pd.DataFrame(np.array(smodels_c).T,columns= ['Gurgiser','Climandes','Garcia','FP','JD','Bucket','Liebmann','CB'])

sdf_w.index = np.arange(1981,2018)
sdf_c.index = np.arange(1981,2018)

# save results
sdf_w.to_csv('./outputs/RSO_WRF_past.csv')
sdf_c.to_csv('./outputs/RSO_CHI_past.csv')



# process RCP 8.5 data
print('doing RSO RCP 8.5')
futuremodels = [[] for _ in np.arange(len(smodels_w))]
cmip_names = []
for file in cmip_85[:]:
    # get model names
    nn = file.split('\\')[-1].split('_')[1]
    print(f'...doing {nn}',end='\r')
    cmip_names.append(nn)
    
    # open cmip5 file
    ds = xr.open_dataset(file)
    
    # run rainy season metrics
    futuremodels[0].append(gurgiser_RSO( params=pS.iloc[:,0].dropna(how='all'),ts=ds['pr'].values,return_onset=True))
    futuremodels[1].append(climandes_RSO(params=pS.iloc[:,2].dropna(how='all'),ts=ds['pr'].values,return_onset=True))
    futuremodels[2].append(garcia_RSO(   params=pS.iloc[:,4].dropna(how='all'),ts=ds['pr'].values,return_onset=True))
    futuremodels[3].append(FP_RSO(       params=pS.iloc[:,6].dropna(how='all'),ts=ds['pr'].values,return_onset=True))
    futuremodels[4].append(JD_RSO(       params=pS.iloc[:,8].dropna(how='all'),ts=ds['pr'].values,return_onset=True))
    futuremodels[5].append(bucket(       params=pS.iloc[:,10][:7],ts=ds['pr'].values,optim=None)[0])
    futuremodels[6].append(Liebmann(ts=ds['pr'].values)[0])
    futuremodels[7].append(CookBuckley_2phase(ts=ds['pr'])[0])
    print('    '*10,end='\r')


futumodelnames = ['Gurgiser','Climandes','Garcia','FP','JD','Bucket','Liebmann','CookBuckley']
final_ens_85,final_std_85 = [],[]
for j,onsetmodel in enumerate(futuremodels): # get predictions from RSO metrics
    print(f'...processing {futumodelnames[j]}')
    ensemble_cmip = []
    valid_models = 0
    ptrend = 0
    ntrend = 0
    for i,cmipmodel in enumerate(onsetmodel): # get each set of rainy season onset from each CMIP model
        mask = np.where(cmipmodel <= 0)[0] 
        y = cmipmodel.astype(float)
        if len(mask) <= 5: # max 5 missing vaals
            y[mask] = np.nan
            ensemble_cmip.append(y)
            
            #get trends
            tmask = np.where(y != np.nan)[0] 
            m, _, _, p, _ = scipy.stats.linregress(cmipmodel[tmask], np.arange(2019,2100)[tmask])
            valid_models+=1
            if p < 0.1:
                if m > 0.1:
                    ptrend+=1
                elif m < 0.1:
                    ntrend+=1
        else:
            print(f'{cmip_names[i]} has {len(mask)}/{len(y)} missing values, removed')
            
    print(f'{ptrend} of {valid_models} ({np.round(ptrend/valid_models*100,1)}%) have a positive trend')
    print(f'{ntrend} of {valid_models} ({np.round(ntrend/valid_models*100,1)}%) have a negative trend')
            
    final_ens_85.append(np.round(np.nanmean(np.array(ensemble_cmip),axis=0)))
    final_std_85.append(np.round(np.nanstd(np.array(ensemble_cmip),axis=0)))  


 # process RCP 4.5 data
print('doing RSO RCP 4.5')
futuremodels = [[] for _ in np.arange(len(smodels_w))]
cmip_names = []
for file in cmip_45[:]:
    # get model names
    nn = file.split('\\')[-1].split('_')[1]
    print(f'...doing {nn}',end='\r')
    cmip_names.append(nn)
    
    # open cmip5 file
    ds = xr.open_dataset(file)
    
    # run rainy season metrics
    futuremodels[0].append(gurgiser_RSO( params=pS.iloc[:,0].dropna(how='all'),ts=ds['pr'].values,return_onset=True))
    futuremodels[1].append(climandes_RSO(params=pS.iloc[:,2].dropna(how='all'),ts=ds['pr'].values,return_onset=True))
    futuremodels[2].append(garcia_RSO(   params=pS.iloc[:,4].dropna(how='all'),ts=ds['pr'].values,return_onset=True))
    futuremodels[3].append(FP_RSO(       params=pS.iloc[:,6].dropna(how='all'),ts=ds['pr'].values,return_onset=True))
    futuremodels[4].append(JD_RSO(       params=pS.iloc[:,8].dropna(how='all'),ts=ds['pr'].values,return_onset=True))
    futuremodels[5].append(bucket(       params=pS.iloc[:,10][:7],ts=ds['pr'].values,optim=None)[0])
    futuremodels[6].append(Liebmann(ts=ds['pr'].values)[0])
    futuremodels[7].append(CookBuckley_2phase(ts=ds['pr'])[0])
    print('    '*10,end='\r')

    
# RCP4.5 trends
futumodelnames = ['Gurgiser','Climandes','Garcia','FP','JD','Bucket','Liebmann','CookBuckley']
final_ens_45,final_std_45 = [],[]
for j,onsetmodel in enumerate(futuremodels): # get each prediction from onset models
    print(f'processing {futumodelnames[j]}')
    ensemble_cmip = []
    valid_models = 0
    ptrend = 0
    ntrend = 0
    for i,cmipmodel in enumerate(onsetmodel): # get each set of rainy season onset from each CMIP model
        mask = np.where(cmipmodel <= 0)[0] # get missing values
        if len(mask) <= 5: # max 5 missing vals
            y = cmipmodel.astype(float)
            y[mask] = np.nan
            ensemble_cmip.append(y)
            
            #get trends
            tmask = np.where(y != np.nan)[0] 
            m, _, _, p, _ = scipy.stats.linregress(cmipmodel[tmask], np.arange(2019,2100)[tmask])
            valid_models+=1
            
            if p < 0.1:
                #print(cmip_names[i],np.round(m*10,2),np.round(p,3))
                if m > 0.1:
                    ptrend+=1
                elif m < 0.1:
                    ntrend+=1
        else:
            print(f'{cmip_names[i]} has {len(mask)}/{len(y)} missing values, removed')
            
    print(f'{ptrend} of {valid_models} ({np.round(ptrend/valid_models*100,1)}%) have a positive trend')
    print(f'{ntrend} of {valid_models} ({np.round(ntrend/valid_models*100,1)}%) have a negative trend')
            
    final_ens_45.append(np.round(np.nanmean(np.array(ensemble_cmip),axis=0)))
    final_std_45.append(np.round(np.nanstd(np.array(ensemble_cmip),axis=0)))  



# save results
data45 = np.array((np.array(final_ens_45),np.array(final_std_45))).reshape(16,81).T
data85 = np.array((np.array(final_ens_85),np.array(final_std_85))).reshape(16,81).T

all_data = np.array((data45,data85)).reshape(81,32).astype(int)
cols = [n + '_avg_45' for n in futumodelnames] + [n + '_std_45' for n in futumodelnames] + [n + '_avg_85' for n in futumodelnames] + [n + '_std_85' for n in futumodelnames]

sdf_f = pd.DataFrame(data=all_data,columns=cols,index=x_f)
sdf_f.to_csv('./outputs/RSO_metrics_CMIP_future.csv')

print('RSO done')

# past metrics RSE
print('doing RSE past')
onsets_w = sdf_w.drop(columns=['FP','JD'])
onsets_c = sdf_c.drop(columns=['FP','JD'])

emodels_w = [gurgiser_RSE( params=pE.iloc[:,0].dropna(how='all').values,onsets=onsets_w.iloc[:,0].values,
                          ts=wrf_past.values,return_end=True),
            climandes_RSE(params=pE.iloc[:,2].dropna(how='all').values,onsets=onsets_w.iloc[:,1].values,
                          ts=wrf_past.values,return_end=True),
            garcia_RSE(   params=pE.iloc[:,4].dropna(how='all').values,onsets=onsets_w.iloc[:,2].values,
                          ts=wrf_past.values,return_end=True),
            bucket(params=pE.iloc[:,6][:7],ts=wrf_past.values)[1],
            Liebmann(ts=wrf_past.values)[1],
            CookBuckley_2phase(ts=wrf_past.values)[1]]
print('...WRF done')

emodels_c = [gurgiser_RSE( params=pE.iloc[:,1].dropna(how='all').values,onsets=onsets_w.iloc[:,0].values,
                          ts=chi_past.values,return_end=True),
            climandes_RSE(params=pE.iloc[:,3].dropna(how='all').values,onsets=onsets_w.iloc[:,1].values,
                          ts=chi_past.values,return_end=True),
            garcia_RSE(   params=pE.iloc[:,5].dropna(how='all').values,onsets=onsets_w.iloc[:,2].values,
                          ts=chi_past.values,return_end=True),
            bucket(params=pE.iloc[:,7][:7],ts=chi_past.values)[1],
            Liebmann(ts=chi_past.values)[1],
            CookBuckley_2phase(ts=chi_past.values)[1]]
print('...CHIRPS done')

edf_w = pd.DataFrame(np.array(emodels_w).T,columns= ['Gurgiser','Climandes','Garcia','Bucket','Liebmann','CB'])
edf_c = pd.DataFrame(np.array(emodels_c).T,columns= ['Gurgiser','Climandes','Garcia','Bucket','Liebmann','CB'])

edf_w.index = np.arange(1981,2018)
edf_c.index = np.arange(1981,2018)

#intermediate save
edf_w.to_csv('./outputs/RSE_WRF_past.csv')
edf_c.to_csv('./outputs/RSE_CHI_past.csv')


# process RCP 8.5 data
print('doing RSE RCP 8.5')
futuremodels = [[] for _ in np.arange(len(emodels_w))]
cmip_names = []
for file in cmip_85[:]:
    nn = file.split('\\')[-1].split('_')[1]
    print(f'...doing {nn}',end='\r')
    cmip_names.append(nn)
    
    ds = xr.open_dataset(file)
    ons_gurg =  gurgiser_RSO(params=pS.iloc[:,0].dropna(how='all'),ts=ds['pr'].values,return_onset=True)
    ons_clim = climandes_RSO(params=pS.iloc[:,2].dropna(how='all'),ts=ds['pr'].values,return_onset=True)
    ons_garc =    garcia_RSO(params=pS.iloc[:,4].dropna(how='all'),ts=ds['pr'].values,return_onset=True)
    
    #ons_garc[ons_garc == -9999] = 0 #this is a temporary fix...
    
    futuremodels[0].append(gurgiser_RSE( params=pE.iloc[:,0].dropna(how='all'),ts=ds['pr'].values,
                                        onsets=ons_gurg, return_end=True))
    futuremodels[1].append(climandes_RSE(params=pE.iloc[:,2].dropna(how='all'),ts=ds['pr'].values,
                                        onsets=ons_clim ,return_end=True))
    futuremodels[2].append(garcia_RSE(   params=pE.iloc[:,4].dropna(how='all'),ts=ds['pr'].values,
                                        onsets=ons_garc ,return_end=True))
    futuremodels[3].append(bucket(       params=pE.iloc[:,6][:7],ts=ds['pr'].values,optim=None)[1])
    futuremodels[4].append(Liebmann(ts=ds['pr'].values)[1])
    futuremodels[5].append(CookBuckley_2phase(ts=ds['pr'].values)[1])
    print('    '*10,end='\r')

futumodelnames = ['Gurgiser','Climandes','Garcia','Bucket','Liebmann','Cook&Buckley']
final_ens_85,final_std_85 = [],[]
for j,endmodel in enumerate(futuremodels): # get each prediction from onset models
    print(f'processing {futumodelnames[j]}')
    ensemble_cmip = []
    valid_models = 0
    ptrend = 0
    ntrend = 0
    
    for i,cmipmodel in enumerate(endmodel): # get each set of rainy season onset from each CMIP model
        mask = np.where(cmipmodel <= 150)[0] 
        if len(mask) <= 5: # max 5 missing vals
            y = cmipmodel.astype(float)
            y[mask] = np.nan
            ensemble_cmip.append(y)
            
            #get trends
            tmask = np.where(y != np.nan)[0] 
            m, _, _, p, _ = scipy.stats.linregress(cmipmodel[tmask], np.arange(2019,2100)[tmask])
            valid_models+=1
            if p < 0.1:
                if m > 0.1:
                    ptrend+=1
                elif m < 0.1:
                    ntrend+=1
        else:
            print(f'{cmip_names[i]} has {len(mask)}/{len(y)} missing values, removed')
            
    print(f'{ptrend} of {valid_models} ({np.round(ptrend/valid_models*100,1)}%) have a positive trend')
    print(f'{ntrend} of {valid_models} ({np.round(ntrend/valid_models*100,1)}%) have a negative trend')
            
    final_ens_85.append(np.round(np.nanmean(np.array(ensemble_cmip),axis=0)))
    final_std_85.append(np.round(np.nanstd(np.array(ensemble_cmip),axis=0)))


 # process RCP 4.5 data
print('doing RSE RCP 4.5')
futuremodels = [[] for _ in np.arange(len(emodels_w))]
cmip_names = []
for file in cmip_45[:]:
    nn = file.split('\\')[-1].split('_')[1]
    print(f'...doing {nn}',end='\r')
    cmip_names.append(nn)
    
    ds = xr.open_dataset(file)
    ons_gurg =  gurgiser_RSO(params=pS.iloc[:,0].dropna(how='all'),ts=ds['pr'].values,return_onset=True)
    ons_clim = climandes_RSO(params=pS.iloc[:,2].dropna(how='all'),ts=ds['pr'].values,return_onset=True)
    ons_garc =    garcia_RSO(params=pS.iloc[:,4].dropna(how='all'),ts=ds['pr'].values,return_onset=True)
    
    #ons_garc[ons_garc == -9999] = 0 #this is a temporary fix...
    
    futuremodels[0].append(gurgiser_RSE( params=pE.iloc[:,0].dropna(how='all'),ts=ds['pr'].values,
                                        onsets=ons_gurg, return_end=True))
    futuremodels[1].append(climandes_RSE(params=pE.iloc[:,2].dropna(how='all'),ts=ds['pr'].values,
                                        onsets=ons_clim ,return_end=True))
    futuremodels[2].append(garcia_RSE(   params=pE.iloc[:,4].dropna(how='all'),ts=ds['pr'].values,
                                        onsets=ons_garc ,return_end=True))
    futuremodels[3].append(bucket(       params=pE.iloc[:,6][:7],ts=ds['pr'].values,optim=None)[1])
    futuremodels[4].append(Liebmann(ts=ds['pr'].values)[1])
    futuremodels[5].append(CookBuckley_2phase(ts=ds['pr'].values)[1])
    print('    '*10,end='\r')

futumodelnames = ['Gurgiser','Climandes','Garcia','Bucket','Liebmann','Cook&Buckley']
final_ens_45,final_std_45 = [],[]
for j,endmodel in enumerate(futuremodels): # get each prediction from onset models
    print(f'processing {futumodelnames[j]}')
    ensemble_cmip = []
    valid_models = 0
    ptrend = 0
    ntrend = 0
    
    for i,cmipmodel in enumerate(endmodel): # get each set of rainy season onset from each CMIP model
        # remove no_data or eventually more (<= X)  
        mask = np.where(cmipmodel <= 150)[0] 
        if len(mask) <= 5: # max 5 missing vals
            y = cmipmodel.astype(float)
            y[mask] = np.nan
            ensemble_cmip.append(y)
            
            #get trends
            tmask = np.where(y != np.nan)[0] 
            m, _, _, p, _ = scipy.stats.linregress(cmipmodel[tmask], np.arange(2019,2100)[tmask])
            valid_models+=1
            if p < 0.1:
                #print(cmip_names[i],np.round(m*10,2),np.round(p,3))
                if m > 0.1:
                    ptrend+=1
                elif m < 0.1:
                    ntrend+=1
        else:
            print(f'{cmip_names[i]} has {len(mask)}/{len(y)} missing values, removed')
            
    print(f'{ptrend} of {valid_models} ({np.round(ptrend/valid_models*100,1)}%) have a positive trend')
    print(f'{ntrend} of {valid_models} ({np.round(ntrend/valid_models*100,1)}%) have a negative trend')
            
    final_ens_45.append(np.round(np.nanmean(np.array(ensemble_cmip),axis=0)))
    final_std_45.append(np.round(np.nanstd(np.array(ensemble_cmip),axis=0)))   

print('RSE done')

 # save result
data45 = np.array((np.array(final_ens_45),np.array(final_std_45))).reshape(12,81).T
data85 = np.array((np.array(final_ens_85),np.array(final_std_85))).reshape(12,81).T

all_data = np.array((data45,data85)).reshape(81,24).astype(int)
cols = [n + '_avg_45' for n in futumodelnames] + [n + '_std_45' for n in futumodelnames] + [n + '_avg_85' for n in futumodelnames] + [n + '_std_85' for n in futumodelnames]

edf_f = pd.DataFrame(data=all_data,columns=cols,index=x_f)
edf_f.to_csv(r'./outputs/RSE_metrics_CMIP_future.csv')

print('all done, outputs saved')
