import pandas as pd
import numpy as np
import time
import scipy.optimize as optimize

from RainySeason import *


# if calibration was done previously we will use the outputs
try:
    RSO_params = pd.read_csv(r'./params/RSO_params_afterOptim.csv',index_col=0)
except:
    # parameter tables to fill
    RSO_params = pd.read_csv(r'./params/RSO_params_b4Optim.csv',index_col=0)

    # target LSP data
    lsp = pd.read_csv(r'./inputs/LSP_targets.csv',index_col=0,parse_dates=True)
    SOS_w = lsp['SOS_no_lag'] - lsp['lag_WRF']
    SOS_c = lsp['SOS_no_lag'] - lsp['lag_CHIRPS']
    SOS_a = lsp['SOS_no_lag'] - lsp['lag_AWS']
    SOS_b = lsp['SOS_no_lag'] # for bucket model only




    # load precip timeseries 
    data = pd.read_csv('./inputs/Calibration_precip_ts.csv',index_col=0,parse_dates=True,dayfirst=True)
    ts_w = data['WRF']
    ts_c = data['CHIRPS']
    ts_a = data['AWS']


    #### Threshold RSO calibration ####

    timeseries = [ts_w,ts_c,ts_a]
    validation = [SOS_w,SOS_c,SOS_a]
    models = [gurgiser_RSO,climandes_RSO,garcia_RSO,FP_RSO,JD_RSO]

    boundaries = [(( 0, 2 ),(10, 40),( 5, 20),( 2, 20),(20, 60),(0,2.5)), # Gurgiser  [0,10,7,10,30,0]
                  (( 0, 2 ),(10, 40),( 5, 20),( 2, 20),(20, 60),(0,2.5)), # Climandes [1,8,5,7,30,0.1]
                  (( 8, 30),(2 , 7),( 8, 20),(20, 50),(0., 1.5)),         # Garcia    [20,3,10,30,0.1]
                  ((10, 50),(10, 60),(10, 35),(10,40)),                   # FP        [25,30,20,20]
                  (( 0, .6),( 3, 18),( 1,  12),(10,50),(10,30),(15, 45))  # JD        [.1,5,3,25,7,30]
                 ]


    # smaller boundaries for quicker testing
    #boundaries = [((0.2,0.8),(15,30),( 10, 15),(15,20),(45, 60),(0,1.5)), # Gurgiser  [0,10,7,10,30,0]
    #              (( 0, 2 ),(28, 40),( 11, 18),(10, 20),(39, 62),(0.5,2.5)), # Climandes [1,8,5,7,30,0.1]
    #              (( 14, 20),(2 , 6),( 8, 12),(15, 55),(0., 1.5)),         # Garcia    [20,3,10,30,0.1]
    #              ((26, 42),(35, 61),(12, 30),(15,40)),                   # FP        [25,30,20,20]
    #              (( 0, .5),( 5, 17),( 1,  10),(27,42),(10,23),(35, 47))  # JD        [.1,5,3,25,7,30]
    #             ]



    sel = 0 # adjust to skip certain models
    c = 0 # do NOT adjust
    for model,bnds in zip(models[sel:],boundaries[sel:]): #iterate rainy season metrics
        print(f'doing {str(model).split()[1]} \n\n')
        c+=1
        for ts,val in zip(timeseries,validation): # iterate precip timeseries and lag corrected targets
            print(f'...doing {ts.name}')
            s = time.time()
            res = optimize.differential_evolution(model,bounds=bnds,seed=42,args=(ts,val,True,11))
            print(300*'   ',end='\r')
            print(f'...execution time: {np.round((time.time() - s)/3600,2)} hours \n')
            print(res)
            RSO_params.iloc[:len(res.x),sel*4+c] = res.x
            print('\n')
            c+=1
        print('\n\n')      



    #### Bucket RSO calibration ####
    # RSO Bucket
    print('doing Bucket \n\n')
    bnds = ((.08,.16),(.3,.7),(.02,.08),(.1,.4),(-9999,-9999),(.3,.7),(1.1,1.9))
    timeseries = [ts_w,ts_c,ts_a]

    c= 0
    for ts in timeseries[:]: #iterate for each timeseries
        print(f'...doing {ts.name}')
        s = time.time()
        # optimize and print progress
        res = optimize.differential_evolution(bucket,bounds=bnds,seed=42,args=(ts,SOS_b,True,15,'RSO',))   
        print(300*'   ',end='\r')
        print(f'...execution time: {np.round((time.time() - s)/3600,2)} hours \n')
        print(res)
        for x in res.x: print(x)
        RSO_params.iloc[:,-3+c] = res.x
        print('\n')
        c+=1

    RSO_params.to_csv(r'./params/RSO_params_afterOptim.csv')


##### Fig02 #####












