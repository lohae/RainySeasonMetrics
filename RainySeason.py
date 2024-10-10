import numpy as np
import pandas as pd

def rsme(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# based on Gurgiser et al., (2016): Rio Santa
def gurgiser_RSO(params,ts,targets=None,print_progress=False,rsme_target=11,return_onset=False):
    """Calculates rainy season onset based on Gurgiser et al. (2016):

    Gurgiser, W., Juen, I., Singer, K., Neuburger, M., Schauwecker, S., Hofer, M., & Kaser, G. (2016).
    Comparing peasants' perceptions of precipitation change with precipitation records in the tropical
    Callejón de Huaylas, Peru. Earth System Dynamics, 7(2), 499-515.

    Function
    ----------
    calculates day since first day of ts where P(d) > alpha1p and
        sum(P(day:day+alpha3d)) > alpha2p and count(P(d:d+alpha5d) > alpha6p) > alpha4d

    Parameters
    ----------
    params : list with 6 elements, values in () as published by the authors
        alpha1p: threshold for rainy day (0mm)
        alpha2p: threshold for onset detection, the sum of n days precipitation (10mm)
        alpha3d: the n days of alpha2p (7 days)
        alpha4d: min wet days in period alpha5d (10 days)
        alpha5d: period to look at for alpha4d (30 days)
        alpha6p: threshold for dry day (0mm)

    ts : 1D numpy array
        daily precipitation timeseries with 29th Feb removed

    targets : list, optional
        only necessary when return_onset=False

    print_progress: bool, optional
        for optimizing purposes, prints rsme, parameters and result, default is False

    rsme_target: int, optional
        only if print_progress=True, highlights rsme results below defined threshold

    return_onset : bool, optional
        A flag used to either return rainy season onsets (True) or returns rsme between RSO and targets (default is
        False)

    Returns
    -------
    list, when return_onset=True
        a list of rainy season onsets for each season

    rsme, when return_onset=False
        rsme value between targets and rainy season

    """


    # list for saving onsets
    lonset = []

    # make vars from params
    alpha1p,alpha2p,alpha3d,alpha4d,alpha5d,alpha6p = params

    # iterate hydrological years
    for i in range(int(len(ts)/365)):
        j=0 # counter
        onset=None
        ts_sub = ts[0+i*365:365+i*365] #get hydrological year
        # count all potential dates for iterating
        all_pot_dates = len(np.where(ts_sub > alpha1p)[0])
        while onset == None:
            if j == all_pot_dates - 1: # no data value if conditions are never met
                onset = -9999
                lonset.append(onset)
            else:
                # criterion 1, iterate through days with more than alpha1p precip
                onset_pot = np.where(ts_sub > alpha1p)[0][j] #[0] as where returns tuple!
                # criterion 2, precip sum of potential day + n days
                condition2 = ts_sub[onset_pot:onset_pot + int(alpha3d)].sum()
                # criterion 3, count number of days within n days which are below dryday threshold
                condition3 = len(np.where(ts_sub[onset_pot:onset_pot + int(alpha5d)] < alpha6p)[0])

                # precip sum needs to larger than threshold and n dry days need to be less than threshold
                if condition2 >= alpha2p and condition3 < alpha4d:
                    onset = onset_pot
                    lonset.append(onset)
                else:
                    j+=1
    if return_onset == False:
        vrsme = rsme(np.array(lonset),targets)
        if print_progress == True:
            prstr = (f'rsme = {np.round(vrsme,1)}, params = {np.round(params,3)}')
            if vrsme < rsme_target:
                print(f"\033[1m{prstr}\033[0m",end='\r',flush=True)
            else:
                print(prstr,end='\r',flush=True)

        return vrsme

    else:
        return np.array(lonset)
def gurgiser_RSE(params,ts,onsets,targets=None,print_progress=False,rsme_target=11,return_end=False):
    """Calculates rainy season end based on Gurgiser et al. (2016):

    Gurgiser, W., Juen, I., Singer, K., Neuburger, M., Schauwecker, S., Hofer, M., & Kaser, G. (2016).
    Comparing peasants' perceptions of precipitation change with precipitation records in the tropical
    Callejón de Huaylas, Peru. Earth System Dynamics, 7(2), 499-515.

    Function
    ----------
    calculates day since first day of ts where P(d) <= alpha1p and sum(P(day:day+alpha3d)) < alpha2p

    Parameters
    ----------
    params : list with 3 elements, values in () as published by the authors
        alpha1p: threshold for rainy day (0mm)
        alpha2p: threshold for end detection, the sum of n days precipitation (10mm)
        alpha3d: the n days for calculating parameter alpha2d (46 days)

    ts : 1D numpy array
        daily precipitation timeseries with 29th Feb removed

    onsets : list / numpy array
        return of gurgiser_RSO, to check only for end of season after the respective onset,
        also speeds up function

    targets : list, optional
        only necessary when return_end=False

    print_progress: bool, optional
        for optimizing purposes, prints rsme, parameters and result, default is False

    rsme_target: int, optional
        only if print_progress=True, highlights rsme results below defined threshold

    return_end : bool, optional
        A flag used to either return rainy season end (True) or
        returns rsme between metric and validation (default is False)
    

    Returns
    -------
    list, when return_end=True
        a list of rainy season end for each season

    rsme, when return_end=False
        rsme value between targets and rainy season metric

    """
    lend = []
    alpha1p,alpha2p,alpha3d = params

    for i in range(int(len(ts)/365)):
        j=0
        end=None
        if onsets[i] < 0:
            end = -9999
            lend.append(end)
        elif onsets[i] > 250: #account for very late onsets
            onsets[i] = 0
        ts_sub = ts[onsets[i]+i*365:365+i*365] # look only after onset for efficiency
        all_pot_dates = len(np.where(ts_sub <= alpha1p)[0]) # get n of dry days

        while end == None and j < all_pot_dates:
            # condition1, dry days
            end_pot = np.where(ts_sub <= alpha1p)[0][j]
            # condition2, sum precip of n days
            condition2 = ts_sub[end_pot:end_pot + int(alpha3d)]
            if j == all_pot_dates - 1: # nan if conditions never met
                end = -9999
                lend.append(end)
            elif condition2.sum() < alpha2p and (end_pot + onsets[i]) - onsets[i] > 90:  # find end
                end = end_pot + onsets[i] 
                lend.append(end)
            else: # or continue searching
                j+=1
    if return_end == False:
        vrsme = rsme(np.array(lend),targets)
        if print_progress == True:
            prstr = (f'rsme = {vrsme}, params = {np.round(params,3)}')
            if vrsme < rsme_target:
                print(f"\033[1m{prstr}\033[0m",end='\r') #bold print
            else:
                print(prstr,end='\r')

        return vrsme
    else:
        return np.array(lend)

# based on Sedlmeier et al., (2023): Southern Peruvian Andes
def climandes_RSO(params,ts,targets=None,print_progress=False,rsme_target=11,return_onset=False):
    """Calculates rainy season onset based on Sedlmeier et al. (2023):

    Sedlmeier, K., Imfeld, N., Gubler, S., Spirig, C., Caiña, K. Q.,
    Escajadillo, Y., ... & Schwierz, C. (2023). The rainy season
    in the Southern Peruvian Andes: A climatological analysis based
    on the new Climandes index. International Journal of Climatology,
    43(6), 3005-3022.

        Function
        ----------
        calculates day since first day of ts where P(d) > alpha1p & sum(P(d:d+alpha3d) >= alpha2p &
        Nc(P(d:d+alpha5d) < alpha6p) < alpha4d

        Parameters
        ----------
        params : list with 6 elements, values in () as published by the authors
            alpha1p: threshold for rainy day definition (1mm)
            alpha2p: threshold of precipitation days in alpha3d window (8mm)
            alpha3d: n days for threshold (5 days)
            alpha4d: number of rainy days within alpha5d (7days)
            alpha5d: n days for threshold alpha4d (30 days)
            alpha6p: threshold for dry day definition (0.1mm)

        ts : 1D numpy array
            daily precipitation timeseries with 29th Feb removed

        targets : list, optional
            only necessary when return_onset=False

        print_progress: bool, optional
            for optimizing purposes, prints rsme, parameters and result, default is False

        rsme_target: int, optional
            only if print_progress=True, highlights rsme results below defined threshold

        return_onset : bool, optional
            A flag used to either return rainy season onsets (True) or returns rsme between RSOs
            and targets (default is False)

        Returns
        -------
        list, when return_onset=True
            a list of rainy season onsets for each season

        rsme, when return_onset=False
            rsme value between targets and rainy season

        """
    lonset = []
    alpha1p,alpha2p,alpha3d,alpha4d,alpha5d,alpha6p = params

    for i in range(int(len(ts)/365)):
        j=0
        onset=None
        ts_sub = ts[0+i*365:365+i*365]
        all_pot_dates = np.where(ts_sub > alpha1p)[0]
        while onset == None:
            if j == len(all_pot_dates) - 1:
                onset = -9999
                lonset.append(onset)
            else:
                #criterion 1
                onset_pot = all_pot_dates[j]
                #criterion 2
                condition2 = ts_sub[onset_pot:onset_pot + int(alpha3d)].sum()
                #criterion 3
                # 1. Get all dry days in period (dry = True)
                dry_or_wet = pd.Series(ts_sub[onset_pot:onset_pot + int(alpha5d)] < alpha6p)
                # 2. Get consecutive dry and wet days by reorganizing ts 
                change_points = dry_or_wet.ne(dry_or_wet.shift()).cumsum()
                consecutive_conditions = [group.tolist() for _, group in dry_or_wet.groupby(change_points)]
                # 3. Get only CDDs and count
                lenCDD = [len(x) for x in consecutive_conditions if x[0]] 
                # 4. Check if CDD is smaller than 
                condition3 = any(x >= alpha4d for x in lenCDD) #generator is faster than list


                if condition2 >= alpha2p and not condition3:
                    onset = onset_pot
                    lonset.append(onset)
                else:
                    j+=1
    if return_onset == False:
        vrsme = rsme(np.array(lonset),targets)
        if print_progress == True:
            prstr = (f'rsme = {np.round(vrsme,1)}, params = {np.round(params,3)}')
            if vrsme < rsme_target:
                print(f"\033[1m{prstr}\033[0m",end='\r')
            else:
                print(prstr,end='\r',flush=True)

        return vrsme

    else:
        return np.array(lonset)

def climandes_RSE(params,ts,onsets,targets=None,print_progress=False,rsme_target=11,return_end=False):
    """Calculates rainy season end based on Sedlmeier et al. (2023):

    Sedlmeier, K., Imfeld, N., Gubler, S., Spirig, C., Caiña, K. Q.,
    Escajadillo, Y., ... & Schwierz, C. (2023). The rainy season
    in the Southern Peruvian Andes: A climatological analysis based
    on the new Climandes index. International Journal of Climatology,
    43(6), 3005-3022.

        Function
        ----------
        calculates day since first day of ts where P(d) <= alpha1p & P(d:d+alpha3d) < alpha2p

        Parameters
        ----------
        params : list with 6 elements, values in () as published by the authors
            alpha1p: threshold for dry day definition (1mm)
            alpha2p: threshold of precipitation sum in alpha3d window (16mm)
            alpha3d: n days for threshold alpha2p (30 days)


        ts : numpy array
            daily precipitation timeseries with 29th Feb removed

        onsets : list / numpy array
            return of climandes_RSO, to check only for end of season after the respective onset,
            also speeds up function

        targets : list, optional
            only necessary when return_onset=False

        print_progress: bool, optional
            for optimizing purposes, prints rsme, parameters and result, default is False

        rsme_target: int, optional
            only if print_progress=True, highlights rsme results below defined threshold

        print_progress: bool, optional
            for optimizing purposes, prints rsme, parameters and result, default is False

        rsme_target: int, optional
            only if print_progress=True, highlights rsme results below defined threshold

        return_end : bool, optional
            A flag used to either return list ofrainy season ends (True)
            or returns rsme between RSOs and targets (default is False)

        Returns
        -------
        list, when return_end=True
            a list of rainy season onsets for each season

        rsme, when return_end=False
            rsme value between targets and rainy season
    """
    lend = []
    # make vars from params
    alpha1p,alpha2p,alpha3d = params

    for i in range(int(len(ts)/365)):
        j=0
        end=None
        if onsets[i] < 0:
            end = -9999
            lend.append(end)
        elif onsets[i] > 250: #account for very late onsets
            onsets[i] = 0
            
        ts_sub = ts[onsets[i]+i*365:365+i*365] #look only after onset
        all_pot_dates = len(np.where(ts_sub <= alpha1p)[0]) #days with ~0 precip

        while end == None and j < all_pot_dates:
            # condition1, pick dry days
            end_pot = np.where(ts_sub <= alpha1p)[0][j]
            # condition2, get the precip of d+n
            condition2 = ts_sub[end_pot:end_pot + int(alpha3d)]
            if j == all_pot_dates - 1: # conditions never met
                end = -9999
                lend.append(end)
            elif condition2.sum() < alpha2p and (end_pot + onsets[i]) - onsets[i] > 90: # find the end
                end = end_pot + onsets[i] 
                lend.append(end)
            else: # or continue searching
                j+=1
    if return_end == False:
        vrsme = rsme(np.array(lend),targets)
        if print_progress == True:
            prstr = (f'rsme = {vrsme}, params = {np.round(params,3)}')
            if vrsme < rsme_target:
                print(f"\033[1m{prstr}\033[0m",end='\r')
            else:
                print(prstr,end='\r')

        return vrsme
    else:
        return np.array(lend)

# based on Garcia et al., (2007): Bolivian Altiplano
def garcia_RSO(params,ts,targets=None,print_progress=False,rsme_target=11,return_onset=False):
    """Calculates rainy season end based on García et al. (2007):

        García, M., Raes, D., Jacobsen, S. E., & Michel, T. (2007).
        Agroclimatic constraints for rainfed agriculture in the Bolivian
        Altiplano. Journal of Arid Environments, 71(1), 109-121.

        Function
        ----------
        calculates day since first day of ts where sum(P(d:d+alpha2d)) > alpha1p &
        Nc(P(d:d+alpha4d) < alpha5p) < alpha3d

        Parameters
        ----------
        params : list with 6 elements, values in () as published by the authors
            alpha1p: precipitation threshold of sum of alpha2d days (20mm)
            alpha2d: n days window for alpha1p (3 days)
            alpha3d: consecutive dry day threshold within alpha4d (10 days)
            alpha4d: n days window for parameter alpha3d (30 days)
            alpha5p: dry day threshold (0.1mm)

        ts : numpy array
            daily precipitation timeseries with 29th Feb removed

        targets : list, optional
            only necessary when return_onset=False

        print_progress: bool, optional
            for optimizing purposes, prints rsme, parameters and result, default is False

        rsme_target: int, optional
            only if print_progress=True, highlights rsme results below defined threshold

        return_onset : bool, optional
            A flag used to either return list of rainy season onsets (True)
            or returns rsme between RSOs and targets (default is False)

        Returns
        -------
        list, when return_onset=True
            a list of rainy season onsets for each season

        rsme, when return_onset=False
            rsme value between targets and rainy season
    """
    lonset = []
    alpha1p, alpha2d, alpha3d, alpha4d, alpha5p = params
    for i in range(int(len(ts)/365)):
        onset_found = False
        j=0
        onset=None
        ts_sub = ts[0+i*365:365+i*365]
        while onset == None and j < 364:
            # criterion 1
            condition1 = ts_sub[j:j + int(alpha2d)].sum()
            if condition1 < alpha1p: # keep searching for condition 1 to be fulfilled
                pass
            else: # condition 2
                # 1. Get all dry days in period (dry = True)
                dry_or_wet = pd.Series(ts_sub[j:j + int(alpha4d)] < alpha5p)
                # 2. Get consecutive dry and wet days by reorganizing ts 
                change_points = dry_or_wet.ne(dry_or_wet.shift()).cumsum()
                consecutive_conditions = [group.tolist() for _, group in dry_or_wet.groupby(change_points)]
                # 3. Get only CDDs and count
                lenCDD = [len(x) for x in consecutive_conditions if x[0]] #avoid checking all
                # 4. Check if CDD is smaller than 
                condition2 = any(x >= alpha3d for x in lenCDD) #generator is faster than list
                #print(j,lenCDD)
                if not condition2:
                    onset = j
                    #print(ts_sub.index[0] + pd.DateOffset(days=onset),ts_sub.index[j])
                    lonset.append(onset)
                    onset_found = True
                    
            j+=1
            
        if not onset_found: 
            lonset.append(-9999)

    if return_onset == False:
        vrsme = rsme(np.array(lonset),targets)
        if print_progress == True:
            prstr = (f'rsme = {np.round(vrsme,1)}, params = {np.round(params,3)}')
            if vrsme < rsme_target:
                print(f"\033[1m{prstr}\033[0m",end='\r')
            else:
                print(prstr,end='\r',flush=True)

        return vrsme

    else:
        return np.array(lonset)
def garcia_RSE(params,ts,onsets,targets=None,print_progress=False,rsme_target=11,return_end=False):
    """Calculates rainy season end based on García et al. (2007):

    García, M., Raes, D., Jacobsen, S. E., & Michel, T. (2007).
    Agroclimatic constraints for rainfed agriculture in the Bolivian
    Altiplano. Journal of Arid Environments, 71(1), 109-121.

    Function
    ----------
    calculates day since first day of ts where P(d:d+alpha2d) <= alpha1p

    Parameters
    ----------
    params : list with 6 elements, values in () as published by the authors
        alpha1p: precipitation sum threshold (0 mm)
        alpha2d: n days window (20 days)

        ts : numpy array
            daily precipitation timeseries with 29th Feb removed

        onsets : list / numpy array
            return of Garcia_RSO, to check only for end of season after the respective onset,
            also speeds up function

        targets : list, optional
            only necessary when return_end=False

        print_progress: bool, optional
            for optimizing purposes, prints rsme, parameters and result, default is False

        rsme_target: int, optional
            only if print_progress=True, highlights rsme results below defined threshold

        print_progress: bool, optional
            for optimizing purposes, prints rsme, parameters and result, default is False

        rsme_target: int, optional
            only if print_progress=True, highlights rsme results below defined threshold

        return_end : bool, optional
            A flag used to either return list of rainy season ends (True)
            or returns rsme between RSEs and targets (default is False)

        Returns
        -------
        list, when return_end=True
            a list of rainy season onsets for each season

        rsme, when return_end=False
            rsme value between targets and rainy season
    """
    lend = []
    alpha1p, alpha2d = params

    for i in range(int(len(ts)/365)):
        end=None
        if onsets[i] == -9999:
            lend.append(-9999)
            continue
        ts_sub = ts[0+i*365+onsets[i]:365+i*365]
        roll_n_day = pd.Series(ts_sub).rolling(window=int(alpha2d)).sum().values
        all_ends = np.where(roll_n_day < alpha1p)[0] + onsets[i]
        if len(all_ends) > 0:
            try:
                for pot_end in all_ends:
                    if pot_end - onsets[i] < 90:
                        continue
                    else:
                        end = pot_end 
                        lend.append(end)
                        break
            except:
                end = -9999
                lend.append(end)
        else:
            end = -9999
            lend.append(end)       

    if return_end == False:
        vrsme = rsme(np.array(lend),targets)
        if print_progress == True:
            prstr = (f'rsme = {vrsme}, params = {np.round(params,3)}')
            if vrsme < rsme_target:
                print(f"\033[1m{prstr}\033[0m",end='\r') #bold print
            else:
                print(prstr,end='\r')

        return vrsme
    else:
        return np.array(lend)

# based on Frere & Popov (1986): Basis for SENAHMIs agrometeorological drought monitoring
def FP_RSO(params,ts,targets=None,print_progress=False,rsme_target=11,return_onset=False):
    """
    Calculates rainy season end based on Frere & Popov (1986):

    Frere, M. and Popov, G. F., 1986. Early agrometeorological
    crop yield assessment, in Plant Production and Protection Paper Nr. 73,
    edited by FAO, FAO, Rome, Italy.

        Function
        ----------
        calculates day since first day of ts where P(d:d+(1/3)*alpha2d) >= alpha1p and 
        P(d+(1/3)*alpha2d:d+(2/3)*alpha2d) >= alpha3p and P(d+(2/3)*alpha2d:d+alpha2d) > alpha4p

        Parameters
        ----------
        params : list with 6 elements, values in () as published by the authors
            alpha1p: precipitation threshold for first tercile of window (25mm)
            alpha2d: n days window for precipitation thresholds (30 days)
            alpha3p: precipitation threshold for second tercile of window (20mm)
            alpha4p: precipitation threshold for third tercile of window (20mm)

        ts : numpy array
            daily precipitation timeseries with 29th Feb removed

        targets : list, optional
            only necessary when return_onset=False

        print_progress: bool, optional
            for optimizing purposes, prints rsme, parameters and result, default is False

        rsme_target: int, optional
            only if print_progress=True, highlights rsme results below defined threshold

        return_onset : bool, optional
            A flag used to either return list of rainy season onsets (True)
            or returns rsme between RSOs and targets (default is False)

        Returns
        -------
        list, when return_onset=True
            a list of rainy season onsets for each season

        rsme, when return_onset=False
            rsme value between validation and rainy season

    """

    alpha1p,alpha2d,alpha3p,alpha4p = params
    alpha2d = int(alpha2d)
    lonset = []
    for i in range(int(len(ts)/365)):
        j=0
        onset=None
        ts_sub = ts[0+i*365:365+i*365]
        while onset == None:
            ts_n_d_window = ts_sub[j:j+alpha2d]
            cond1 = ts_n_d_window[0:int(alpha2d/3)].sum()
            cond2 = ts_n_d_window[int(alpha2d/3):int(2*alpha2d/3)].sum()
            cond3 = ts_n_d_window[int(2*alpha2d/3):alpha2d].sum()
            if cond1 > alpha1p and cond2 > alpha3p and cond3 > alpha4p:
                onset = j+1
                lonset.append(onset)
            elif j > len(ts_sub):
                onset = -9999
                lonset.append(onset)
            j+=1

    if return_onset == False:
        vrsme = rsme(np.array(lonset),targets)
        if print_progress == True:
            prstr = (f'rsme = {np.round(vrsme,1)}, params = {np.round(params,3)}')
            if vrsme < rsme_target:
                print(f"\033[1m{prstr}\033[0m",end='\r')
            else:
                print(prstr,end='\r',flush=True)

        return vrsme
    else:
        return np.array(lonset)

# based on Jolliffe & Dodd (1994): Western Africa
def JD_RSO(params,ts,targets=None,print_progress=False,rsme_target=11,return_onset=False):
    """
    Calculates rainy season end based on Jolliffe & Dodd (1994):

        Jolliffe, I.T. and Dodd, D.E.S. (1994) Early detection of the start of the wet season in
        semiarid tropical climates of Western Africa. International Journal of Climatology, 14, 71–76.
        https://doi.org/10.1002/joc.640.

        Function
        ----------
        calculates day since first day of ts where alpha3d out of alpha2d days have rainfall greater alpha1p and
        sum(P(d:d+alpha2d) > alpha4p and Nc(P(d:d+alpha6d) < alpha1p) < alpha5d

        Parameters
        ----------
        params : list with 6 elements, values in () as published by the authors
            alpha1p: precipitation threshold for rainy day (0.1)
            alpha2d: n days window for alpha1p (5)
            alpha3d: n days within alpha2d (3)
            alpha4p: precipitation threshold for the sum of d:d+alpha2d (25)
            alpha5d: n consecutive days with precipitation < alpha1p threshold (10)
            alpha6d: n days window for alpha5d (30)

        ts : numpy array
            daily precipitation timeseries with 29th Feb removed

        targets : list, optional
            only necessary when return_onset=False

        print_progress: bool, optional
            for optimizing purposes, prints rsme, parameters and result, default is False

        rsme_target: int, optional
            only if print_progress=True, highlights rsme results below defined threshold

        return_onset : bool, optional
            A flag used to either return list of rainy season onsets (True)
            or returns rsme between RSOs and targets (default is False)

        Returns
        -------
        list, when return_onset=True
            a list of rainy season onsets for each season

        rsme, when return_onset=False
            rsme value between validation and rainy season

    """

    alpha1p,alpha2d,alpha3d,alpha4p,alpha5d,alpha6d = params
    lonset = []
    for i in range(int(len(ts)/365)):
        onset_found = False
        j=0
        onset=None
        ts_sub = ts[0+i*365:365+i*365]
        while onset == None and j < 364:
            # Condition1: X (3) in N (5) days have to have p > dd_thres (0.1mm) 
            window = ts_sub[j:j+int(alpha2d)]
            condition1 = len(np.where(window >= alpha1p)[0])
            # Condition2: The same N (5) day window has to have a precip sum > X (25mm)
            condition2 = window.sum()
            # if conditions 1&2 are true check for condition3, else continue
            if condition1 >= alpha3d and condition2 > alpha4p:
                ## Condition 3, check if CDD in n day window are greater than thres     
                # 1.Get all dry days in period (dry = True)
                dry_or_wet = pd.Series(ts_sub[j:j + int(alpha6d)] < alpha1p)         
                # 2.Get consecutive dry and wet days by reorganizing ts into lists of days with no change
                change_points = dry_or_wet.ne(dry_or_wet.shift()).cumsum()
                consecutive_conditions = [group.tolist() for _, group in dry_or_wet.groupby(change_points)]
                # 3. get only CDDs and count
                lenCDD = [len(x) for x in consecutive_conditions if x[0]] # x[0] = avoid checking all
                # 4. check if CDD is smaller than 
                condition3 = any(x >= alpha5d for x in lenCDD)
                if not condition3:
                    onset = j
                    lonset.append(onset)
                    onset_found = True     
            j+=1

        if not onset_found: 
            lonset.append(-9999)            

    if return_onset == False:
        vrsme = rsme(np.array(lonset),targets)
        if print_progress == True:
            prstr = (f'rsme = {np.round(vrsme,1)}, params = {np.round(params,3)}')
            if vrsme < rsme_target:
                print(f"\033[1m{prstr}\033[0m",end='\r')
            else:
                print(prstr,end='\r',flush=True)

        return vrsme
    else:
        return np.array(lonset)

# based on Liebmann & Marengo (2001): Brazilian Amazon Basin
def Liebmann(ts):
    """
    Calculate the accumulated seasonal rainfall indices against the seasonal average.

    Liebmann, B. and Marengo, J. (2001). Interannual Variability of the Rainy Season 
    and Rainfall in the Brazilian Amazon Basin. Journal of Climate, 14, 4308-4318. 
    https://doi.org/10.1175/1520-0442(2001)014<4308:IVOTRS>2.0.CO;2

    Parameters:
    ----------
    ts : numpy array
        daily precipitation timeseries with 29th Feb removed

    Returns:
    -------
    RSO_list : numpy.ndarray
        An array of the start of the rainy season for each year 

    RSE_list : numpy.ndarray
        An array of the end of the rainy season for each year 
    """
    RSO_list,RSE_list = list(),list()
    for i in range(int(len(ts)/365)):
        ts_sub = ts[0+i*365:365+i*365]
        cumudiff = (ts_sub - ts_sub.mean()).cumsum()
        RSO_list.append(np.argmin(cumudiff))
        RSE_list.append(np.argmax(cumudiff))

    return np.array(RSO_list), np.array(RSE_list)

# based on Cook&Buckley (2009): Thailand
def CookBuckley_2phase(ts):
    """
    Perform a two-phase optimized regression to determine the onset and end of the rainy season.

    Cook, B. I., & Buckley, B. M. (2009). Objective determination of monsoon season 
    onset, withdrawal, and length. Journal of Geophysical Research: Atmospheres, 114(D23).

    Parameters:
    ----------
    ts : numpy array
        daily precipitation timeseries with 29th Feb removed

    Returns:
    -------
    RSO_list : numpy.ndarray
        An array of the start of the rainy season for each year 

    RSE_list : numpy.ndarray
        An array of the end of the rainy season for each year 
    
    Notes:
    -------
    GitHub Repository for pwlf:
    https://github.com/cjekel/piecewise_linear_fit_py
    """
    import pwlf
    RSO_list,RSE_list = list(),list()

    for i in range(int(len(ts)/365)):
        ts_cumsum = ts[0+i*365:365+i*365].cumsum()


        ts_RSO = ts_cumsum[:250]
        ts_RSE = ts_cumsum[200:]

        for l,ts_ in zip([RSO_list,RSE_list],[ts_RSO,ts_RSE]):
            fitted = pwlf.PiecewiseLinFit(np.arange(len(ts_)),ts_)
            breaks = fitted.fit(2)
            l.append(int(np.round(breaks[1])))


    RSE_list = [x+200 for x in RSE_list]

    return np.array(RSO_list),np.array(RSE_list)    

# Hänchen et al. (subm.): Bucket metric, Rio Santa basin, Peruvian Andes
def bucket(params,ts,targets=None,print_progress=False,rsme_target=11,optim=None):
    """ Models LSP RSO/RSE from rainfall data input

        Function
        ----------
        simulates LSP RSO/RSE from rainfall data using a simplified water balance approach

        Parameters
        ----------
        params : list with 7 elements:
            bwc_init: initial water content (m3/m3)
            bwc_max: maximum water content (m3/m3)
            bwc_min: minimum water content (m3/m3)
            t_rso: threshold for RSO (m3/m3)
            t_rse: threshold for RSE (m3/m3)
            sd: depth of "bucket" (m)
            et: evapotranspiration constant (mm/day)

        ts : 1D numpy array
            daily precipitation timeseries with 29th Feb removed

        targets : list, optional
            targets for optimization

        print_progress: bool, optional
            for optimizing purposes, prints rsme, parameters and result, default is False

        rsme_target: int, optional
            only if print_progress=True, highlights rsme results below defined threshold

        optim : str, ('RSO' or 'RSE')
            parameter to specify the target of optimization

        Returns
        -------
        list, when optim=None
            a list of rainy season end for each season

        rsme, when optim = 'RSO' or optim = 'RSE'
            rsme value between targets and RSO or RSE

    """

    wd = 1000 # density of water (kg/m3)
    bwc_init,bwc_max,bwc_min,t_rso,t_rse,sd,et = params
    if optim == 'RSO':
        t_rse = 999
    bwc = ts*np.nan 
    lonset,lend = [],[]
    RSO_temp,RSE_temp = [],[]

    j = -1
    for i in range(len(ts)):
        if i % 365 == 0: #reset year
            if i==0:
                bwc[i] = bwc_init
            else:
                bwc[i] = bwc[i-1]

                if len(RSO_temp) > 0:
                    lonset.append(max(RSO_temp))
                else:
                    lonset.append(-9999)
                if len(RSE_temp) > 0:
                    lend.append(max(RSE_temp))
                else:
                    lend.append(-9999)

            j+=1 #year counter
            rso_flag = 0
            yearly_max = 0
            rse_flag = 1
            RSO_temp,RSE_temp = [],[]
        else:
            bwc[i] =   bwc[i-1]

        bwc[i] = bwc[i] + (ts[i] - et) / sd / wd

        #if maximum sw is exceeded, no further increase in bwc (surplus runs off)
        if bwc[i] > bwc_max:
            bwc[i] = bwc_max

        # if minimum bwc is reached, no further drawdown occurs
        if bwc[i] < bwc_min:
            bwc[i] = bwc_min

        # constantly update yearly maximum bwc
        if bwc[i] > yearly_max:
            yearly_max = bwc[i]

        # simulated rso occurs when critical bwc is exceeded first time in fall
        if (bwc[i] > t_rso) and (rso_flag == 0):
            rso = (i+0) - j*365 # i starts at 0 (+1) and j*365 gives you the day since 01-09
            if rso < 10: #rarely early detections can happen
                pass
            else:
                RSO_temp.append(rso)
                rso_flag = 1;

        # set flag for RSE condition to occur
        if (bwc[i] > t_rse) & (rse_flag == 1):
            rse_flag = 0;

        # simulated rse occurs after yearly maximum bwc has been reached and
        # bwc falls below critical value
        if (bwc[i] < yearly_max) & (bwc[i] < t_rse) & (rse_flag == 0):
            rse = (i+0) - j*365
            RSE_temp.append(rse)
            #print('RSE',rse)
            rse_flag = 1

    #after last iteration append last RSO,RSE
    if len(RSO_temp) > 0:
        lonset.append(max(RSO_temp))
    else:
        lonset.append(-9999)
    if len(RSE_temp) > 0:
        lend.append(max(RSE_temp))
    else:
        lend.append(-9999)


    if optim=='RSO': 
        vrsme = rsme(np.array(lonset),targets)
        if print_progress == True:
            prstr = (f'rsme = {vrsme}, params = {np.round(params,3)}')
            if vrsme < rsme_target:
                print(f"\033[1m{prstr}\033[0m",end='\r',flush=True)
            else:
                print(prstr,end='\r',flush=True)
        return vrsme

    elif optim=='RSE':
        vrsme = rsme(np.array(lend),targets)
        if print_progress == True:
            prstr = (f'rsme = {vrsme}, params = {np.round(params,3)}')
            if vrsme < rsme_target:
                print(f"\033[1m{prstr}\033[0m",end='\r',flush=True)
            else:
                print(prstr,end='\r',flush=True)
        return vrsme

    else:
        return np.array(lonset),np.array(lend)      