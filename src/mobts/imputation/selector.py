"""
Mixed utility module, concerned with selections for the donor-methods

This module contains:
- identifying counters with missing counts
- determining the minimum mutual period of donors from the config, based on the temporal frequency of the project
- determining the minimum prediction period used in regression from the config, based on the temporal frequency of the project
- function for determining if the counter is eligible to be filled in using the scaled medians method
- function for selecting donor stations for the regression method
- function for determining if the counter is eligible to be filled in using the regression method
- determining eligible imputation method for each counter

"""



import pandas as pd
import numpy as np

from ..configs.config_common import ColumnsConfig
from ..configs.config_imputation import DonorsConfig, OutputConfig
from..utils.formatting import _validate_frequency



def _find_counters_with_holes(
    df: pd.DataFrame,
    count_col: str,
    counter_col: str,
) -> list:

    """
    Finds counters with missing values

    ------
    Parameters:
    
    - df: preprocessed network dataframe
    - count_col: count column
    - counter_col: counter column

    -----
    Returns:

    - list of counters that have missing counts
    """
    
    miss = df[count_col].isna()
    return df.loc[miss, counter_col].unique().tolist()



def _get_min_mutual_period(
    freq: str, 
    donors_cfg: DonorsConfig = DonorsConfig(),
) -> int:

    """
    determines the minimun mutual period for donors from config

    ------
    Parameters:
    
    - freq: temporal frequency of the project
    - donors_cfg: donor config

    -----
    Returns:

    - minimum mutual period
    """
    
    freq = _validate_frequency(freq)
    if freq == "hourly":
        return donors_cfg.min_mutual_hours
    if freq == "daily":
        return donors_cfg.min_mutual_days
    else:
        raise ValueError(f"Unsupported frequency: {freq}")


        
def _get_min_prediction_period(
    freq: str, 
    donors_cfg: DonorsConfig = DonorsConfig(),
) -> int:

    """
    determines the minimum prediction for donors from config

    ------
    Parameters:
    
    - freq: temporal frequency of the project
    - donors_cfg: donor config

    -----
    Returns:

    - minimum prediction period needed for regression 
    """
    
    freq = _validate_frequency(freq)
    if freq == "hourly":
        return donors_cfg.min_pred_hours
    if freq == "daily":
        return donors_cfg.min_pred_days
    else:
        raise ValueError(f"Unsupported frequency: {freq}")  


        
def _is_eligible_for_scaled_median(
    target: str,
    pivot: pd.DataFrame,
    freq: str,
    donors: list[str],
    donors_cfg: DonorsConfig = DonorsConfig(),
) -> bool:

    """
    Determines if the counter is eligible for scaled median method

    ------
    Parameters:
    
    - target: the counter that is the target of the function
    - pivot: pivotted form the data (timestamp index, counter columns, count values)
    - freq: temporal frequency of the project
    - donors: list of donors retrieved from the donor map
    - donors_cfg: donor config

    -----
    Returns:

    - boolean indicating if the counter is eligible for scaled median imputation method
    """
    
    if freq == "daily":
        sm_min_overlap = donors_cfg.sm_min_overlap_day
        
    elif freq == "hourly":
        sm_min_overlap = donors_cfg.sm_min_overlap_hour

    # an initial check to see if the target itself has enough available valid counts
    y_t = pivot[target]
    avail_idx = y_t.index[y_t.notna()]
    if avail_idx.size < sm_min_overlap:
        return False


    # goes through donors and checks if the overlapping observations are more than the minimum
    valid_donors = []
    sm_counter = 0
    
   
    sm_counter = 0 
    for d in donors:
        
        if d in pivot.columns and pivot[[target, d, *valid_donors]].notna().all(axis=1).sum() > sm_min_overlap:

            # if a donor is eligible, the validity (non-infinite) of the medians is also checked, then donor is added
            # to the valid donors list
            arr = pivot.loc[avail_idx, d].to_numpy(dtype=float)
            
            if np.isfinite(arr).any():
                
                md = np.nanmedian(arr)
                
                if np.isfinite(md):
                    valid_donors.append(d)
                    
            sm_counter +=1

        # once the donor numbet hits the top K donor count, the loop stops
        if sm_counter == donors_cfg.top_k_donor:
            break
    # checks if sm_counter is bigger than minimum neighbor number set in config        
    is_eligible  = (sm_counter > donors_cfg.sm_min_neighbors)
    
    return is_eligible
    



def _select_regression_donors(
    target: str,
    pivot: pd.DataFrame,
    freq: str,
    donors: list,
    donors_cfg: DonorsConfig = DonorsConfig(),
) -> list:

    """
    Selects donors for regression

    ------
    Parameters:
    
    - target: the counter that is the target of the function
    - pivot: pivotted form the data (timestamp index, counter columns, count values)
    - freq: temporal frequency of the project
    - donors: list of donors retrieved from the donor map
    - donors_cfg: donor config

    -----
    Returns:

    - list of eligible donors for the regression imputation
    """
    
    y = pivot[target]
    miss_idx = y.index[y.isna()]

    min_mutual_period = _get_min_mutual_period(freq, donors_cfg)
    min_pred_period = _get_min_prediction_period(freq, donors_cfg)

    # going through each donor to check if it satisfies the conditions for regression donors
    selected: list = []
    max_d = int(len(donors) * donors_cfg.max_donor_rate)

    for d in donors[:max_d]:

        # locate the missing period in the target, and see if the donor has enough observations for imputation in that period
        if miss_idx.size > 0:
            coverage = pivot.loc[miss_idx, d].notna().mean()
            if (miss_idx.size < min_pred_period) and (coverage == 0):
                continue
            if (miss_idx.size >= min_pred_period) and (coverage < donors_cfg.min_pred_coverage):
                continue
                
        # adds the candidate and check the new group's eligibility in terms of minimula mutual days
        cand = selected + [d]
        Xc = pivot[cand]
        mutual = (y.notna() & Xc.notna().all(axis=1)).sum()
        if mutual < min_mutual_period:
            continue

        selected.append(d)

        # stops adding donors once the number hits top_k
        if len(selected) >= donors_cfg.top_k_donor:
            break

    return selected


def _is_eligible_for_regression(
    target: str,
    pivot: pd.DataFrame,
    freq: str,
    donors: list,
    donors_cfg: DonorsConfig = DonorsConfig(),
) -> bool:

    """
    Determines if the counter is eligible for regression imputation method

    ------
    Parameters:
    
    - target: the counter that is the target of the function
    - pivot: pivotted form the data (timestamp index, counter columns, count values)
    - freq: temporal frequency of the project
    - donors: list of donors retrieved from the donor map
    - donors_cfg: donor config

    -----
    Returns:

    - boolean indicating if the counter is eligible for regression imputation method
    """

    selected = _select_regression_donors(target=target, pivot=pivot, freq=freq, donors=donors, donors_cfg=donors_cfg)

    if len(selected) > 0:
        return True
    else:
        return False



def _counter_method_choice(
    target: str,
    pivot: pd.DataFrame,
    donor_map: dict[str, list],
    freq: str,
    donors_cfg: DonorsConfig = DonorsConfig(),
    out_cfg: OutputConfig = OutputConfig(),
) -> str:

    """
    picks the best eligible method for each counter (first M8, then M7, and then STL)

    ------
    Parameters:
    
    - target: the counter that is the target of the function
    - pivot: pivotted form the data (timestamp index, counter columns, count values)
    - donor_map: dictionary map of donors
    - freq: temporal frequency of the project
    - donors_cfg: donor config
    - out_cfg: output config

    -----
    Returns:

    - string indicating the best eligible method for the target counter
    """
    
    donors = donor_map.get(target, [])
    if _is_eligible_for_regression(target, pivot, freq, donors, donors_cfg):
        return out_cfg.reg_method
    elif _is_eligible_for_scaled_median(target, pivot, freq, donors, donors_cfg):
        return out_cfg.sm_method
    else:
        return out_cfg.stl_method
