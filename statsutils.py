import pandas as pd
import numpy as np
import datetime
import scipy.stats as st

def cut_dates_idx(df):
    ll = datetime.datetime(2019, 11, 16) # first survey (week 0)
    ul = datetime.datetime(2021, 1, 25) # last survey was on the day before this date
    df = df[((df.index >= ll) & (df.index < ul))]
    return df

def cut_dates(df, dt_col="timestamp_dt"):
    ll = datetime.datetime(2019, 11, 16) # first survey (week 0)
    ul = datetime.datetime(2021, 1, 25) # last survey was on the day before this date
    df = df[((df[dt_col] >= ll) & (df[dt_col] < ul))]
    return df


def norm_for_person(df_grp_in, col_list, person_id_col):
    df_grp = df_grp_in.copy(deep=True)
    for col in col_list:
        tmp = (df_grp[col] - df_grp[col].mean())/df_grp[col].std(ddof=0)
        df_grp[col] = tmp
    df_grp = df_grp.drop(columns=[person_id_col])
    return df_grp

def norm_per_person_wrapper(df, col_list, person_id_col):
    df_out = df.groupby(person_id_col).apply((lambda x: norm_for_person(x, col_list, person_id_col))).reset_index()
    return df_out


def get_95_ci(a):
    tmp = st.t.interval(0.95, len(a)-1, loc=np.nanmean(a), scale=st.sem(a, nan_policy='omit', ddof=1))
    return tmp
def get_95_ci_lower(a):
    tmp = get_95_ci(a)
    return tmp[0]
def get_95_ci_upper(a):
    tmp = get_95_ci(a)
    return tmp[1]

def get_95_ci_med(a):
    tmp = st.t.interval(0.95, len(a)-1, loc=np.nanmedian(a), scale=st.sem(a, nan_policy='omit', ddof=1))
    return tmp
def get_95_ci_med_lower(a):
    tmp = get_95_ci_med(a)
    return tmp[0]
def get_95_ci_med_upper(a):
    tmp = get_95_ci_med(a)
    return tmp[1]

