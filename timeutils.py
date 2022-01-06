import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import pytz
from timesettings import SURVEY_WINDOW_DAYS
GMT = pytz.timezone('GMT')
EASTERN = pytz.timezone('America/New_York')
UTC = pytz.utc

def add_survey_time_cols(df_survey):
    window = SURVEY_WINDOW_DAYS
    df_survey["timestamp_dt"] = pd.to_datetime(df_survey['timestamp'], format= '%Y-%m-%d %H:%M:%S' )
    df_survey["date_dt"] = df_survey["timestamp_dt"].dt.date
    df_survey["timestamp_dt_dow"] = df_survey["timestamp_dt"].dt.dayofweek
    df_survey["timestamp_dt_dow_wrt_sat"] = np.where(df_survey["timestamp_dt_dow"].isin([5,6]), df_survey["timestamp_dt_dow"]-5, 2+df_survey["timestamp_dt_dow"])
    df_survey["date_dt_prev_sat"] = df_survey.apply((lambda x: x["date_dt"]-pd.Timedelta(x["timestamp_dt_dow_wrt_sat"], unit='D')), axis=1)
    df_survey["date_dt_win_start"] = df_survey.apply((lambda x: x["date_dt"]-pd.Timedelta(window, unit='D')), axis=1)
#     df_survey["date_dt_prev_sat_win_start"] = df_survey.apply((lambda x: x["date_dt_prev_sat"]-pd.Timedelta(window, unit='D')), axis=1)
    return df_survey


def sensor_time_to_local_with_tz(raw_ts):
    utcDt = datetime.utcfromtimestamp(raw_ts/1000.0)
    utcDt = utcDt.replace(tzinfo=UTC)
    etDt = utcDt.astimezone(EASTERN)  
    return (etDt)

def sensor_time_to_local(raw_ts):
    etDt = sensor_time_to_local_with_tz(raw_ts)
    dt = etDt.replace(tzinfo=None)
    return (dt)

def fix_times(df, cols=["timestamp"]):
    """Returns the same dataframe with a new datetime column accounting for daylight savings time"""
    x = 0
    for col in cols:
        datetimes = pd.to_datetime(df[col], unit="ms")
        eastern = EASTERN
        datetimes_eastern = datetimes.apply(lambda x: x.tz_localize(UTC).astimezone(eastern))
        if col == "timestamp":
            name = "datetime_EST"
        else:
            name = col + "_EST"
        df[name] = datetimes_eastern
    return df

def resampleseparatedays(data, sr_in_min, unit_in_str):
#     warnings.warn("resampleseparate arrays strips timezone info from input data to deal with daylight savings. Disregard timezone moving forward")
    # data is not continuous, Eg: night of day 1, night of day 2, and so on
    g = data.groupby(data.index.date)
    gkeys = sorted(g.groups.keys())
    days_data = []
    for gk in gkeys:
        day_data = g.get_group(gk)
        day_data = day_data.tz_localize(None) # throws time zone info, keeps same time
        day_data = day_data.resample(str(sr_in_min)+unit_in_str).ffill()
        #display(day_data)
        dlist = day_data.index.view('int64')/1000000
        day_data["timestamp"] = dlist
        # ffill can create nans at the beginning (cause there's nothing to forward fill). Lets drop them
        idxnotnull = ~day_data.isnull().any(axis=1).values
        day_data = day_data[idxnotnull]
        #print ("{0} has {1} samples".format(gk, len(day_data)))
        days_data.append(day_data)
    resampled = pd.concat(days_data)
    return resampled

def resampleseparatedays_min(data, sr_in_min):
    return resampleseparatedays(data, sr_in_min, "min")

def resampleseparatedays_sec(data, sr_in_min):
    return resampleseparatedays(data, sr_in_min, "S")
