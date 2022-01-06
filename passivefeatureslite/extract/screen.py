import pandas as pd
import numpy as np



def number_of_unlocks(g, status_col="screen_status", unlock_val=3):
    g = g[(g[status_col]==unlock_val)]
    return len(g)

def mean_unlocks_per_minute(g, status_col="screen_status", unlock_val=3, timestamp_dt_col="timestamp_dt"):
    g = g[(g[status_col]==unlock_val)]
    if len(g)==0:
        return 0
#     g = g.set_index(timestamp_dt_col)
    unlocks_vs_min = g.groupby(pd.Grouper(key=timestamp_dt_col, freq='1Min'))[status_col].count()
    return unlocks_vs_min.mean()

def median_unlocks_per_minute(g, status_col="screen_status", unlock_val=3, timestamp_dt_col="timestamp_dt"):
    g = g[(g[status_col]==unlock_val)]
    if len(g)==0:
        return 0
#     g = g.set_index(timestamp_dt_col)
    unlocks_vs_min = g.groupby(pd.Grouper(key=timestamp_dt_col, freq='1Min'))[status_col].count()
    return unlocks_vs_min.median()

def interaction_time_minutes(g, status_col="screen_status", time_spent_col="time_spent_secs"):
    if g["time_spent_secs"].count()==0:
        return 0
    return (g["time_spent_secs"].sum()/(60.0))
                             
def mean_interaction_time_per_use_secs(g, status_col="screen_status", time_spent_col="time_spent_secs"):
    if g["time_spent_secs"].count()==0:
        return 0
    return (g["time_spent_secs"].mean())
                                       
def median_interaction_time_per_use_secs(g, status_col="screen_status", time_spent_col="time_spent_secs"):
    if g["time_spent_secs"].count()==0:
        return 0
    return (g["time_spent_secs"].median())


# def get_interactions(g, status_col="screen_status", map_dict = {0:"off", 1:"on", 2:"lock", 3:"unlock"}, timestamp_ms_col="timestamp"):
#     g["lbl"] = g[status_col].apply((lambda x: map_dict[x]))
#     g["lock_unlock"] = [np.nan]*len(g)
#     g["lock_unlock"] = np.where(g["lbl"]=="unlock", 1, g["lock_unlock"])
#     g["lock_unlock"] = np.where(g["lbl"]=="lock", 0, g["lock_unlock"])
#     g["off_on"] = [np.nan]*len(g)
#     g["off_on"] = np.where(g["lbl"]=="on", 1, g["off_on"])
#     g["off_on"] = np.where(g["lbl"]=="off", 0, g["off_on"])
#     g["lock_unlock"] = g["lock_unlock"].ffill()
#     g["off_on"] = g["off_on"].ffill()
#     g["next_{0}".format(timestamp_ms_col)] = g[timestamp_ms_col].shift(periods=-1)
#     g["time_spent_secs"] = (g["next_{0}".format(timestamp_ms_col)] - g[timestamp_ms_col])/(1000.0)
#     cond = ((g["lock_unlock"]==1) & ((g["off_on"]==1) | (g["off_on"].isnull())))
#     g["time_spent_secs"] = np.where(cond, g["time_spent_secs"], np.nan)
#     g = g[(g["time_spent_secs"].notnull())]
#     return (g)

# def interaction_time_minutes(g, status_col="screen_status", map_dict = {0:"off", 1:"on", 2:"lock", 3:"unlock"}, timestamp_ms_col="timestamp"):
#     g_inter = get_interactions(g, status_col, map_dict, timestamp_ms_col)
#     return (g["time_spent_secs"].sum()/(60.0))

# def mean_interaction_time_per_use_secs(g, status_col="screen_status", map_dict = {0:"off", 1:"on", 2:"lock", 3:"unlock"}, timestamp_ms_col="timestamp"):
#     g_inter = get_interactions(g, status_col, map_dict, timestamp_ms_col)
#     return (g["time_spent_secs"].mean())

# def median_interaction_time_per_use_secs(g, status_col="screen_status", map_dict = {0:"off", 1:"on", 2:"lock", 3:"unlock"}, timestamp_ms_col="timestamp"):
#     g_inter = get_interactions(g, status_col, map_dict, timestamp_ms_col)
#     return (g["time_spent_secs"].median())
