import pandas as pd
import numpy as np

def add_extra_status_cols_for_did(g, status_col="screen_status", map_dict={0:"off", 1:"on", 2:"lock", 3:"unlock"}, timestamp_ms_col="timestamp"):
    g["lbl"] = g[status_col].apply((lambda x: map_dict[x]))
    g["lock_unlock"] = [np.nan]*len(g)
    g["lock_unlock"] = np.where(g["lbl"]=="unlock", 1, g["lock_unlock"])
    g["lock_unlock"] = np.where(g["lbl"]=="lock", 0, g["lock_unlock"])
    g["off_on"] = [np.nan]*len(g)
    g["off_on"] = np.where(g["lbl"]=="on", 1, g["off_on"])
    g["off_on"] = np.where(g["lbl"]=="off", 0, g["off_on"])
    g["lock_unlock"] = g["lock_unlock"].ffill()
    g["off_on"] = g["off_on"].ffill()
    g["next_{0}".format(timestamp_ms_col)] = g[timestamp_ms_col].shift(periods=-1)
    g["time_spent_secs"] = (g["next_{0}".format(timestamp_ms_col)] - g[timestamp_ms_col])/(1000.0)
    cond = ((g["lock_unlock"]==1) & ((g["off_on"]==1) | (g["off_on"].isnull())))
    g["time_spent_secs"] = np.where(cond, g["time_spent_secs"], np.nan)
    return g
    
    