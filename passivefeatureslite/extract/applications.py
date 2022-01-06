import numpy as np
import pandas as pd




def number_of_total_launches_apps(g, pkg_col):
    return g[pkg_col].count()

def number_of_unique_apps(g, pkg_col):
    return g[pkg_col].nunique()






def number_of_launches_category(g, cat_col, sel_cat_name):
    sel_g = g[(g[cat_col]==sel_cat_name.upper())]
    return len(sel_g)

def sum_time_spent_category(g, cat_col, sel_cat_name, time_spent_col):
    sel_g = g[(g[cat_col]==sel_cat_name.upper())]
    return (sel_g[time_spent_col].sum())

def mean_time_spent_category(g, cat_col, sel_cat_name, time_spent_col):
    sel_g = g[(g[cat_col]==sel_cat_name.upper())]
    return (sel_g[time_spent_col].mean())






def apps_per_minute(g, pkg_col, timestamp_ms_col): 
    ## may not give us meaningful values for large lengths of time
    count = g[pkg_col].count()
    minutes = (g[timestamp_ms_col].max() - g[timestamp_ms_col].min()) / (1000.0 * 60.0)
    if minutes == 0:
        return 0
    else:
        return count / minutes

 