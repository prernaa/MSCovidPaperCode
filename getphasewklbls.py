import os
import pandas as pd
import numpy as np
# import pingouin as pg
from timeutils import sensor_time_to_local, EASTERN, add_survey_time_cols
from timesettings import COVID_CUT_OFF, LOCKDOWN_YELLOW_START, LOCKDOWN_GREEN_START


def add_phase_label(df, ts_col):
    df["phase"] = np.where(df[ts_col]<COVID_CUT_OFF, "pre", \
                           np.where( (df[ts_col]>=COVID_CUT_OFF) & (df[ts_col]<LOCKDOWN_YELLOW_START), "lock", \
                                    np.where( (df[ts_col]>=LOCKDOWN_YELLOW_START) & (df[ts_col]<LOCKDOWN_GREEN_START), "yellow", \
                                             "green"
                                            )
                                   )
                          )
    return df

def add_phase_label_wrapper(df, ts_col):
    df = add_phase_label(df, ts_col)
    df["phase_wrapper"] = np.where( ((df["phase"]=="yellow") |  (df["phase"]=="green") ), "post", df["phase"])
    return df


def add_week_label_row_of_person(df_survey_did, data_row, ts_col):
#     display (data_row)
#     display (df_survey_did)
    data_ts = data_row[ts_col]
#     display ((df_survey_did[ts_col]-data_ts).dt.days)
#     print (type((df_survey_did[ts_col]-data_ts)))
    df_survey_gte = df_survey_did[(df_survey_did[ts_col]>=data_ts)]
    df_survey_gte_biweekly = df_survey_gte[(df_survey_gte["redcap_event_name"].isin(BIWEEKLY_EVENTS))]
    df_survey_gte_monthly = df_survey_gte[(df_survey_gte["redcap_event_name"].isin(MONTHLY_EVENTS))]
    df_survey_lte = df_survey_did[(df_survey_did[ts_col]< data_ts)] # is it in between the study
#     display (df_survey_gte)

    ## BIWEEKLY
    if len(df_survey_gte_biweekly)==0: # after the last week
        biweekly_lbl = "extra"
        biweekly_date = np.nan
    else:
        df_survey_select_14_days = df_survey_gte_biweekly[((df_survey_gte_biweekly[ts_col]-data_ts).dt.days)<=14]
        if len(df_survey_select_14_days)==0: ## there is a survey after data point but diff is > 14 days
            if len(df_survey_lte)==0: # if there is no survey before data then "pre-study"
                biweekly_lbl = "pre"
            else:
                biweekly_lbl = np.nan # else, missing in between due to delay etc 
            biweekly_date = np.nan
        else:
            biweekly_lbl = df_survey_select_14_days["redcap_event_name"].iloc[0]
            biweekly_date = df_survey_select_14_days["date_dt"].iloc[0]

    ## MONTHLY
    if len(df_survey_gte_monthly)==0: # after the last week
        monthly_lbl = "extra"
        monthly_date = np.nan
    else:
        df_survey_select_31_days = df_survey_gte_monthly[((df_survey_gte_monthly[ts_col]-data_ts).dt.days)<=31]
        if len(df_survey_select_31_days)==0: ## there is a survey after data point but diff is > 14 days
            if len(df_survey_lte)==0:
                monthly_lbl = "pre"
            else:
                monthly_lbl = "np.nan"
            monthly_date = np.nan
        else:
            monthly_lbl = df_survey_select_31_days["redcap_event_name"].iloc[0]
            monthly_date = df_survey_select_31_days["date_dt"].iloc[0]

    
#     if len(df_survey_gte)==0: # after the last week
#         biweekly_lbl = "extra"
#         monthly_lbl = "extra"
#         biweekly_date = np.nan
#         monthly_date = np.nan
#     else:
#         df_survey_select_14_days = df_survey_gte_biweekly[((df_survey_gte_biweekly[ts_col]-data_ts).dt.days)<=14]
#         try:
#             df_survey_select_31_days = df_survey_gte_monthly[((df_survey_gte_monthly[ts_col]-data_ts).dt.days)<=31]
#         except:
#             display(data_row)
#             display(df_survey_gte)
#             display(df_survey_gte_monthly)
#     #     display (df_survey_select_14_days)
#     #     display (df_survey_select_31_days)
#         if len(df_survey_select_14_days)==0: ## there is a survey after data point but diff is > 14 days
#             if len(df_survey_lte)==0: # if there is no survey before data then "pre-study"
#                 biweekly_lbl = "pre"
#             else:
#                 biweekly_lbl = np.nan # else, missing in between due to delay etc 
#             biweekly_date = np.nan
#         else:
#             biweekly_lbl = df_survey_select_14_days["redcap_event_name"].iloc[0]
#             biweekly_date = df_survey_select_14_days["date_dt"].iloc[0]
#         if len(df_survey_select_31_days)==0: ## there is a survey after data point but diff is > 14 days
#             if len(df_survey_lte)==0:
#                 monthly_lbl = "pre"
#             else:
#                 monthly_lbl = "np.nan"
#             monthly_date = np.nan
#         else:
#             monthly_lbl = df_survey_select_31_days["redcap_event_name"].iloc[0]
#             monthly_date = df_survey_select_31_days["date_dt"].iloc[0]

    data_row["biweekly_lbl"] = biweekly_lbl
    data_row["monthly_lbl"] = monthly_lbl
    data_row["biweekly_date"] = biweekly_date
    data_row["monthly_date"] = monthly_date
    return data_row
            
    

def add_week_label(df, ts_col, df_survey):
#     print (type(df[ts_col].iloc[0]))
#     print (type(df_survey[ts_col].iloc[0]))
    
#     display(df_survey.head(1))
    
    df_survey = df_survey.sort_values(by=["device_id", "timestamp_dt"])
    df_did_list = []
    print ("Processing did counts...")
    cnt = 0
    for did in df_survey["device_id"].unique():
        cnt += 1
#         print ("{0}, ".format(cnt), end='')
        print "{0}, ".format(cnt), 
        df_survey_did = df_survey[(df_survey["device_id"]==did)][[ts_col, "redcap_event_name"]]
        df_did = df[(df["device_id"]==did)]
        df_did = df_did.apply((lambda x: (add_week_label_row_of_person(df_survey_did, x, ts_col))), axis=1)
        df_did["biweekly_date_dt_diff"] = df_did["biweekly_date"] - df_did["date_dt"] 
        df_did["monthly_date_dt_diff"] = df_did["monthly_date"] - df_did["date_dt"] 
        cols_disp = ["timestamp_dt", "date_dt", "biweekly_lbl", "monthly_lbl", \
                        "biweekly_date", "monthly_date",\
                        "biweekly_date_dt_diff", "monthly_date_dt_diff"
                       ]
#         display(df_did[cols_disp])
#         display(df_did)
        df_did_list.append(df_did)
#         break # debug
    df_dids_all = pd.concat(df_did_list)
    return df_dids_all
        
        
        

