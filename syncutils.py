import pandas as pd
import numpy as np

def get_sensor_samples_from_grp(x, df_sensor_did):
    dts = x.name
    st = dts[0]
    en = dts[1]
    sel = df_sensor_did[( (df_sensor_did["date_dt"]>st) & (df_sensor_did["date_dt"]<=en) )]
    #print (dts)
    #display(x)
    #display(sel)
    return sel
    
def get_sensor_samples_per_survey_measure(df_sensor_in, df_survey_in):
    df_sensor = df_sensor_in.copy(deep=True)
    df_survey = df_survey_in.copy(deep=True)
    df_sensor = df_sensor.sort_values(by=["device_id", "timestamp_dt"])
    df_sensor = df_sensor.set_index("device_id")
    df_survey = df_survey.set_index("device_id")
    df_survey["redcap_date_dt"] = df_survey["date_dt"]
    df_survey["redcap_date_dt_win_start"] = df_survey["date_dt_win_start"]
    ## add columns to sensor df
    cols = ["redcap_event_name", "redcap_date_dt", "redcap_date_dt_win_start", \
            "phq9_score", "phq12_functioning", "msrsr_m", "pss_m", "mfis_m", "psqi_total", "pdds"
           ]
    all_dids = list(df_sensor.index.unique())
    total_dids = len(all_dids)
    cnt = 0
    df_sensors_out_list = []
    for did in all_dids:
        cnt += 1
        if (cnt%30)==0:
            print ("Syncing for {0} out of {1} dids".format(cnt, total_dids))
        df_sensor_did = df_sensor.loc[[did]]
        df_survey_did = df_survey[cols].loc[[did]]
        ## for each survey measure get sensor_samples
        df_sensor_did_out = df_survey_did.groupby(["redcap_date_dt_win_start", "redcap_date_dt"]).apply((lambda x: get_sensor_samples_from_grp(x, df_sensor_did))).reset_index()
        df_sensor_did_out = pd.merge(df_sensor_did_out, df_survey_did,  how='left', left_on=["redcap_date_dt_win_start", "redcap_date_dt"], right_on = ["redcap_date_dt_win_start", "redcap_date_dt"])
        df_sensor_did_out["device_id"] = [did]*len(df_sensor_did_out)
        df_sensor_did_out = df_sensor_did_out[["device_id"]+list(df_sensor_did.columns)+cols]
        #display(df_survey_did)
        #display(df_sensor_did_out)
        df_sensors_out_list.append(df_sensor_did_out)
    df_sensors_out = pd.concat(df_sensors_out_list)
    print ("Done!")
    return df_sensors_out
        

