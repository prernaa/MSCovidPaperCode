import pandas as pd
import numpy as np

PHQ_BIWEEKLY_EVENTS_MAIN = ['week_0', 'week_2', 'week_4', 'week_6', 'week_8', \
                   'week_10', 'week_12']
PHQ_BIWEEKLY_EVENTS_EXTENDED = ['week_14', 'week_16', 'week_18', 'week_20', 'week_22', \
              'week_24', 'week_26']

MONTHLY_EVENTS_MAIN = ['week_0', 'week_4', 'week_8', 'week_12']
MONTHLY_EVENTS_EXTENDED = ['week_14', 'week_18', 'week_22', 'week_26']

PHQ_SCORE_COL = "phq9_score"
PHQ_FUNC_COL = "phq12_functioning"
PHQ_LVL_COL = "phq9_lvl"
PHQ_CLASS_COL = "phq9_class"
PHQ_CLASS_HIGH_COL = "phq9_class_high"

def transform_phq_df(df, events, val_col):
    pids = list(df["record_id"].unique())
    print ("{0} number of subjects".format(len(pids)))
    pid_dict_list = []
    for pid in pids:
        pid_dict = {}
        pid_dict["record_id"] = pid
        for e in events:
            df_pid_e = df[((df["record_id"]==pid) & (df["redcap_event_name"]==e))]
            if len(df_pid_e) == 0:
                phq_val = np.nan
            else:
                phq_val = df_pid_e[val_col].iloc[0]
            pid_dict[e] = phq_val
        pid_dict_list.append(pid_dict)
    trans_df = pd.DataFrame(pid_dict_list)
    trans_df = trans_df[["record_id"]+events]
    trans_df = trans_df.set_index("record_id")
    return trans_df

def add_phq_lvl(x_row):
    x_row["wk_num"] = x_row["redcap_event_name"].strip("week_")
    if pd.isnull(x_row[PHQ_SCORE_COL]):
        x_row[PHQ_LVL_COL] = np.nan
    elif x_row[PHQ_SCORE_COL] >= 0 and  x_row[PHQ_SCORE_COL] <= 4:
        x_row[PHQ_LVL_COL] = 0
    elif x_row[PHQ_SCORE_COL] >= 5 and  x_row[PHQ_SCORE_COL] <= 9:
        x_row[PHQ_LVL_COL] = 1
    elif x_row[PHQ_SCORE_COL] >= 10 and  x_row[PHQ_SCORE_COL] <= 14:
        x_row[PHQ_LVL_COL] = 2
    elif x_row[PHQ_SCORE_COL] >= 15 and  x_row[PHQ_SCORE_COL] <= 19:
        x_row[PHQ_LVL_COL] = 3
    elif x_row[PHQ_SCORE_COL] >= 20 and  x_row[PHQ_SCORE_COL] <= 27:
        x_row[PHQ_LVL_COL] = 4
    else:
        x_row[PHQ_LVL_COL] = 5# never happens
    return x_row

def add_phq_class(x_row):
    x_row["wk_num"] = x_row["redcap_event_name"].strip("week_")
    if pd.isnull(x_row[PHQ_SCORE_COL]) and pd.isnull(x_row[PHQ_FUNC_COL]):
        x_row[PHQ_CLASS_COL] = np.nan
    elif pd.isnull(x_row[PHQ_SCORE_COL]) and not pd.isnull(x_row[PHQ_FUNC_COL]):
        if x_row[PHQ_FUNC_COL] >= 1:
            x_row[PHQ_CLASS_COL] = 1
        else:
            x_row[PHQ_CLASS_COL] = 0
    elif not pd.isnull(x_row[PHQ_SCORE_COL]) and pd.isnull(x_row[PHQ_FUNC_COL]):
        if x_row[PHQ_SCORE_COL] >= 10:
            x_row[PHQ_CLASS_COL] = 1
        else:
            x_row[PHQ_CLASS_COL] = 0
    else:
        if (x_row[PHQ_FUNC_COL] >= 1) or (x_row[PHQ_SCORE_COL] >= 10):
            x_row[PHQ_CLASS_COL] = 1
        else:
            x_row[PHQ_CLASS_COL] = 0
    return x_row

def add_phq_class_high(x_row):
    PHQ_HIGH_THRES = 14
    x_row["wk_num"] = x_row["redcap_event_name"].strip("week_")
    if pd.isnull(x_row[PHQ_SCORE_COL]) and pd.isnull(x_row[PHQ_FUNC_COL]):
        x_row[PHQ_CLASS_HIGH_COL] = np.nan
    elif pd.isnull(x_row[PHQ_SCORE_COL]) and not pd.isnull(x_row[PHQ_FUNC_COL]):
        if x_row[PHQ_FUNC_COL] >= 1:
            x_row[PHQ_CLASS_HIGH_COL] = 1
        else:
            x_row[PHQ_CLASS_HIGH_COL] = 0
    elif not pd.isnull(x_row[PHQ_SCORE_COL]) and pd.isnull(x_row[PHQ_FUNC_COL]):
        if x_row[PHQ_SCORE_COL] >= PHQ_HIGH_THRES:
            x_row[PHQ_CLASS_HIGH_COL] = 1
        else:
            x_row[PHQ_CLASS_HIGH_COL] = 0
    else:
        if (x_row[PHQ_FUNC_COL] >= 1) or (x_row[PHQ_SCORE_COL] >= PHQ_HIGH_THRES):
            x_row[PHQ_CLASS_HIGH_COL] = 1
        else:
            x_row[PHQ_CLASS_HIGH_COL] = 0
    return x_row


# def clean_phq(x_row):
#     x_row["wk_num"] = x_row["redcap_event_name"].strip("week_").strip("_arm_1")
#     if pd.isnull(x_row["biweeklyq_timestamp"]) and pd.isnull(x_row["monthlyq_timestamp"]):
#         x_row["phq"] = np.nan
#     elif pd.isnull(x_row["phq_bi"]) and pd.isnull(x_row["phq_m"]):
#         x_row["phq"] = 0
#     elif pd.isnull(x_row["phq_bi"]):
#         x_row["phq"] = x_row["phq_m"]
#     else:
#         x_row["phq"] = x_row["phq_bi"]
#     if pd.isnull(x_row["biweeklyq_timestamp"]) and pd.isnull(x_row["monthlyq_timestamp"]):
#         x_row["q_timestamp"] = np.nan
#         x_row["q_type"] = np.nan
#     elif pd.isnull(x_row["biweeklyq_timestamp"]):
#         x_row["q_timestamp"] = x_row["monthlyq_timestamp"]
#         x_row["q_type"] = "monthly"
#     else:
#         x_row["q_timestamp"] = x_row["biweeklyq_timestamp"]
#         x_row["q_type"] = "biweekly"
#     if pd.isnull(x_row["phq"]):
#         x_row["phq_lvl"] = np.nan
#     elif x_row["phq"] >= 0 and  x_row["phq"] <= 4:
#         x_row["phq_lvl"] = 0
#     elif x_row["phq"] >= 5 and  x_row["phq"] <= 9:
#         x_row["phq_lvl"] = 1
#     elif x_row["phq"] >= 10 and  x_row["phq"] <= 14:
#         x_row["phq_lvl"] = 2
#     elif x_row["phq"] >= 15 and  x_row["phq"] <= 19:
#         x_row["phq_lvl"] = 3
#     elif x_row["phq"] >= 20 and  x_row["phq"] <= 27:
#         x_row["phq_lvl"] = 4
#     else:
#         x_row["phq_lvl"] = 5# never happens
#     return x_row

# def clean_data(df):
#     df["doe"] = df["doe"].ffill()
#     df = df.apply(clean_phq, axis=1)
#     df = df[~(df["q_timestamp"].isna())]
#     data_phq = df
#     data_other = df[(df["q_type"]=="monthly")]
#     cols_for_id = ["record_id", "redcap_event_name", "doe", "q_timestamp", "q_type"]
#     cols_for_phq = ["phq", "phq_lvl"]
#     cols_for_other = ["msrsr_m", "pss_m", "mfis_m", "psqi_total"]
#     data_phq = data_phq[cols_for_id+cols_for_phq]
#     data_other = data_other[cols_for_id+cols_for_other]
#     return (data_phq, data_other)
