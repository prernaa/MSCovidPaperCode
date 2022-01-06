#!/usr/bin/env python
# coding: utf-8

# In[21]:


## IMPORTS - libraries
import sys
import json
import numpy as np
import pandas as pd
import time
import os
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RandomizedLogisticRegression
from getphasewklbls import *
from sklearn.model_selection import StratifiedKFold

pd.options.display.max_rows = 300

## IMPORTS - To turn off warnings
import warnings
warnings.filterwarnings("ignore")

## timer
startt = time.time()

## constants - important for reproducible results
np.random.seed(0)


## SETTINGS
# ["blue", "calls", "hr", "loc", "scr", "slp", "steps"]
FEAT_FOLDER = "./data/feats_for_multimodal_time_slices/"
# SENSOR_NAME = "loc"  -- if you want to run using main()
# 
# SCENARIO_SUFFIX = "per_phase"
# EXCLD_BASED_ON_SCENARIO_COL = ["lock", "post"] # we only want pre-covid
TOD_SUFFIX_LIST = ["all", "mo", "af", "ev", "ni"]
DOW_SUFFIX_LIST = ["all", "wkdy", "wkend"]


## 15 time slices: 1 suffix per time slice
# weekly/monthly/biweekly/per_phase --> All days of the week X  All times of the day  → 1  (<phase>_all_all)
# per phase --> All days of the week X 4 TODs --> 4 (<phase>_<tod>_all)
# per phase --> 2 Days of the week X All times of the day --> 2 (<phase>_all_<dow>)
# per phase --> 2 Days of the week X 4 TODs --> 8 (<phase>_<tod>_<dow>)


# THRES_PERCENT_NANS_PER_SUBJECT = 0.2 # must be less than this
# THRES_NUM_NANS_PER_COL = 14 # if a column is null in more than these many people, it'll be removed.



DATA_FOLDER = "./data/"
## SURVEY PATH FOR OUTCOMES
SURVEY_PATH = os.path.join(DATA_FOLDER, "QTotalScoresMod.csv")
## MAPPING FILE
PRT_TO_DID_MAPPING_FILE = os.path.join(DATA_FOLDER, "mapping_ids/participant_id_MAP_fitbit_id_MAP_device_id.csv")


# OUTCOME_COL = u'phq9_score_lbl'
# outcome_status = OUTCOME_COL
classesPerOutcome = {
    u'phq9_score_lbl': [0,1],
    u'mfis_m_lbl': [0,1],
    u'psqi_total_lbl': [0,1],
    u'pss_m_lbl': [0,1],
    u'msrsr_m_lbl': [0,1]
}


# ## Model params -- if you want to run using main()
# n_jobs = 1
# selParamsDict = {"C": 0.5, "scaling": 0.5, "sample_fraction": 0.80, "n_resampling": 200, "selection_threshold": 0.20,  "tol": 0.001, "normalize": False, "random_state": 0, "n_jobs": n_jobs}
# modelname = "LOGR"
# suffix_foldername = 1




# In[ ]:


## timer
STARTTIME = time.time()
## GET ARGUMENTS FUNCTION
def getArgs():
    global serverDataPath
    serverDataPath = sys.argv[1]
    global THRES_NUM_NANS_PER_COL
    THRES_NUM_NANS_PER_COL = float(sys.argv[2])
    global THRES_PERCENT_NANS_PER_SUBJECT
    THRES_PERCENT_NANS_PER_SUBJECT = float(sys.argv[3])
    global outcome_status
    outcome_status = sys.argv[4]
    global OUTCOME_COL
    OUTCOME_COL = outcome_status
    global SCENARIO_SUFFIX
    SCENARIO_SUFFIX = sys.argv[5]
    global SCENARIO
    SCENARIO = sys.argv[6]
    global sensorname
    sensorname = sys.argv[7]
    global SENSOR_NAME
    SENSOR_NAME = sensorname
    global modelname
    modelname = sys.argv[8]
    global suffix_foldername
    suffix_foldername = int(sys.argv[9])
    global selParamsDict
    selParamsDict = json.loads(sys.argv[10])
#     global excld_weeks_for_grp
#     excld_weeks_for_grp = json.loads(sys.argv[9])
#     global pre_status
#     pre_status = sys.argv[10]
#     global sen_epochs
#     sen_epochs = sys.argv[11].split("*")
#     global sen_weekdays
#     sen_weekdays = sys.argv[12].split("*")
#     global sen_grps
#     sen_grps = sys.argv[13].split("*")
    print ("ARGUMENTS:")
    print (serverDataPath)
    print (THRES_NUM_NANS_PER_COL)
    print (THRES_PERCENT_NANS_PER_SUBJECT)
    print (outcome_status)
    print (SCENARIO_SUFFIX)
    print (SCENARIO)
    print (sensorname)
    print (modelname)
    print (suffix_foldername)
    print (selParamsDict)
#     print (excld_weeks_for_grp)
#     print (pre_status)
#     print (sen_epochs)
#     print (sen_weekdays)
#     print (sen_grps)
    global processName
    processName = "{0}_{1}_{2}_{3}".format(outcome_status, sensorname, modelname, suffix_foldername)

## GET ARGUMENTS
getArgs()


# In[ ]:


## output folder
serverDataPath = "."
folderpath = serverDataPath+"/models/{0}".format(SENSOR_NAME)+"_{1}_{2}_results{0}_10Fold".format(modelname, outcome_status, SCENARIO)+"_rlog_{0}/"
# folderpath = serverDataPath+"/models/{0}".format(SENSOR_NAME)+"_{1}_results{0}_10Fold".format(modelname, outcome_status)+"_rlog_{0}/"
folderpath = folderpath.format(suffix_foldername)
print ("Created folder {0}".format(folderpath))
if os.path.isdir(folderpath):
    raise Exception("{0} folder already exists".format(folderpath))
else:
    os.makedirs(folderpath)


# In[22]:


## load outcome file
## GET SURVEY DATA AND MAP TO DEVICE_ID, ADD EXTRA COLS
MAPPING_DF = pd.read_csv(PRT_TO_DID_MAPPING_FILE)
MAPPING_DICT = MAPPING_DF.set_index("participant_id")["final_device_id"].to_dict()
FB_MAPPING_DICT = MAPPING_DF.set_index("fitbit_id")["final_device_id"].to_dict()

df_survey = pd.read_csv(SURVEY_PATH)
df_survey["device_id"] = df_survey["record_id"].apply((lambda x: MAPPING_DICT[x]))
df_survey = add_survey_time_cols(df_survey)
df_survey = df_survey.sort_values(by=["device_id", "timestamp_dt"])

## Add PHASE
df_survey =  add_phase_label_wrapper(df_survey, "timestamp_dt")  

mh_cols_base = [u'phq9_score', u'phq12_functioning', u'msrsr_m', u'pss_m', u'mfis_m', u'psqi_total', u'pdds']

df_survey_mean_mh_per_phase = (df_survey.groupby(["device_id", "phase_wrapper"])[mh_cols_base].mean()).reset_index()

## calc phq-9 label
print (df_survey_mean_mh_per_phase[u'phq9_score'].min())
print (df_survey_mean_mh_per_phase[u'phq9_score'].max())
print (df_survey_mean_mh_per_phase[u'phq9_score'].isna().sum())

df_survey_mean_mh_per_phase[u'phq9_score_lbl'] = np.where( df_survey_mean_mh_per_phase[u'phq9_score'] >=5, 1, 0)
df_survey_mean_mh_per_phase[u'mfis_m_lbl'] = np.where( df_survey_mean_mh_per_phase[u'mfis_m'] >=8, 1, 0)
df_survey_mean_mh_per_phase[u'psqi_total_lbl'] = np.where( df_survey_mean_mh_per_phase[u'psqi_total'] >=9, 1, 0)
df_survey_mean_mh_per_phase[u'pss_m_lbl'] = np.where( df_survey_mean_mh_per_phase[u'pss_m'] >=23.5, 1, 0)
df_survey_mean_mh_per_phase[u'msrsr_m_lbl'] = np.where( df_survey_mean_mh_per_phase[u'msrsr_m'] >=6.4, 1, 0)


df_survey_mean_mh_lock = df_survey_mean_mh_per_phase[(df_survey_mean_mh_per_phase["phase_wrapper"]=="lock")]
df_survey_mean_mh_pre = df_survey_mean_mh_per_phase[ (df_survey_mean_mh_per_phase["phase_wrapper"]=="pre")]

# print (df_survey_mean_mh_lock[u'phq9_score_lbl'].value_counts())
# print (df_survey_mean_mh_pre[u'phq9_score_lbl'].value_counts())


# In[23]:


print (df_survey.columns)
# display(df_survey.head(10))
# display(df_survey_mean_mh_lock.head(2))
print (len(df_survey_mean_mh_lock))

# display(df_survey_mean_mh_pre.head(2))
print (len(df_survey_mean_mh_pre))

did_pre_lock = list(set(df_survey_mean_mh_pre["device_id"].to_list()).intersection(set(df_survey_mean_mh_lock["device_id"].to_list())) )
print (len(did_pre_lock))


# In[24]:


scenario_suffix_col_dict = {
    "per_phase": "phase_wrapper",
    #@TODO_LATER: ADD OTHERS LATER
}

## load data matrix where feature names = <sensor>_<time slice suffix>

def clean_loaded_feats_df(df_in, scenario_col):
    df = df_in.copy(deep=True)
    df = df.drop(columns=["Unnamed: 0"])
    df = df.set_index(["device_id", scenario_col])
    return df

def load_matrix(sensor_name, scenario_suffix, scenario, tod_suffix, dow_suffix):
    LoadedDict = {}
    LoadedList = []
    SuffixLists = []
    scenario_col = scenario_suffix_col_dict[scenario_suffix]
    
    cnt = 0
    for tod in tod_suffix:
        for  dow in dow_suffix:
#             print ("{0}_tod_{1}_dow".format(tod, dow))
            if dow=="all" and tod=="all":
                fname = "feats_{0}_{1}.csv".format(sensor_name, scenario_suffix)
                print ("loading from {0}".format(fname))
                feats_df = pd.read_csv(os.path.join(FEAT_FOLDER, fname))
                feats_df = clean_loaded_feats_df(feats_df, scenario_col)
                feats_df.columns = [col+"_all_all" for col in feats_df.columns]
                feats_df_final = feats_df
#                 display(feats_df_final.head(10))
            elif dow=="all" and tod!="all":
                fname = "feats_{0}_{1}_tod.csv".format(sensor_name, scenario_suffix)
                print ("loading from {0}".format(fname))
                feats_df = pd.read_csv(os.path.join(FEAT_FOLDER, fname))
                feats_df = clean_loaded_feats_df(feats_df, scenario_col)
                feats_df_sel = feats_df[(feats_df["tod_lbl"]==tod)]
                feats_df_sel = feats_df_sel.drop(columns=["tod_lbl"])
                feats_df_sel.columns = [col+"_{0}_all".format(tod) for col in feats_df_sel.columns]
                feats_df_final = feats_df_sel
#                 display(feats_df_final.head(10))
            elif dow!="all" and tod=="all":
                fname = "feats_{0}_{1}_dow.csv".format(sensor_name, scenario_suffix)
                print ("loading from {0}".format(fname))
                feats_df = pd.read_csv(os.path.join(FEAT_FOLDER, fname))
                feats_df = clean_loaded_feats_df(feats_df, scenario_col)
                feats_df_sel = feats_df[(feats_df["dow_lbl"]==dow)]
                feats_df_sel = feats_df_sel.drop(columns=["dow_lbl"])
                feats_df_sel.columns = [col+"_all_{0}".format(dow) for col in feats_df_sel.columns]
                feats_df_final = feats_df_sel
#                 display(feats_df_final.head(10))
            else:
                fname = "feats_{0}_{1}_tod_dow.csv".format(sensor_name, scenario_suffix)
                print ("loading from {0}".format(fname))
                feats_df = pd.read_csv(os.path.join(FEAT_FOLDER, fname))
                feats_df = clean_loaded_feats_df(feats_df, scenario_col)
                feats_df_sel = feats_df[( (feats_df["dow_lbl"]==dow) &  (feats_df["tod_lbl"]==tod)) ]
                feats_df_sel = feats_df_sel.drop(columns=["tod_lbl", "dow_lbl"])
                feats_df_sel.columns = [col+"_{0}_{1}".format(tod, dow) for col in feats_df_sel.columns]
                feats_df_final = feats_df_sel
            ## SETTINGS BASED ON SCENARIO
            if SCENARIO == "pre_feats_to_post_lock_mh":
                EXCLD_BASED_ON_SCENARIO_COL = ["lock", "post"]
            ##Exclude data based on scenario col 
                if len(EXCLD_BASED_ON_SCENARIO_COL)>0:
                    feats_df_final = feats_df_final.reset_index()
                    feats_df_final = feats_df_final[~(feats_df_final[scenario_suffix_col_dict[scenario_suffix]].isin(EXCLD_BASED_ON_SCENARIO_COL))]
#                     print (feats_df_final.columns)
            elif SCENARIO == "pre_lock_change_feats_to_lock_mh":
                feats_df_final = feats_df_final.reset_index()
                feats_df_pre = feats_df_final[~(feats_df_final[scenario_suffix_col_dict[scenario_suffix]].isin(["lock", "post"]))]
                feats_df_lock = feats_df_final[~(feats_df_final[scenario_suffix_col_dict[scenario_suffix]].isin(["pre", "post"]))]
                did_intersect = list(set(feats_df_pre["device_id"].to_list()).intersection(set(feats_df_lock["device_id"].to_list())))
                feats_df_pre = feats_df_pre[(feats_df_pre["device_id"].isin(did_intersect))]
                feats_df_lock = feats_df_lock[(feats_df_lock["device_id"].isin(did_intersect))]
                feats_df_pre = feats_df_pre.set_index("device_id")
                feats_df_pre = feats_df_pre.drop(columns=[scenario_col])
                feats_df_lock = feats_df_lock.set_index("device_id")
                feats_df_lock = feats_df_lock.drop(columns=[scenario_col])
                feats_df_final = feats_df_lock.subtract(feats_df_pre, axis="index")
                feats_df_final[scenario_col] = ["lock_minus_pre"]*len(feats_df_final)
                feats_df_pre = feats_df_pre.reset_index()
                feats_df_lock = feats_df_lock.reset_index()
                feats_df_final = feats_df_final.reset_index()
                #print (len(did_intersect))
#                 print (feats_df_pre.columns)
#                 print (feats_df_lock.columns)
#                 print (len(feats_df_pre.columns))
#                 print (len(feats_df_lock.columns))
#                 print (len(feats_df_pre))
#                 print (len(feats_df_lock))
#                 print (feats_df_pre["device_id"].is_unique)
#                 print (feats_df_lock["device_id"].is_unique)
#                 print (feats_df_final.columns)
#                 print (len(feats_df_final))
#                 print (feats_df_final["device_id"].is_unique)
            else:
                raise Exception('SCENARIO {0} not recognized in tochiBasedPipeline.py'.format(SCENARIO)) 
#             display(feats_df_final.head(10))
#             print (feats_df_final["device_id"].is_unique)
#             print (len(feats_df_final))
#             print (len(feats_df_final.columns))
#             print (feats_df_final.columns)
            suffix_list_list = [sensor_name, scenario_suffix, tod, dow]
            suffix_list_str = ",".join(suffix_list_list)
            LoadedList.append(feats_df_final)
#             print ("appending to LoadedList")
#             print (feats_df_final.columns)
            LoadedDict[suffix_list_str] = cnt
            SuffixLists.append(suffix_list_list)
            cnt += 1
    return (LoadedDict, LoadedList, SuffixLists)


## TESTING LOAD MATRIX
# load_matrix(SENSOR_NAME, SCENARIO_SUFFIX, TOD_SUFFIX_LIST, DOW_SUFFIX_LIST)

# ["blue", "calls", "hr", "loc", "scr", "slp", "steps"]

LoadedDict, LoadedList, SuffixLists = load_matrix(SENSOR_NAME, SCENARIO_SUFFIX, SCENARIO, TOD_SUFFIX_LIST, DOW_SUFFIX_LIST)
print ("Loaded {0} dataframes".format(len(LoadedList)))


# load_matrix(SENSOR_NAME, SCENARIO_SUFFIX, ["all"], ["all"])
# load_matrix(SENSOR_NAME, SCENARIO_SUFFIX, ["mo"], ["wkdy"])



# In[25]:


# display(SuffixLists)


# In[26]:


# ## Pre-process all loaded featmats
def getXIdxsForFG(suffix_list_str, fgColNames, cols2matidx):
    if suffix_list_str not in fgColNames:
        return (None)
    fgcols = fgColNames[suffix_list_str]
    fgidxs = []
    fgidxs_names = []
#     print (cols2matidx.keys())
    for c in fgcols:
        if c in cols2matidx.keys():
            i = cols2matidx[c]
            fgidxs.append(i)
            fgidxs_names.append(c)
    return (fgidxs, fgidxs_names)

def removeColsAndSubjectsWithThresNans(featmat, THRES_NUM_NANS_PER_COL, THRES_PERCENT_NANS_PER_SUBJECT):
    before = len(featmat.columns)
    featmat = enforceNansInCols(featmat, THRES_NUM_NANS_PER_COL)
    print (
    "{0} columns remaining from {1} after removing columns nan for >x subjects".format(len(featmat.columns), before))
    ## Remove subjects with more than X% nan values
    before = len(featmat)
    featmat = enforceNansInSubjects(featmat, THRES_PERCENT_NANS_PER_SUBJECT)
    print ("{0} subjects remaining from {1} after removing subjects with >x% nans".format(len(featmat), before))
    return featmat

def enforceNansInSubjects(featmat, THRES_PERCENT_NANS_PER_SUBJECT):
    nans_per_row = featmat.isnull().sum(axis=1)
    # print (nans_per_row)
    # print (nans_per_row.mean())
    # print (nans_per_row.max())
    selected_nans_per_row_index = list(nans_per_row[nans_per_row <= THRES_PERCENT_NANS_PER_SUBJECT * len(featmat.columns)].index.values)
    # print (selected_nans_per_row_index)
    # print (len(selected_nans_per_row_index))
    featmat = featmat[featmat.index.isin(selected_nans_per_row_index)]
    return featmat

def enforceNansInCols(featmat, THRES_NUM_NANS_PER_COL):
    nans_per_col = featmat.isnull().sum(axis=0)
    selected_nans_per_col = list(nans_per_col[nans_per_col <= THRES_NUM_NANS_PER_COL].index.values)
    featmat = featmat[selected_nans_per_col]
    return featmat

def rmAllNansAfterImputation(featmat):
    ## Remove columns with ANY nan values because we could be left with nans even after imputation
    before = len(featmat.columns)
    featmat = featmat.replace([np.inf, -np.inf], 0)
    featmat = featmat.dropna(axis=1, how='any')
    print ("{0} columns remaining from {1} after removing columns with any nans AGAIN".format(len(featmat.columns), before))
    return featmat

def imputeForRemainingNans(featmat):
    for col in featmat.columns:
        col_min = featmat[col].min()
        featmat[col] = featmat[col].fillna(col_min-1)
    print ("imputed remaining nans in X")
    featmat = rmAllNansAfterImputation(featmat)
    return featmat


def concatAndPreprocessFGs(scenario_suffix, suffix_list_sel, LoadedDict, LoadedList):
    scenario_col = scenario_suffix_col_dict[scenario_suffix]
    ## assign a number to each FG for each reference
    fgIdxs = {}
    fgnum = 0
    for s in suffix_list_sel:
        suffix_list_str = ",".join(s)
        fgnum = fgnum + 1
        fgIdxs[suffix_list_str] = fgnum
    ## Concatenation
    fgColNames = {}
    print ("Begin concatenation")
#     print (suffix_list_sel)
    df_list = []
    for s in suffix_list_sel:
        suffix_list_str = ",".join(s)
        df = LoadedList[LoadedDict[suffix_list_str]]
        df = df.set_index(["device_id", scenario_col])
        if len(df) == 0:
            continue
        ## suffix FG number to column names
        fgnum = fgIdxs[suffix_list_str]
        df.columns = ['FG{0}_'.format(fgnum) + str(col) for col in df.columns]
        ## store feature names for each fg
        fgColNames[suffix_list_str] = df.columns
#         print (suffix_list_str)
#         print (df.columns)
#         print (len(df.columns))
        ## append df
        df_list.append(df)
    featmat = pd.concat(df_list, axis=1)
    print ("Final featmat length = {0}. Features = {1}".format(len(featmat), len(featmat.columns)))
    ## Preprocessing - nan handling (same as old)
    featmat = removeColsAndSubjectsWithThresNans(featmat, THRES_NUM_NANS_PER_COL, THRES_PERCENT_NANS_PER_SUBJECT)
    print ("After removeColsAndSubjectsWithThresNans, featmat length = {0}. Features = {1}".format(len(featmat), len(featmat.columns)))
    featmat = imputeForRemainingNans(featmat)
#     print ("Num features after nan removal = {0}".format(len(featmat.columns)))
#     devices_to_keep = featmat.index
#     if pre_status == "LBL":
#         pre_feature = prelbl
#     elif pre_status == "SC":
#         pre_feature = presc
#     if pre_status == "SC" or pre_status == "LBL":
#         if pre_feature is None:
#             raise Exception("Cannot retrieve any column for pre_status = {0}. Please check configOutcomes.py and outcomeFetcher.py.".format(pre_status))
#         pre_feature = pre_feature.loc[devices_to_keep]
#         featmat = pd.concat([pre_feature, featmat], axis=1)
#         print ("Num features after adding pre_status = {0}".format(len(featmat.columns)))
    ## Map column names to index
    cols2matidx = {}
    cols = featmat.columns
    for i in range(0, len(cols)):
        coliname = cols[i]
        cols2matidx[coliname] = i
#     print (fgIdxs.keys())
#     print (fgColNames.keys())
#     print (cols2matidx.keys())
    ## fgColIdxs in featmat.values
    fgColIdxs = {}
    fgColNames_New = {}
    for s in suffix_list_sel:
        suffix_list_str = ",".join(s)
        fgColIdxs[suffix_list_str], fgColNames_New[suffix_list_str] = getXIdxsForFG(suffix_list_str, fgColNames, cols2matidx)
#     print (featmat.columns)
    return (featmat, fgIdxs, fgColNames_New, fgColIdxs)


# In[27]:


## NEED TO CHANGE!
def makeIdxSameForFeatmatAndOutcomePerPhase(featmat, df_outcome, outcome_col, join_idxs = ["device_id"]):
    print ("Before correction, N for outcome = {0}. N for featmat = {1}".format(df_outcome[outcome_col].count(), len(featmat)))
    df_outcome_final = df_outcome[["device_id", outcome_col]]
    df_outcome_final = df_outcome_final[(df_outcome_final[outcome_col].notnull())]
    df_outcome_final = df_outcome_final.set_index("device_id")
    featmat_with_outcome = pd.merge(featmat, df_outcome_final, left_index=True, right_index=True, how='inner')
    outcome = featmat_with_outcome[outcome_col]
    featmat = featmat_with_outcome.drop(outcome_col, axis = 1)
    return (featmat, outcome)


# In[28]:



def get_dataset_pre_feats_lock_phq9(SCENARIO_SUFFIX, SuffixLists, LoadedDict, LoadedList, df_outcome, outcome_col):
    ## Preprocess FeatMat -- pre --> lock (phq9)
#     print ("loaded list cols")
#     print (LoadedList[0].columns)
    featmat, fgIdxs, fgColNames, fgColIdxs = concatAndPreprocessFGs(SCENARIO_SUFFIX, SuffixLists, LoadedDict, LoadedList)
#     display(featmat.head(2))
#     display(df_outcome.head(2))
    featmat = featmat.reset_index().set_index("device_id")
    scenario_suffix_col = scenario_suffix_col_dict[SCENARIO_SUFFIX]
    featmat = featmat.drop(columns=[scenario_suffix_col])    
    ## concat featmat and outcome
    featmat, outcome = makeIdxSameForFeatmatAndOutcomePerPhase(featmat, df_outcome, outcome_col, join_idxs = ["device_id"])
#     display(featmat.head(2))
    # print ("{0}: Final featmat_all {1} subjects, {2} features\n".format(processName, len(featmat), len(featmat.columns)))
    print ("Final featmat_all {0} subjects, {1} features. Outcome {2} subjects\n".format(len(featmat), len(featmat.columns), len(outcome)))
    return (featmat, outcome, fgIdxs, fgColNames, fgColIdxs)



featmat, outcome, fgIdxs, fgColNames, fgColIdxs = get_dataset_pre_feats_lock_phq9(SCENARIO_SUFFIX, SuffixLists, LoadedDict, LoadedList, df_survey_mean_mh_lock, OUTCOME_COL)
print (df_survey_mean_mh_lock.columns)

# print(featmat.isnull().sum(axis = 0))
# print(featmat.min(axis = 0))

# display(featmat.head(10))


# In[29]:


## PIPELINE ML FUNCTIONS
def getTrainTestSplit(featmat, outcome):
    # if TEST_PERSON_DEVICE_IDS not in list(featmat.index):
    #     return (None, None, None, None, 0)
    if len(featmat.columns) == 0:
        return (None, None, None, None, 0)
    # train
    featmat_train = featmat[~featmat.index.isin(TEST_PERSON_DEVICE_IDS)]
    outcome_train = outcome[~outcome.index.isin(TEST_PERSON_DEVICE_IDS)]
    # test
    featmat_test = featmat[featmat.index.isin(TEST_PERSON_DEVICE_IDS)]
    outcome_test = outcome[outcome.index.isin(TEST_PERSON_DEVICE_IDS)]
    return (featmat_train, outcome_train, featmat_test, outcome_test, 1)

def getValueCounts(outcome_train_lbl, outcome_test_lbl):
    outcome_train_lbl_cnts = outcome_train_lbl.value_counts()
    outcome_test_lbl_cnts = outcome_test_lbl.value_counts()
    added_cnts = outcome_train_lbl_cnts.add(outcome_test_lbl_cnts, fill_value=0)
    added_cnts = added_cnts.to_dict()
    return (added_cnts)

def runTest(featmat_train, outcome_train_lbl, featmat_test, outcome_test_lbl, sel, paramsDict, bestmodelnum):
    # print ("runTest run")
    # print (featmat_train.shape)
    # print (outcome_train_lbl.shape)
    # print (featmat_test.shape)
    # print (outcome_test_lbl.shape)
    # print ("Running Test for #{0} ({1})".format(TEST_PERSON_NUM, TEST_PERSON_DEVICE_ID))
    X_train_allfg = featmat_train.values
    Y_train = outcome_train_lbl.values
    featnames_allfg = featmat_train.columns
    X_test_allfg = featmat_test.values
    Y_test = outcome_test_lbl.values
    Y_true = Y_test
    sel_featnames_per_fg = {}
    sel_featnames_list_ordered = []
    sel_X_train = []
    sel_X_test = []
    countNumSel = 0
    fgi = 0
#     print ("runTest b4 loop")
#     print (X_train_allfg.shape)
#     print (Y_train.shape)
#     print (X_test_allfg.shape)
#     print (Y_true.shape)
#     print("printing column name where infinity is present")
#     col_name = featmat_train.columns.to_series()[np.isinf(featmat_train).any()]
#     print(col_name)

    #print (featmat_train.isna().sum())
    for s in SuffixLists:
#         print (s)
        fgi = fgi + 1
        #print fgi,
        suffix_list_str = ",".join(s)
        fgidxs = fgColIdxs[suffix_list_str]
        X_train = X_train_allfg[:, fgidxs]
        X_test = X_test_allfg[:, fgidxs]
        featnames_fg = featnames_allfg[fgidxs]
        # continue if empty
        if X_train.shape[1] == 0:
            continue
        ## scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        # variance thresholding
#         print ("X_train shape before vartransform = {0}".format(X_train.shape))
        vartransform = VarianceThreshold()
        X_train = vartransform.fit_transform(X_train)
        X_test = vartransform.transform(X_test)
        varthres_support = vartransform.get_support()
        featnames_fg = featnames_fg[varthres_support]
#         print ("X_train shape after vartransform = {0}".format(X_train.shape))
        ## feature selection
        if sel == "rlog":
            # print ("IN SEL")
            # print (X_train.shape)
            randomized_rlog = RandomizedLogisticRegression(**paramsDict)
            X_train = randomized_rlog.fit_transform(X_train, Y_train)
            X_test = randomized_rlog.transform(X_test)
            chosen_col_idxs = randomized_rlog.get_support()
#             print (chosen_col_idxs)
            # print (len(featnames_fg))
            if len(chosen_col_idxs) > 0:
                featnames_fg_chosen = list(featnames_fg[chosen_col_idxs])
#                 print (len(featnames_fg_chosen))
                sel_featnames_per_fg[suffix_list_str] = featnames_fg_chosen
                sel_featnames_list_ordered = sel_featnames_list_ordered + featnames_fg_chosen
                sel_X_train.append(X_train)
                sel_X_test.append(X_test)
                countNumSel = countNumSel + len(featnames_fg_chosen)
        else:
            raise ("Unrecognized sel (feature selection algorithm)")
    ## feature selection:  sel{sel{fg1}.....sel{fg45}}
    X_train_concat = np.hstack(sel_X_train)
    X_test_concat = np.hstack(sel_X_test)
    print ("\nSum of number of features selected from all fgs = {0}".format(countNumSel))
    print ("Concatenated X_train has {0} features".format(X_train_concat.shape[1]))
    print ("Concatenated X_test has {0} features".format(X_test_concat.shape[1]))
    if sel == "rlog":
        randomized_rlog = RandomizedLogisticRegression(**paramsDict)
        X_train_concat = randomized_rlog.fit_transform(X_train_concat, Y_train)
        X_test_concat = randomized_rlog.transform(X_test_concat)
        chosen_col_idxs = randomized_rlog.get_support()
        sel_featnames_list_ordered = np.array(sel_featnames_list_ordered)
        chosen_col_idxs = np.array(chosen_col_idxs)
        chosen_cols_final = sel_featnames_list_ordered[chosen_col_idxs]
    else:
        raise ("Unrecognized sel (feature selection algorithm)")
    print ("Final number of features in model = {0}".format(X_train_concat.shape[1]))
    # GBCT
    if modelname == "GBC":
        clf = GradientBoostingClassifier(random_state=0)
    elif modelname == "LOGR":
        clf = LogisticRegression(random_state=0, C=paramsDict["C"], tol=1e-3, penalty="l1", n_jobs=paramsDict["n_jobs"], intercept_scaling=1, class_weight="balanced")
    else:
        raise ("Unrecognized model name")
    clf.fit(X_train_concat, Y_train)
    pred = clf.predict(X_test_concat)
    pred_proba = clf.predict_proba(X_test_concat)
    ## get coeff/ feature importance
    if modelname == "GBC":
        final_feat_importance = clf.feature_importances_.tolist()
    else:
        final_feat_importance = clf.coef_.tolist()[0]
    final_feat_importance = [str(x) for x in final_feat_importance]
    #print (final_feat_importance)
    ## end of coeff/ feature importance code
    Y_pred = pred
    Y_pred_classnames = clf.classes_
    Y_pred_proba_all_classes = pred_proba
    Y_pred_proba = {}
    Y_pred_proba_colnames = []
    # print (Y_pred_proba_all_classes)
    if len(classesPerOutcome[outcome_status]) == 2: # if binary
        Y_pred_proba["Y_pred_proba"] = [x[1] for x in Y_pred_proba_all_classes]
        Y_pred_proba_colnames.append("Y_pred_proba")
    else:
        for ci in range(0, len(Y_pred_classnames)):
            classlbl = Y_pred_classnames[ci]
            classkey = "Y_pred_proba{0}".format(classlbl)
            Y_pred_proba[classkey] = [x[ci] for x in Y_pred_proba_all_classes]
            Y_pred_proba_colnames.append(classkey)
    # print (Y_pred_proba_colnames)
    # print (Y_pred_proba)

    ## Logging file for the fold - logs as many lines as the number of test participants in the fold
    final_feat_importance_str = ",".join(final_feat_importance)
    chosen_cols_final_str = ",".join(chosen_cols_final)
    paramsDict_str = ','.join("%s:%r" % (key, val) for (key, val) in paramsDict.iteritems())
    fgIdxs_str = ','.join("%s:%r" % (key, val) for (key, val) in fgIdxs.iteritems())
    cnts_per_lbl_dict = getValueCounts(outcome_train_lbl, outcome_test_lbl)
    cnts_per_lbl_str = ','.join("%s:%r" % (key, val) for (key, val) in cnts_per_lbl_dict.iteritems())
    dfoutDict = {"did": TEST_PERSON_DEVICE_IDS, "cnts_per_lbl": [cnts_per_lbl_str]*len(TEST_PERSON_DEVICE_IDS), "sel": [sel]*len(TEST_PERSON_DEVICE_IDS), "selParams": [paramsDict_str]*len(TEST_PERSON_DEVICE_IDS), "Y_pred": Y_pred, "Y_true": Y_true, "fgIdxs": [fgIdxs_str]*len(TEST_PERSON_DEVICE_IDS), "sel_final": [chosen_cols_final_str]*len(TEST_PERSON_DEVICE_IDS), "sel_final_importances": [final_feat_importance_str]*len(TEST_PERSON_DEVICE_IDS)}
    dfoutDict.update(Y_pred_proba)
    dfout = pd.DataFrame(dfoutDict)
    dfout = dfout.set_index("did")
    cols = ["cnts_per_lbl", "sel", "selParams", "Y_pred"] + Y_pred_proba_colnames + ["Y_true", "fgIdxs", "sel_final", "sel_final_importances"]
    for s in SuffixLists:
        suffix_list_str = ",".join(s)
        if suffix_list_str in sel_featnames_per_fg:
            sel_feats_fg_str = ",".join(sel_featnames_per_fg[suffix_list_str])
        else:
            sel_feats_fg_str = ""
        dfcol = pd.DataFrame({"did": TEST_PERSON_DEVICE_IDS, "sel_{0}".format(suffix_list_str): [sel_feats_fg_str]*len(TEST_PERSON_DEVICE_IDS)})
        dfcol = dfcol.set_index("did")
        dfout = pd.concat([dfout, dfcol], axis=1)
        cols.append("sel_{0}".format(suffix_list_str))
    dfout.to_csv(folderpath + "{0}_test_model{1}.csv".format("FOLD{0}".format(FOLDNUM), bestmodelnum), columns=cols, header=True)
    print ("{0} minutes elapsed since start of program ".format((time.time() - STARTTIME) / 60.0))
    return (Y_pred, Y_pred_proba)


def checkFeatMat():
    print ("MAIN: Final featmat_all {0} subjects, {1} features\n".format(len(featmat), len(featmat.columns)))
    print ("fgIdxs")
    print (len(fgIdxs.keys()))
    print ("fgColNames")
    print (len(fgColNames.keys()))
    vallensum = 0
    for k in fgColNames.keys():
        vallen = len(fgColNames[k])
        vallensum = vallensum + vallen
    print (vallensum)
    print ("fgColIdxs")
    print (len(fgColIdxs.keys()))
    vallensum = 0
    for k in fgColIdxs.keys():
        vallen = len(fgColNames[k])
        vallensum = vallensum + vallen
    print (vallensum)

# def main():
#     global STARTTIME
#     STARTTIME = time.time()
#     #     checkFeatMat()
#     ## Code runs only for test person
#     global TEST_PERSON_NUM
#     global TEST_PERSON_DEVICE_ID
#     for ti in range(0, len(outcome)):
#         TEST_PERSON_NUM = int(ti)
#         print ("test num arg {0}".format(TEST_PERSON_NUM))
#         DEVICE_IDS_ALL = outcome.index
#         TEST_PERSON_DEVICE_ID = DEVICE_IDS_ALL[TEST_PERSON_NUM]
#         ## Get train and test featmats
#         featmat_train, outcome_train_lbl, featmat_test, outcome_test_lbl, exists = getTrainTestSplit(featmat, outcome)
#         if exists == 1:
#             print ("Train has {0} subjects and {1} features".format(len(featmat_train), len(featmat_train.columns)))
#             print ("Test has {0} subjects and {1} features".format(len(featmat_test), len(featmat_test.columns)))
#             sel = "rlog"
#             print (sel)
#             print (selParamsDict)
#             bestmodelnum = 1
#             Y_pred, Y_pred_proba = runTest(featmat_train, outcome_train_lbl, featmat_test, outcome_test_lbl, sel, selParamsDict, bestmodelnum)

def main():
#     global STARTTIME
#     STARTTIME = time.time()
    ## Code runs for test FOLD
    global FOLDNUM
    global TEST_PERSON_NUMS
    global TEST_PERSON_DEVICE_IDS
    y = list(outcome.values)
    X = range(0, len(y))
    skf = StratifiedKFold(n_splits=5, random_state = 0)
    foldnum = 1
    for train_index, test_index in skf.split(X, y):
        FOLDNUM = foldnum
        print ("RUNNING FOR FOLD {0}".format(FOLDNUM))
        TEST_PERSON_NUMS = test_index
        DEVICE_IDS_ALL = outcome.index
        TEST_PERSON_DEVICE_IDS = DEVICE_IDS_ALL[TEST_PERSON_NUMS]
        featmat_train, outcome_train_lbl, featmat_test, outcome_test_lbl, exists = getTrainTestSplit(featmat, outcome)
        if exists == 1:
            # print ("Train has {0} subjects and {1} features".format(len(featmat_train), len(featmat_train.columns)))
            # print ("Test has {0} subjects and {1} features".format(len(featmat_test), len(featmat_test.columns)))
            sel = "rlog"
            # print (sel)
            # print (selParamsDict)
            bestmodelnum = 1
            Y_pred, Y_pred_proba = runTest(featmat_train, outcome_train_lbl, featmat_test, outcome_test_lbl, sel, selParamsDict, bestmodelnum)
        foldnum = foldnum + 1

# def outputDataset():
#     featcols = list(featmat.columns)
#     outcomecols = list(pd.DataFrame(outcome).columns)
#     featmat_outcome = pd.concat([featmat, outcome], axis = 1)
#     featmat_outcome.to_csv("./dataset/{0}_{1}.csv".format(outcome_status, sensorname), columns = featcols + outcomecols)


# In[30]:


# # main()
if __name__ == '__main__':
    #outputDataset()
    main()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




