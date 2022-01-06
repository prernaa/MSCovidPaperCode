import glob
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, mean_absolute_error, confusion_matrix
# from configParams import getBestConfig
# from configOutcomes import outSuffix, classesPerOutcome
# from outcomeFetcher import getOutcome
# from utils import imputeForRemainingNans
from tochiBasedPipelineResultsHelper import runForFolderDetailed10Fold
from sklearn.model_selection import StratifiedKFold

def rmAllNansAfterImputation(featmat):
    ## Remove columns with ANY nan values because we could be left with nans even after imputation
    before = len(featmat.columns)
#     featmat = featmat.replace([np.inf, -np.inf], 0)
    featmat = featmat.dropna(axis=1, how='any')
    print ("{0} columns remaining from {1} after removing columns with any nans AGAIN".format(len(featmat.columns), before))
    return featmat

def imputeForRemainingNans(featmat):
    for col in featmat.columns:
        col_min = featmat[col].min()
        featmat[col] = featmat[col].fillna(col_min-1)
#     print ("imputed remaining nans in X")
#     featmat = rmAllNansAfterImputation(featmat)
    return featmat


def LoadPredictionsTenFold(sensors_to_combine, outcome_in, scenario_in, classes_this_outcome, best_models, thressensors, serverDataPath):
    df_list_sensors = []
    df_list_outcomes = []
    df_list_fold_indicator_per_sensor = []
    sensorcolnames = []
    foldindicatornames = []

    for sensor in sensors_to_combine:
        sensor_model = best_models[sensor]
#         if not timelimited:
#             limflag = ""
#         else:
#             limflag = limflag
#     folderpath = serverDataPath+"/models/{0}".format(sensorname)+"_{1}_{2}_results{0}_10Fold".format(modelname, outcome_status, scenario)+"_rlog_{0}/"
        sensor_model_folder_path = serverDataPath + "/models/{0}".format(sensor) + "_{1}_{2}_results{0}_10Fold".format(sensor_model[0], outcome_in, scenario_in) + "_rlog_{0}/".format(sensor_model[1])
        print ("Model folder path is {0}".format(sensor_model_folder_path))
        filenames = glob.glob(sensor_model_folder_path+"*_test_model1.csv")
        print (sensor_model_folder_path)
        print ("Found {0} files for {1} sensor".format(len(filenames), sensor))
        testpred_list = []
        testtrue_list = []
        testpredproba_per_lbl = {}
        testpredproba_cols = []
        ## fixing issue encountered for change in levels
        if len(classes_this_outcome) > 2:
            for c in classes_this_outcome:
                classkey = "Y_pred_proba{0}".format(c)
                testpredproba_cols.append(classkey)
                testpredproba_per_lbl[classkey] = []
        did_list_all = []
        did_list_per_fold = {}
        did_list_fold_indicator = []
        for filename in filenames:
            foldnum = filename.split("_")[-3].split("/")[-1]
            result = pd.read_csv(filename, index_col = False)
            result = result.set_index("did")
            did_this_fold = list(result.index)
            did_list_all = did_list_all + did_this_fold
            did_list_per_fold[foldnum] = did_this_fold
            did_list_fold_indicator += [foldnum]*len(did_this_fold)
            if len(classes_this_outcome) == 2:
                if "Y_pred_proba" in testpredproba_per_lbl:
                    testpredproba_per_lbl["Y_pred_proba"]+= list(result["Y_pred_proba"].values)
                else:
                    testpredproba_per_lbl["Y_pred_proba"] = list(result["Y_pred_proba"].values)
                if "Y_pred_proba" not in testpredproba_cols:
                    testpredproba_cols.append("Y_pred_proba")
            else:
                for c in classes_this_outcome:
                    classkey = "Y_pred_proba{0}".format(c)
                    if classkey not in testpredproba_cols:
                        testpredproba_cols.append(classkey)
                    # get result for class
                    if classkey in result:
                        res = list(result[classkey].values)
                    else:
                        res = [0.0]*len(did_this_fold)
                    if classkey in testpredproba_per_lbl:
                        testpredproba_per_lbl[classkey] += res
                    else:
                        testpredproba_per_lbl[classkey] = res
                    print ("{0}: {1}".format(classkey, testpredproba_per_lbl[classkey]))
            test_pred = list(result["Y_pred"].values)
            test_true = list(result["Y_true"].values)
            testpred_list += test_pred
            testtrue_list += test_true
            # print (foldnum)
            # print (len(test_pred))
            # print (len(testpred_list))
        ## Calculating result metrics
        dfRow, dfRow_per_fold, cols, cols_per_fold, mainMetric, mainMetricAscending, N, accuracy, f1meanOrf11, error = runForFolderDetailed10Fold(sensor_model_folder_path, sensor_model[0], sensor_model[1], outcome_in, printtruepreds=False, verbose=True)
        print ("{3} {4} paramDict#{5} \nN = {2}. Accuracy = {0}. Error = {6}. F1 = {1}.".format(accuracy, f1meanOrf11, N, sensor, sensor_model[0], sensor_model[1], error))
        sensorDict = {"device_id": did_list_all, "{0}_test_pred".format(sensor): testpred_list}
        sensorProbaDict = {}
        for c in testpredproba_cols:
            sensorProbaDict["{0}_".format(sensor) + c] = testpredproba_per_lbl[c]
        sensorDict.update(sensorProbaDict)
        dfSensorCols = pd.DataFrame(sensorDict)
        dfSensorCols = dfSensorCols.set_index("device_id")
        sensorcolnames = sensorcolnames + ["{0}_{1}".format(sensor, x) for x in testpredproba_per_lbl] + ["{0}_test_pred".format(sensor)]
        dfOutcomeCols = pd.DataFrame({"device_id": did_list_all, "{0}_true".format(sensor): testtrue_list})
        dfOutcomeCols = dfOutcomeCols.set_index("device_id")
        # print ("Fold indicator")
        # print (len(did_list_all))
        # print (len(did_list_fold_indicator))
        # print (did_list_all)
        # print (did_list_fold_indicator)
        dfFoldIndicator = pd.DataFrame({"device_id": did_list_all, "{0}_fold".format(sensor): did_list_fold_indicator})
        dfFoldIndicator = dfFoldIndicator.set_index("device_id")
        df_list_sensors.append(dfSensorCols)
        df_list_outcomes.append(dfOutcomeCols)
        df_list_fold_indicator_per_sensor.append(dfFoldIndicator)

    ## for fold indicator
    df_list_fold_indicator_all = pd.concat(df_list_fold_indicator_per_sensor, axis=1)
    ## for outcomes
    df_all_outcomes = pd.concat(df_list_outcomes, axis=1)
    did_true_final = []
    for index, row in df_all_outcomes.iterrows():
        for c in df_all_outcomes.columns:
            if row[c] not in [None, np.nan]:
                did_true_final.append(row[c])
                break
    df_all_outcomes["true_final"] = did_true_final
    # print (df_all_outcomes)
    # print (df_all_outcomes["true_final"])

    ## for test proba and testpred
    df_list_sensors.append(pd.DataFrame(df_all_outcomes["true_final"]))
    df_all = pd.concat(df_list_sensors, axis=1)
    df_all["cntNotNull"] = df_all.apply(lambda x: x.count(), axis=1)
    # print (df_all.columns)
    # print (len(df_all.columns))

    # printAllRows(df_all["cntNotNull"])
    # printAllRows(df_all)
    # print ("before imp")
    # print (df_all.columns)
    # print (df_all)
    df_all = imputeForRemainingNans(df_all)
    # print ("after imp")
    # print (df_all.columns)
    # print (df_all)

    # print (df_all)
    # print (len(df_all))
    # print (df_all.columns)

    ## sensor column names
    # print (sensorcolnames)
    # print (len(sensorcolnames))

    ## remove rows based on certain condition
    # thres_sensors = len(sensors_to_combine)*0.6
    thres_sensors = thressensors
    if len(classes_this_outcome) == 2:
        thres = (1 + 1) * thres_sensors
    else:
        thres = (1 + len(
            classes_this_outcome)) * thres_sensors  # (1 for testpred + numclasses for proba of each class) we use > and not >= below coz we also have true label
    print ("threshold = {0}".format(thres))
    # print ("Before missing {0}".format(df_all.columns))
    df_all = df_all[df_all.cntNotNull > thres]
    # print ("After missing {0}".format(df_all.columns))
    df_all_sensors = df_all[sensorcolnames]
    print ("{0} people left, {1} columns".format(len(df_all_sensors), len(df_all_sensors.columns)))
    df_all_post = df_all["true_final"]
    return (df_all, df_all_sensors, df_all_post, df_list_fold_indicator_all, thres)


# ## Loading predictions
# def LoadPredictions(sensors_to_combine, outcome_in, classes_this_outcome, best_models, thressensors, serverDataPath, timelimited = False, limflag = None):
#     df_list_sensors = []
#     df_list_outcomes = []
#     sensorcolnames = []
#     for sensor in sensors_to_combine:
#         sensor_model = best_models[sensor]
#         if not timelimited:
#             limflag = ""
#         else:
#             limflag = limflag
#         sensor_model_folder_path = serverDataPath + "/models/{0}".format(sensor) + "_{3}_results{0}{2}prestatus{1}".format(limflag, "NO", sensor_model[0], outSuffix[outcome_in]) + "_rlog_{0}/".format(sensor_model[1])
#         print ("Model folder path is {0}".format(sensor_model_folder_path))
#         filenames = glob.glob(sensor_model_folder_path+"*_test_model1.csv")
#         print (sensor_model_folder_path)
#         print ("Found {0} files for {1} sensor".format(len(filenames), sensor))
#         testpred_list = []
#         testtrue_list = []
#         testpredproba_per_lbl = {}
#         testpredproba_cols = []
#         did_list = []

#         for filename in filenames:
#             did = filename.split("_")[-3].split("/")[-1]
#             result = pd.read_csv(filename)
#             result = result.iloc[0]
#             #print ("DID {0} got pred proba {1}".format(did, result["Y_pred_proba"]))
#             if len(classes_this_outcome) == 2:
#                 if "Y_pred_proba" in testpredproba_per_lbl:
#                     #print ("before appending")
#                     #print (testpredproba_per_lbl["Y_pred_proba"])
#                     testpredproba_per_lbl["Y_pred_proba"].append(result["Y_pred_proba"])
#                     #print ("after appending")
#                     #print (testpredproba_per_lbl["Y_pred_proba"])
#                 else:
#                     #print ("initial")
#                     testpredproba_per_lbl["Y_pred_proba"] = [result["Y_pred_proba"]]
#                     #print (testpredproba_per_lbl["Y_pred_proba"])
#                 if "Y_pred_proba" not in testpredproba_cols:
#                     testpredproba_cols.append("Y_pred_proba")
#             else:
#                 for c in classes_this_outcome:
#                     classkey = "Y_pred_proba{0}".format(c)
#                     if classkey not in testpredproba_cols:
#                         testpredproba_cols.append(classkey)
#                     # get result for class
#                     if classkey in result:
#                         res = result[classkey]
#                     else:
#                         res = 0.0
#                     #print (classkey)
#                     # append result to dict list - each element of the list is for each person
#                     if classkey in testpredproba_per_lbl:
#                         #print (testpredproba_per_lbl[classkey])
#                         testpredproba_per_lbl[classkey] = testpredproba_per_lbl[classkey]+[res]
#                     else:
#                         #print ("new key")
#                         testpredproba_per_lbl[classkey] = [res]
#             test_pred = result["Y_pred"]
#             test_true = result["Y_true"]
#             did_list.append(did)
#             testpred_list.append(test_pred)
#             testtrue_list.append(test_true)
#         testpred_list = np.array(testpred_list)
#         testtrue_list = np.array(testtrue_list)
#         numpple = len(did_list)
#         accuracy = accuracy_score(testtrue_list, testpred_list)
#         if classes_this_outcome > 2:
#             precision = precision_score(testtrue_list, testpred_list, average="macro")
#             recall = recall_score(testtrue_list, testpred_list, average="macro")
#             f1 = f1_score(testtrue_list, testpred_list, average="macro")
#             if numpple == 0:
#                 mae = np.nan
#             else:
#                 mae = mean_absolute_error(testtrue_list, testpred_list)
#             print ("{5} {6} paramDict#{7} \nN = {4}. Accuracy = {0}. Error = {8}. Precision = {1}. Recall = {2}. F1 = {3}.".format(accuracy, precision, recall, f1, numpple, sensor, sensor_model[0], sensor_model[1], mae))
#         else:
#             precision = precision_score(testtrue_list, testpred_list, average = "binary")
#             recall = recall_score(testtrue_list, testpred_list, average = "binary")
#             f1 = f1_score(testtrue_list, testpred_list, average = "binary")
#             print ("{5} {6} paramDict#{7} \nN = {4}. Accuracy = {0}. Precision = {1}. Recall = {2}. F1 = {3}.".format(accuracy, precision, recall, f1, numpple, sensor, sensor_model[0], sensor_model[1]))
#         sensorDict = {"device_id" : did_list, "{0}_test_pred".format(sensor) : testpred_list}
#         sensorProbaDict = {}
#         for c in testpredproba_cols:
#             sensorProbaDict["{0}_".format(sensor)+c] = testpredproba_per_lbl[c]
#         sensorDict.update(sensorProbaDict)
#         dfSensorCols = pd.DataFrame(sensorDict)
#         dfSensorCols = dfSensorCols.set_index("device_id")
#         sensorcolnames = sensorcolnames + ["{0}_{1}".format(sensor,x) for x in testpredproba_per_lbl] + ["{0}_test_pred".format(sensor)]
#         dfOutcomeCols = pd.DataFrame({"device_id" : did_list, "{0}_true".format(sensor) : testtrue_list})
#         dfOutcomeCols = dfOutcomeCols.set_index("device_id")
#         df_list_sensors.append(dfSensorCols)
#         df_list_outcomes.append(dfOutcomeCols)

#     ## for outcomes
#     df_all_outcomes = pd.concat(df_list_outcomes, axis = 1)
#     did_true_final = []
#     for index, row in df_all_outcomes.iterrows():
#         for c in df_all_outcomes.columns:
#             if row[c] not in [None, np.nan]:
#                 did_true_final.append(row[c])
#                 break
#     df_all_outcomes["true_final"] = did_true_final
#     #print (df_all_outcomes)
#     #print (df_all_outcomes["true_final"])

#     ## for test proba and testpred
#     df_list_sensors.append(pd.DataFrame(df_all_outcomes["true_final"]))
#     df_all = pd.concat(df_list_sensors, axis=1)
#     df_all["cntNotNull"] = df_all.apply(lambda x: x.count(), axis=1)
#     #print (df_all.columns)
#     #print (len(df_all.columns))

#     # printAllRows(df_all["cntNotNull"])
#     # printAllRows(df_all)
#     # print ("before imp")
#     # print (df_all.columns)
#     # print (df_all)
#     df_all = imputeForRemainingNans(df_all, -1)
#     # print ("after imp")
#     # print (df_all.columns)
#     # print (df_all)

#     # print (df_all)
#     # print (len(df_all))
#     # print (df_all.columns)

#     ## sensor column names
#     #print (sensorcolnames)
#     #print (len(sensorcolnames))

#     ## remove rows based on certain condition
#     #thres_sensors = len(sensors_to_combine)*0.6
#     thres_sensors = thressensors
#     if len(classes_this_outcome) == 2:
#         thres = (1+1) * thres_sensors
#     else:
#         thres = (1+len(classes_this_outcome))*thres_sensors # (1 for testpred + numclasses for proba of each class) we use > and not >= below coz we also have true label
#     print ("threshold = {0}".format(thres))
#     # print ("Before missing {0}".format(df_all.columns))
#     df_all = df_all[df_all.cntNotNull > thres]
#     # print ("After missing {0}".format(df_all.columns))
#     df_all_sensors = df_all[sensorcolnames]
#     print ("{0} people left, {1} columns".format(len(df_all_sensors), len(df_all_sensors.columns)))
#     df_all_post = df_all["true_final"]
#     return (df_all, df_all_sensors, df_all_post, thres)


# def LoadBestModelsDict(sensors_to_combine, outcome):
#     best_models = {}
#     for sensorname in sensors_to_combine:
#         modelname, selParamsDict, suffix_foldername = getBestConfig(sensorname, outcome, 1)
#         best_models[sensorname] = [modelname, suffix_foldername]
#     return best_models

## To work with automatic pipeline
def LoadBestModelsDictAutomatePipeline(sensors_to_combine, outcome, classes_this_outcome, serverDataPath, combineFolder, combineFolderFileSuffix, tenFolds = True):
    best_models = {}
    for sensorname in sensors_to_combine:
        if not tenFolds:
            print ("LOO not supported yet")
#             fname = os.path.join(serverDataPath, combineFolder, "{0}_{1}_{2}_models_{3}.csv".format(outcome, sensorname, limflag, combineFolderFileSuffix))
        else:
            limflag = ""
            print (serverDataPath)
            print (combineFolder)
            print ("outcome: {0}, sensorname: {1}, limflag: {2}, combineFolderFileSuffix: {3}".format(outcome, sensorname, limflag, combineFolderFileSuffix))
            fname = os.path.join(serverDataPath, combineFolder, "{0}_{1}_{2}_models_{3}_10Fold.csv".format(outcome, sensorname, limflag, combineFolderFileSuffix))
        df = pd.read_csv(fname)
        print ("in file")
        print (fname)
        print (df.columns)
        Nmax = df["N"].max()
        df = df[df["N"] >= (Nmax-3)]
        #print (df.columns)
        if len(classes_this_outcome) == 2:
#             try:
            print ("Opening {0} failed".format(fname))
            df = df.sort_values(by=['accuracy', 'f11'], ascending=[False, False])
#             except:
#                 print ("Opening {0} failed".format(fname))
        else:
            df = df.sort_values(by=['error', 'accuracy'], ascending=[True, False])
            if tenFolds:
                df = df.sort_values(by=['accuracy', 'f1allmean'], ascending=[False, False])
        best_modelname = df.iloc[0]['modeltype']
        best_modelsuffix = df.iloc[0]['modeli']
        best_models[sensorname] = [best_modelname, best_modelsuffix]
    return best_models


# ## LOO
# def check_pre_status_input_status(pre_status):
#     if pre_status == "LBL" or pre_status == "SC" or pre_status == "NO":
#         return 1
#     else:
#         return 0

def runTenFolds(df_all_sensors, df_all_post, thres, outcome_in, classes_this_outcome, sensors_to_combine, best_models, paramsDict, outbase, outpath):
#     if check_pre_status_input_status(pre_status) == 0:
#         raise Exception("unrecognized pre_status {0}".format(pre_status))
#     ## get pre col
#     outcome, presc, prelbl = getOutcome(outbase, outcome_in, tenFolds = True)
    ## LOO start
    featcols = []
    for col in df_all_sensors.columns:
        if "Y_pred_proba" in col:
            featcols.append(col)
    df_all_sensors_proba = df_all_sensors[featcols]
    # print (df_all_sensors_proba)
    dids_all = df_all_sensors_proba.index
    devices_to_keep = dids_all
    print ("Num features = {0}".format(len(df_all_sensors_proba.columns)))
#     if pre_status == "LBL":
#         pre_feature = prelbl
#     elif pre_status == "SC":
#         pre_feature = presc
#     if pre_status == "SC" or pre_status == "LBL":
#         if pre_feature is None:
#             raise Exception(
#                 "Cannot retrieve any column for pre_status = {0}. Please check configOutcomes.py and outcomeFetcher.py.".format(
#                     pre_status))
#         pre_feature = pre_feature.loc[devices_to_keep]
#         df_all_sensors_proba = pd.concat([pre_feature, df_all_sensors_proba], axis=1)
#         print ("Num features after adding pre_status = {0}".format(len(df_all_sensors_proba.columns)))
    X = df_all_sensors_proba.values
    Y = df_all_post.values
    print(df_all_post.value_counts())
    Xfeatnames = df_all_sensors_proba.columns
    # leave-one-out loop
#     loo = LeaveOneOut()
#     loo.get_n_splits(X)
    cvnum = 0
    dfout = pd.DataFrame()
    Y_true = []
    Y_pred = []
    adaboost = paramsDict["adaboost"]
    scaling = paramsDict["scaling"]
    base_estimator = paramsDict["base_estimator"]
    n_estimators = paramsDict["n_estimators"]
    n_estimators_gbc = paramsDict["n_estimators_gbc"]
    learning_rate = paramsDict["learning_rate"]
    learning_rate_gbc = paramsDict["learning_rate_gbc"]
    ## 10-fold cv
    numpple = 0
    accuracy_sum = 0
    mcc_sum = 0
    prf_results_dict_sum = {}
    mae_sum = 0
    prf_cols_list = []
    skf = StratifiedKFold(n_splits=10, random_state=0)
    for train_index, test_index in skf.split(X, Y):
        cvnum = cvnum + 1
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        did_test = dids_all[test_index]
        #         print (Y_test)
        if ((cvnum == 1) or ((cvnum % 5) == 0)):
            print ("fold #{0}".format(cvnum))
        #         ## scaling
        if scaling:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        if adaboost:
            clf = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=n_estimators, learning_rate=learning_rate, random_state=0)
        else:
            clf = GradientBoostingClassifier(random_state=0, n_estimators=n_estimators_gbc, learning_rate=learning_rate_gbc)
        clf.fit(X_train, Y_train)
        pred = clf.predict(X_test)
        # pred_proba = clf.predict_proba(X_test)
        # pred_proba_allclasses = pred_proba[0]
        # pred_classnames = clf.classes_
        if adaboost:
            importances = clf.feature_importances_
            dfimp = pd.DataFrame({"features": Xfeatnames, "scores": importances})
        #             print (dfimp)
        Y_pred += list(np.array(pred))
        Y_true += list(np.array(Y_test))
        numpple_fold = X_test.shape[0]
        numpple += numpple_fold
        accuracy = accuracy_score(Y_test, pred)
        accuracy_sum += accuracy
        mcc = matthews_corrcoef(Y_test, pred)
        mcc_sum += mcc
        prf_results_dict = {}
        prf_cols_list = [] # reinitialize
        if len(classes_this_outcome) > 2:
            for c in classes_this_outcome:
                prf_results_dict["precision{0}".format(c)] = [precision_score(Y_test, pred, pos_label=c, labels=[c], average='macro')]
                prf_results_dict["recall{0}".format(c)] = [recall_score(Y_test, pred, pos_label=c, labels=[c], average='macro')]
                prf_results_dict["f1{0}".format(c)] = [f1_score(Y_test, pred, pos_label=c, labels=[c], average='macro')]
                prf_cols_list = prf_cols_list + ["precision{0}".format(c), "recall{0}".format(c), "f1{0}".format(c)]
                ## summing for avg
                if "precision{0}".format(c) not in prf_results_dict_sum:
                    prf_results_dict_sum["precision{0}".format(c)] = [0]
                if "recall{0}".format(c) not in prf_results_dict_sum:
                    prf_results_dict_sum["recall{0}".format(c)] = [0]
                if "f1{0}".format(c) not in prf_results_dict_sum:
                    prf_results_dict_sum["f1{0}".format(c)] = [0]
                prf_results_dict_sum["precision{0}".format(c)] = [prf_results_dict_sum["precision{0}".format(c)][0] + prf_results_dict["precision{0}".format(c)][0]]
                prf_results_dict_sum["recall{0}".format(c)] = [prf_results_dict_sum["recall{0}".format(c)][0] + prf_results_dict["recall{0}".format(c)][0]]
                prf_results_dict_sum["f1{0}".format(c)] = [prf_results_dict_sum["f1{0}".format(c)][0] + prf_results_dict["f1{0}".format(c)][0]]
            if numpple == 0:
                mae = np.nan
            else:
                mae = mean_absolute_error(Y_test, pred)
                mae_sum += mae
            # print ("N = {4}. Accuracy = {0}. Error = {5}. Precision = {1}. Recall = {2}. F1 = {3}.".format(accuracy, precision, recall, f1, numpple, mae))
        else:
            for c in classes_this_outcome:
                prf_results_dict["precision{0}".format(c)] = [precision_score(Y_test, pred, pos_label=c, average='binary')]
                prf_results_dict["recall{0}".format(c)] = [recall_score(Y_test, pred, pos_label=c, average='binary')]
                prf_results_dict["f1{0}".format(c)] = [f1_score(Y_test, pred, pos_label=c, average='binary')]
                prf_cols_list = prf_cols_list + ["precision{0}".format(c), "recall{0}".format(c), "f1{0}".format(c)]
                ## summing for avg
                if "precision{0}".format(c) not in prf_results_dict_sum:
                    prf_results_dict_sum["precision{0}".format(c)] = [0]
                if "recall{0}".format(c) not in prf_results_dict_sum:
                    prf_results_dict_sum["recall{0}".format(c)] = [0]
                if "f1{0}".format(c) not in prf_results_dict_sum:
                    prf_results_dict_sum["f1{0}".format(c)] = [0]
                prf_results_dict_sum["precision{0}".format(c)] = [prf_results_dict_sum["precision{0}".format(c)][0] + prf_results_dict["precision{0}".format(c)][0]]
                prf_results_dict_sum["recall{0}".format(c)] = [prf_results_dict_sum["recall{0}".format(c)][0] + prf_results_dict["recall{0}".format(c)][0]]
                prf_results_dict_sum["f1{0}".format(c)] = [prf_results_dict_sum["f1{0}".format(c)][0] + prf_results_dict["f1{0}".format(c)][0]]
            # print ("N = {4}. Accuracy = {0}. Precision = {1}. Recall = {2}. F1 = {3}.".format(accuracy, precision, recall, f1, numpple))
        ## @TODO - output per-fold results too!
    accuracy_avg = accuracy_sum/float(cvnum)
    mcc_avg = mcc_sum/float(cvnum)
    mae_avg = mae_sum/float(cvnum)
    prf_results_dict_avg = {}
    for col in prf_cols_list:
        prf_results_dict_avg[col] = [prf_results_dict_sum[col][0]/float(cvnum)]
    print ("COMBI RESULTS")
    print (prf_results_dict_sum)
    print (prf_results_dict_avg)
    # logging - append to a single file
    # thres, scaling, base_estimator, n_estimators, learning_rate, sensors_to_combine, best_models
    # N, accuracy, precision, recall, f1 for each column label
    sensors_to_combine_str = ", ".join(sensors_to_combine)
    if base_estimator is None:
        base_estimator_str = "None"
    else:
        base_estimator_str = "GBC"
    best_models_str = ','.join("%s:%r" % (key, val) for (key, val) in best_models.iteritems())
    if adaboost is False:
        base_estimator_str = "GBC_NoAdaboost"
        n_estimators = n_estimators_gbc
        learning_rate = learning_rate_gbc

    if len(classes_this_outcome) > 2:
        cols = ["missing_data_thres", "scaling", "adaboost", "base_estimator", "n_estimators", "learning_rate", "n_estimators_gbc", "learning_rate_gbc", "sensors_combined", "num_sensors", "best_models", "N", "accuracy", "error", "mcc"]
        datadict = {"missing_data_thres": [thres], "scaling": [scaling], "adaboost": [adaboost], "base_estimator": [base_estimator_str], "n_estimators": [n_estimators], "learning_rate": [learning_rate], "n_estimators_gbc": [n_estimators_gbc], "learning_rate_gbc": [learning_rate_gbc], "sensors_combined": [sensors_to_combine_str], "num_sensors": [len(sensors_to_combine)], "best_models": [best_models_str], "N": [numpple], "accuracy": [accuracy_avg], "error": [mae_avg], "mcc": [mcc_avg]}
        cols = cols + prf_cols_list
        datadict.update(prf_results_dict_avg)
        dfRow = pd.DataFrame(datadict)
    else:
        cols = ["missing_data_thres", "scaling", "adaboost", "base_estimator", "n_estimators", "learning_rate", "n_estimators_gbc", "learning_rate_gbc", "sensors_combined", "num_sensors", "best_models", "N", "accuracy", "mcc"]
        datadict = {"missing_data_thres": [thres], "scaling": [scaling], "adaboost": [adaboost], "base_estimator": [base_estimator_str], "n_estimators": [n_estimators], "learning_rate": [learning_rate], "n_estimators_gbc": [n_estimators_gbc], "learning_rate_gbc": [learning_rate_gbc], "sensors_combined": [sensors_to_combine_str], "num_sensors": [len(sensors_to_combine)], "best_models": [best_models_str], "N": [numpple], "accuracy": [accuracy_avg], "mcc": [mcc_avg]}
        cols = cols + prf_cols_list
        datadict.update(prf_results_dict_avg)
        dfRow = pd.DataFrame(datadict)
    if os.path.exists(outpath):
        dfRow.to_csv(outpath, mode='a', columns=cols, header=False)
    else:
        dfRow.to_csv(outpath, mode='a', columns=cols, header=True)
    confusionmat = confusion_matrix(Y_true, Y_pred)
    print (confusionmat)
    return (df_all_post.index, Y_pred, Y_true)


# def runLOO(df_all_sensors, df_all_post, thres, pre_status, outcome_in, classes_this_outcome, sensors_to_combine, best_models, paramsDict, outbase, outpath):
#     if check_pre_status_input_status(pre_status) == 0:
#         raise Exception("unrecognized pre_status {0}".format(pre_status))
#     ## get pre col
#     outcome, presc, prelbl = getOutcome(outbase, outcome_in)
#     ## LOO start
#     featcols = []
#     for col in df_all_sensors.columns:
#         if "Y_pred_proba" in col:
#             featcols.append(col)
#     df_all_sensors_proba = df_all_sensors[featcols]
#     #print (df_all_sensors_proba)
#     dids_all = df_all_sensors_proba.index
#     devices_to_keep = dids_all
#     print ("Num features = {0}".format(len(df_all_sensors_proba.columns)))
#     if pre_status == "LBL":
#         pre_feature = prelbl
#     elif pre_status == "SC":
#         pre_feature = presc
#     if pre_status == "SC" or pre_status == "LBL":
#         if pre_feature is None:
#             raise Exception("Cannot retrieve any column for pre_status = {0}. Please check configOutcomes.py and outcomeFetcher.py.".format(pre_status))
#         pre_feature = pre_feature.loc[devices_to_keep]
#         df_all_sensors_proba = pd.concat([pre_feature, df_all_sensors_proba], axis=1)
#         print ("Num features after adding pre_status = {0}".format(len(df_all_sensors_proba.columns)))
#     X = df_all_sensors_proba.values
#     Y = df_all_post.values
#     print(df_all_post.value_counts())
#     Xfeatnames = df_all_sensors_proba.columns
#     # leave-one-out loop
#     loo = LeaveOneOut()
#     loo.get_n_splits(X)
#     cvnum = 0
#     dfout = pd.DataFrame()
#     Y_true = []
#     Y_pred = []
#     adaboost= paramsDict["adaboost"]
#     scaling = paramsDict["scaling"]
#     base_estimator = paramsDict["base_estimator"]
#     n_estimators = paramsDict["n_estimators"]
#     n_estimators_gbc = paramsDict["n_estimators_gbc"]
#     learning_rate = paramsDict["learning_rate"]
#     learning_rate_gbc = paramsDict["learning_rate_gbc"]
#     for train_index, test_index in loo.split(X):
#         cvnum = cvnum + 1
#         X_train, X_test = X[train_index], X[test_index]
#         Y_train, Y_test = Y[train_index], Y[test_index]
#         did_test = dids_all[test_index]
# #         print (Y_test)
#         if ((cvnum == 1) or ((cvnum % 10) == 0)):
#             print ("fold #{0}".format(cvnum))
# #         ## scaling
#         if scaling:
#             scaler = StandardScaler()
#             X_train = scaler.fit_transform(X_train)
#             X_test = scaler.transform(X_test)
#         if adaboost:
#             clf = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=n_estimators, learning_rate=learning_rate, random_state=0)
#         else:
#             clf = GradientBoostingClassifier(random_state=0, n_estimators = n_estimators_gbc, learning_rate = learning_rate_gbc)
#         clf.fit(X_train, Y_train)
#         pred = clf.predict(X_test)
#         # pred_proba = clf.predict_proba(X_test)
#         # pred_proba_allclasses = pred_proba[0]
#         # pred_classnames = clf.classes_
#         if adaboost:
#             importances = clf.feature_importances_
#             dfimp = pd.DataFrame({"features": Xfeatnames, "scores": importances})
# #             print (dfimp)
#         Y_pred.append(pred[0])
#         Y_true.append(Y_test[0])
#     Y_true = np.array(Y_true)
#     Y_pred = np.array(Y_pred)
#     numpple = len(df_all_sensors_proba)
#     accuracy = accuracy_score(Y_true, Y_pred)
#     mcc = matthews_corrcoef(Y_true, Y_pred)
#     prf_results_dict = {}
#     prf_cols_list = []
#     if len(classes_this_outcome) >2:
#         for c in classes_this_outcome:
#             prf_results_dict["precision{0}".format(c)] = [precision_score(Y_true, Y_pred, pos_label=c, labels = [c], average='macro')]
#             prf_results_dict["recall{0}".format(c)] = [recall_score(Y_true, Y_pred, pos_label=c, labels = [c], average='macro')]
#             prf_results_dict["f1{0}".format(c)] = [f1_score(Y_true, Y_pred, pos_label=c, labels = [c], average='macro')]
#             prf_cols_list = prf_cols_list + ["precision{0}".format(c), "recall{0}".format(c), "f1{0}".format(c)]
#         if numpple == 0:
#             mae = np.nan
#         else:
#             mae = mean_absolute_error(Y_true, Y_pred)
#         # print ("N = {4}. Accuracy = {0}. Error = {5}. Precision = {1}. Recall = {2}. F1 = {3}.".format(accuracy, precision, recall, f1, numpple, mae))
#     else:
#         for c in classes_this_outcome:
#             prf_results_dict["precision{0}".format(c)] = [precision_score(Y_true, Y_pred, pos_label=c, average='binary')]
#             prf_results_dict["recall{0}".format(c)] = [recall_score(Y_true, Y_pred, pos_label=c, average='binary')]
#             prf_results_dict["f1{0}".format(c)] = [f1_score(Y_true, Y_pred, pos_label=c, average='binary')]
#             prf_cols_list = prf_cols_list + ["precision{0}".format(c), "recall{0}".format(c), "f1{0}".format(c)]
#         # print ("N = {4}. Accuracy = {0}. Precision = {1}. Recall = {2}. F1 = {3}.".format(accuracy, precision, recall, f1, numpple))
#     # logging - append to a single file
#     # thres, scaling, base_estimator, n_estimators, learning_rate, sensors_to_combine, best_models
#     # N, accuracy, precision, recall, f1 for each column label
#     sensors_to_combine_str = ", ".join(sensors_to_combine)
#     if base_estimator is None:
#         base_estimator_str = "None"
#     else:
#         base_estimator_str = "GBC"
#     best_models_str = ','.join("%s:%r" % (key,val) for (key,val) in best_models.iteritems())
#     if adaboost is False:
#         base_estimator_str = "GBC_NoAdaboost"
#         n_estimators = n_estimators_gbc
#         learning_rate = learning_rate_gbc

#     if len(classes_this_outcome) > 2:
#         cols = ["missing_data_thres", "pre_status", "scaling", "adaboost", "base_estimator", "n_estimators", "learning_rate", "n_estimators_gbc", "learning_rate_gbc", "sensors_combined", "num_sensors", "best_models", "N", "accuracy", "error", "mcc"]
#         datadict = {"missing_data_thres": [thres], "pre_status": [pre_status], "scaling": [scaling], "adaboost": [adaboost], "base_estimator": [base_estimator_str], "n_estimators": [n_estimators], "learning_rate": [learning_rate], "n_estimators_gbc": [n_estimators_gbc], "learning_rate_gbc": [learning_rate_gbc], "sensors_combined": [sensors_to_combine_str], "num_sensors": [len(sensors_to_combine)],"best_models": [best_models_str], "N": [numpple], "accuracy": [accuracy], "error": [mae], "mcc": [mcc]}
#         cols = cols + prf_cols_list
#         datadict.update(prf_results_dict)
#         dfRow = pd.DataFrame(datadict)
#     else:
#         cols = ["missing_data_thres", "pre_status", "scaling", "adaboost", "base_estimator", "n_estimators", "learning_rate", "n_estimators_gbc", "learning_rate_gbc", "sensors_combined", "num_sensors", "best_models", "N", "accuracy", "mcc"]
#         datadict = {"missing_data_thres": [thres], "pre_status": [pre_status], "scaling": [scaling], "adaboost": [adaboost], "base_estimator": [base_estimator_str], "n_estimators": [n_estimators], "learning_rate": [learning_rate], "n_estimators_gbc": [n_estimators_gbc], "learning_rate_gbc": [learning_rate_gbc], "sensors_combined": [sensors_to_combine_str], "num_sensors": [len(sensors_to_combine)],"best_models": [best_models_str], "N": [numpple], "accuracy": [accuracy], "mcc": [mcc]}
#         cols = cols + prf_cols_list
#         datadict.update(prf_results_dict)
#         dfRow = pd.DataFrame(datadict)
#     if os.path.exists(outpath):
#         dfRow.to_csv(outpath, mode='a', columns = cols, header=False)
#     else:
#         dfRow.to_csv(outpath, mode='a', columns = cols, header=True)
#     confusionmat = confusion_matrix(Y_true, Y_pred)
#     print (confusionmat)
#     return (df_all_post.index, Y_pred, Y_true)

def combineSensors(sensors_to_combine, thressensors_in, outcome_in, scenario_in, classes_this_outcome, scaling_in, n_estimators_gbc_in, outbase, outpath, serverDataPath, automaticPipeline = True, combineFolder = None, combineFolderFileSuffix = None, tenFolds = True):
    sensors_to_combine = sorted(sensors_to_combine)
#     print ("Params: {0} {1} {2} {3} {4} {5} {6} {7} {8}".format(sensors_to_combine, thressensors_in, outcome_in, scaling_in, n_estimators_gbc_in, outpath, timelimited, limflag))
    outcome = outcome_in
    # loadCorrectFilePathForOutcome(outcome)
    thressensors = thressensors_in
    adaboost= True
    scaling = scaling_in
    n_estimators = 10
    n_estimators_gbc = n_estimators_gbc_in
    learning_rate = 0.1
    learning_rate_gbc = 0.1
    bs = GradientBoostingClassifier(random_state=0, n_estimators = n_estimators_gbc, learning_rate = learning_rate_gbc)
    base_estimator = bs
    paramsDict = {"adaboost": adaboost, "scaling": scaling, "base_estimator": base_estimator, "n_estimators": n_estimators, "n_estimators_gbc": n_estimators_gbc, "learning_rate": learning_rate, "learning_rate_gbc": learning_rate_gbc}
    if (automaticPipeline):
        best_models = LoadBestModelsDictAutomatePipeline(sensors_to_combine, outcome, classes_this_outcome, serverDataPath, combineFolder, combineFolderFileSuffix, tenFolds)
    else:
        print ("Manual pipeline not supported yet")
#         best_models = LoadBestModelsDict(sensors_to_combine, outcome) ## DOES NOT SUPPORT 10-FOLD
    if not tenFolds:
        print ("LOO not supported yet")
#         df_all, df_all_sensors, df_all_post, thres = LoadPredictions(sensors_to_combine, outcome_in, classes_this_outcome, best_models, thressensors, serverDataPath, timelimited, limflag)
#         p_idx, p_preds, p_true = runLOO(df_all_sensors, df_all_post, thres, pre_status, outcome_in, classes_this_outcome, sensors_to_combine, best_models, paramsDict, outbase, outpath)
    else:
        df_all, df_all_sensors, df_all_post, df_fold_indicator_all, thres = LoadPredictionsTenFold(sensors_to_combine, outcome_in, scenario_in, classes_this_outcome, best_models, thressensors, serverDataPath)
        p_idx, p_preds, p_true = runTenFolds(df_all_sensors, df_all_post, thres, outcome_in, classes_this_outcome, sensors_to_combine, best_models, paramsDict, outbase, outpath)
    #accuracyByLevel(df_all, p_idx, p_preds, p_true)
    return (p_idx, p_true, p_preds)