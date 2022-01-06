import glob
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error
from collections import Counter
# from utils import printAllCols
# from configOutcomes import outSuffix, classesPerOutcome

classesPerOutcome = {
    u'phq9_score_lbl': [0,1],
    u'mfis_m_lbl': [0,1],
    u'psqi_total_lbl': [0,1],
    u'pss_m_lbl': [0,1],
    u'msrsr_m_lbl': [0,1]
}

def printAllCols(df):
    with pd.option_context('display.max_rows', 3, 'display.max_columns', None):
        print(df)

def runForFolderDetailed10Fold(folderpath, modeltype, modeli, outcome_status, printtruepreds = False, verbose = True, printtruevcs=False):
    print (folderpath)
    filenames = glob.glob(folderpath + "*_test_model1.csv")
#     print (folderpath + "*_test_model1.csv")
#     print (filenames)
    print ("{0} files found".format(len(filenames)))
    
    resultsDict_avg = {}
    resultsDict_avg["modelpath"] = [folderpath]
    resultsDict_avg["modeltype"] = [modeltype]
    resultsDict_avg["modeli"] = [modeli]
    resultsDict_avg["outcome_status"] = [outcome_status]
    resultsDict_avg["N"] = [0]
    resultsDict_avg["accuracy"] = [0]
    resultsDict_avg["f1allmean"] = [0]
    resultsDict_avg["f11"] = [0]
    resultsDict_avg["error"] = [np.nan]
    cols_avg = ["modeltype", "modeli", "outcome_status", "N", "accuracy", "f1allmean", "error"]
    cols_to_avg = ["accuracy", "f1allmean", "error"]
    cols = []
    if len(classesPerOutcome[outcome_status]) == 2:  # binary
#         mainMetric = "accuracy"
        mainMetric = "f11"
        mainMetricAscending = False
        f1mainidx = "f11"
        cols_avg+= ['precision0', 'recall0', 'f10', 'precision1', 'recall1', 'f11']
    else:
        f1mainidx = "f1allmean"
        mainMetric = f1mainidx
        mainMetricAscending = False #True

    NumFolds = len(filenames)
    print ("NUMFOLDS = {0}".format(NumFolds))
    dfOut_per_fold = pd.DataFrame()
    loopCounter = 0
    test_true_df_list = []
    for filename in filenames:
        loopCounter = loopCounter + 1
        foldid = filename.split("_")[-3].split("/")[-1]
        result = pd.read_csv(filename, index_col=False)
        result = result.set_index("did")
        did_fold_list= result.index
        test_pred = result["Y_pred"]
        test_true = result["Y_true"]
        test_true_df = pd.DataFrame(test_true).reset_index()
        test_true_df_list.append(test_true_df)
#         print (did_fold_list)
#         print (test_true_df)
        cnts_per_label = dict(Counter(test_true))
        cnts_per_label_str = ','.join("%s:%r" % (key, val) for (key, val) in cnts_per_label.iteritems())
        if printtruepreds:
            df = pd.DataFrame({"Pred": test_pred, "True": test_true})
            printAllCols(df.T)
        ## results to output
        resultsDict = {}
        resultsDict["foldnum"] = [foldid]
        resultsDict["modelpath"] = [folderpath]
        resultsDict["modeltype"] = [modeltype]
        resultsDict["modeli"] = [modeli]
        resultsDict["outcome_status"] = [outcome_status]
        resultsDict["N"] = [len(did_fold_list)]
        resultsDict["accuracy"] = [accuracy_score(test_true, test_pred)]
        resultsDict["f1allmean"] = [f1_score(test_true, test_pred, average="macro")] # this was precision_score earlier
        resultsDict["error"] = [np.nan]
        cols = ["foldnum", "modeltype", "modeli", "outcome_status", "N", "accuracy", "f1allmean", "error"]
        ## summing in avg dict
        resultsDict_avg["N"] = [resultsDict_avg["N"][0] + resultsDict["N"][0]]
        resultsDict_avg["accuracy"] = [resultsDict_avg["accuracy"][0] + resultsDict["accuracy"][0]]
        resultsDict_avg["f1allmean"] = [resultsDict_avg["f1allmean"][0] + resultsDict["f1allmean"][0]]
        resultsDict_avg["error"] = [resultsDict_avg["error"][0] + resultsDict["error"][0]]
        # Calculating fpr per class
        if len(classesPerOutcome[outcome_status]) == 2:  # binary
            for c in classesPerOutcome[outcome_status]:
                resultsDict["precision{0}".format(c)] = [precision_score(test_true, test_pred, pos_label = c, average="binary")]
                resultsDict["recall{0}".format(c)] = [recall_score(test_true, test_pred, pos_label = c, average="binary")]
                resultsDict["f1{0}".format(c)] = [f1_score(test_true, test_pred, pos_label = c, average="binary")]
                cols = cols + ["precision{0}".format(c), "recall{0}".format(c), "f1{0}".format(c)]
                # summing in avg dict
                if "precision{0}".format(c) not in resultsDict_avg:
                    resultsDict_avg["precision{0}".format(c)] = [0]
                if "recall{0}".format(c) not in resultsDict_avg:
                    resultsDict_avg["recall{0}".format(c)] = [0]
                if "f1{0}".format(c) not in resultsDict_avg:
                    resultsDict_avg["f1{0}".format(c)] = [0]
                resultsDict_avg["precision{0}".format(c)] = [resultsDict_avg["precision{0}".format(c)][0] + resultsDict["precision{0}".format(c)][0]]
                resultsDict_avg["recall{0}".format(c)] = [resultsDict_avg["recall{0}".format(c)][0] + resultsDict["recall{0}".format(c)][0]]
                resultsDict_avg["f1{0}".format(c)] = [resultsDict_avg["f1{0}".format(c)][0] + resultsDict["f1{0}".format(c)][0]]
                if loopCounter == 1:
                    cols_avg = cols_avg + ["precision{0}".format(c), "recall{0}".format(c), "f1{0}".format(c)]
                    cols_to_avg = cols_to_avg + ["precision{0}".format(c), "recall{0}".format(c), "f1{0}".format(c)]
        else:
            for c in classesPerOutcome[outcome_status]:
                resultsDict["precision{0}".format(c)] = [precision_score(test_true, test_pred, pos_label = c, labels = [c], average="macro")]
                resultsDict["recall{0}".format(c)] = [recall_score(test_true, test_pred, pos_label = c, labels = [c], average="macro")]
                resultsDict["f1{0}".format(c)] = [f1_score(test_true, test_pred, pos_label = c, labels = [c], average="macro")]
                cols = cols + ["precision{0}".format(c), "recall{0}".format(c), "f1{0}".format(c)]
                # summing in avg dict
                if "precision{0}".format(c) not in resultsDict_avg:
                    resultsDict_avg["precision{0}".format(c)] = [0]
                if "recall{0}".format(c) not in resultsDict_avg:
                    resultsDict_avg["recall{0}".format(c)] = [0]
                if "f1{0}".format(c) not in resultsDict_avg:
                    resultsDict_avg["f1{0}".format(c)] = [0]
                resultsDict_avg["precision{0}".format(c)] = [resultsDict_avg["precision{0}".format(c)][0] + resultsDict["precision{0}".format(c)][0]]
                resultsDict_avg["recall{0}".format(c)] = [resultsDict_avg["recall{0}".format(c)][0] + resultsDict["recall{0}".format(c)][0]]
                resultsDict_avg["f1{0}".format(c)] = [resultsDict_avg["f1{0}".format(c)][0] + resultsDict["f1{0}".format(c)][0]]
                if loopCounter == 1:
                    cols_avg = cols_avg + ["precision{0}".format(c), "recall{0}".format(c), "f1{0}".format(c)]
                    cols_to_avg = cols_to_avg + ["precision{0}".format(c), "recall{0}".format(c), "f1{0}".format(c)]
            if (resultsDict["N"][0] != 0):
                # print ("Num People {0}".format(resultsDict["N"]))
                resultsDict["error"] = [mean_absolute_error(test_true, test_pred)]
                # summing in avg dict
                if np.isnan(resultsDict_avg["error"][0]):
                    resultsDict_avg["error"] = [0]
                resultsDict_avg["error"] = [resultsDict_avg["error"][0] + resultsDict["error"][0]]
        dfOut_this_fold = pd.DataFrame(resultsDict)
        dfOut_per_fold = dfOut_per_fold.append(dfOut_this_fold)
    if NumFolds>0:
        for colavg in cols_to_avg:
            resultsDict_avg[colavg] = [resultsDict_avg[colavg][0]/float(NumFolds)]
    dfOut = pd.DataFrame(resultsDict_avg)
    dfOut = dfOut.set_index("modelpath")
    if NumFolds>0:
        dfOut_per_fold = dfOut_per_fold.set_index("modelpath", "foldnum")
        test_true_df_all = pd.concat(test_true_df_list)
        test_true_df_all = test_true_df_all.drop_duplicates(subset=["did"])
        #print (test_true_df_all)
        if printtruevcs:
            print ("true value cnts")
            vcs = test_true_df_all["Y_true"].value_counts().to_dict()
            print (vcs)
            baseline = float(vcs[0])/(vcs[0]+vcs[1])
            print (baseline)
    return (dfOut, dfOut_per_fold, cols_avg, cols, mainMetric, mainMetricAscending, resultsDict_avg["N"], resultsDict_avg["accuracy"], resultsDict_avg[f1mainidx], resultsDict_avg["error"])


def runForFolderDetailed(folderpath, modeltype, modeli, outcome_status, printtruepreds = False, verbose = True):
    print (folderpath)
    filenames = glob.glob(folderpath + "*_test_model1.csv")
    print ("{0} files found".format(len(filenames)))
    testpred_list = []
    testtrue_list = []
    did_list = []
    fcnt = 0
    for filename in filenames:
        did = filename.split("_")[-3].split("/")[-1]
        result = pd.read_csv(filename)
        result = result.iloc[0]
        test_pred = result["Y_pred"]
        test_true = result["Y_true"]
        did_list.append(did)
        testpred_list.append(test_pred)
        testtrue_list.append(test_true)
    testpred_list = np.array(testpred_list)
    testtrue_list = np.array(testtrue_list)
    cnts_per_label = dict(Counter(testtrue_list))
    cnts_per_label_str = ','.join("%s:%r" % (key, val) for (key, val) in cnts_per_label.iteritems())
    if printtruepreds:
        df = pd.DataFrame({"Pred": testpred_list, "True": testtrue_list})
        printAllCols(df.T)
    ## results to output
    mainMetric = None
    mainMetricAscending = None
    resultsDict = {}
    resultsDict["modelpath"] = [folderpath]
    resultsDict["modeltype"] = [modeltype]
    resultsDict["modeli"] = [modeli]
    resultsDict["outcome_status"] = [outcome_status]
    resultsDict["N"] = [len(did_list)]
    resultsDict["accuracy"] = [accuracy_score(testtrue_list, testpred_list)]
    resultsDict["f1allmean"] = [f1_score(testtrue_list, testpred_list, average="macro")] # this was precision_score earlier
    resultsDict["error"] = [np.nan]
    cols = ["modeltype", "modeli", "outcome_status", "N", "accuracy", "f1allmean", "error"]
    if len(classesPerOutcome[outcome_status]) == 2:  # binary
        mainMetric = "accuracy"
        mainMetricAscending = False
        for c in classesPerOutcome[outcome_status]:
            resultsDict["precision{0}".format(c)] = [precision_score(testtrue_list, testpred_list, pos_label = c, average="binary")]
            resultsDict["recall{0}".format(c)] = [recall_score(testtrue_list, testpred_list, pos_label = c, average="binary")]
            resultsDict["f1{0}".format(c)] = [f1_score(testtrue_list, testpred_list, pos_label = c, average="binary")]
            cols = cols + ["precision{0}".format(c), "recall{0}".format(c), "f1{0}".format(c)]
    else:
        mainMetric = "error"
        mainMetricAscending = True
        for c in classesPerOutcome[outcome_status]:
            resultsDict["precision{0}".format(c)] = [precision_score(testtrue_list, testpred_list, pos_label = c, labels = [c], average="macro")]
            resultsDict["recall{0}".format(c)] = [recall_score(testtrue_list, testpred_list, pos_label = c, labels = [c], average="macro")]
            resultsDict["f1{0}".format(c)] = [f1_score(testtrue_list, testpred_list, pos_label = c, labels = [c], average="macro")]
            cols = cols + ["precision{0}".format(c), "recall{0}".format(c), "f1{0}".format(c)]
        if (resultsDict["N"][0] != 0):
            # print ("Num People {0}".format(resultsDict["N"]))
            resultsDict["error"] = [mean_absolute_error(testtrue_list, testpred_list)]            
    dfOut = pd.DataFrame(resultsDict)
    dfOut = dfOut.set_index("modelpath")
    return (dfOut, cols, mainMetric, mainMetricAscending, resultsDict["N"], resultsDict["accuracy"], resultsDict["f1allmean"], resultsDict["error"])

def runForFolder(folderpath, modeltype, modeli, outcome_status, printtruepreds = False, verbose = True):
    print (folderpath)
    filenames = glob.glob(folderpath + "*_test_model1.csv")
    #     print ("Found {0} files".format(len(filenames)))
    #     print ("For {0} model # {1}".format(modeltype, modeli))
    testpred_list = []
    #testpredproba_list = []
    testtrue_list = []
    did_list = []
    fcnt = 0
    for filename in filenames:
        did = filename.split("_")[-3].split("/")[-1]
        result = pd.read_csv(filename)
        result = result.iloc[0]
        ## We actually dont need the below. Can just get counts from testtrue_list
        #         fcnt = fcnt + 1
        #         if fcnt == 1:
        #             if "cnts_per_lbl" in result:
        #                 cnts_per_label = result["cnts_per_lbl"]
        #                 print (cnts_per_label)
        #             else:
        #                 print ("No counts per label found")
        #test_proba = result["Y_pred_proba"]
        test_pred = result["Y_pred"]
        test_true = result["Y_true"]
        did_list.append(did)
        #testpredproba_list.append(test_proba)
        testpred_list.append(test_pred)
        testtrue_list.append(test_true)
        #     print ("Calculating over {0} subjects".format(len(did_list)))
    testpred_list = np.array(testpred_list)
    testtrue_list = np.array(testtrue_list)
    cnts_per_label = dict(Counter(testtrue_list))
    cnts_per_label_str = ','.join("%s:%r" % (key, val) for (key, val) in cnts_per_label.iteritems())
    if printtruepreds:
        df = pd.DataFrame({"Pred": testpred_list, "True": testtrue_list})
        printAllCols(df.T)
    numpple = len(did_list)
    ma_error = None
    if len(classesPerOutcome[outcome_status]) == 2:  # binary
        accuracy = accuracy_score(testtrue_list, testpred_list)
        precision = precision_score(testtrue_list, testpred_list, average="binary")
        recall = recall_score(testtrue_list, testpred_list, average="binary")
        f1 = f1_score(testtrue_list, testpred_list, average="binary")
        if verbose:
            print ("{5} paramDict#{6} \nN = {4}. Cnts_Per_Label = {7}. Accuracy = {0}. Precision = {1}. Recall = {2}. F1 = {3}.".format( accuracy, precision, recall, f1, numpple, modeltype, modeli, cnts_per_label_str))
    elif len(classesPerOutcome[outcome_status]) > 2 :  # multiclass
        accuracy = accuracy_score(testtrue_list, testpred_list)
        precision = precision_score(testtrue_list, testpred_list, average="macro")
        recall = recall_score(testtrue_list, testpred_list, average="macro")
        f1 = f1_score(testtrue_list, testpred_list, average="macro")
        wprecision = precision_score(testtrue_list, testpred_list, average="weighted")
        wrecall = recall_score(testtrue_list, testpred_list, average="weighted")
        wf1 = f1_score(testtrue_list, testpred_list, average="weighted")
        if (numpple == 0):
            ma_error = np.nan
        else:
            ma_error = mean_absolute_error(testtrue_list, testpred_list)
        if verbose:
            print ("{5} paramDict#{6} \nN = {4}. Cnts_Per_Label = {7}. Accuracy = {0}. Error = {11}. Macro Precision = {1}. Macro Recall = {2}. Macro F1 = {3}. Weighted Precision = {8}. Weighted Recall = {9}. Weighted F1 = {10}.".format( accuracy, precision, recall, f1, numpple, modeltype, modeli, cnts_per_label_str, wprecision, wrecall, wf1, ma_error))
        labelorder = cnts_per_label.keys()
        labelorder_str = ", ".join(str(v) for v in labelorder)
        precisionlbls = precision_score(testtrue_list, testpred_list, labels=labelorder, average=None)
        recalllbls = recall_score(testtrue_list, testpred_list, labels=labelorder, average=None)
        f1lbls = f1_score(testtrue_list, testpred_list, labels=labelorder, average=None)
        precisionlbls_str = ", ".join(str(v) for v in precisionlbls)
        recalllbls_str = ", ".join(str(v) for v in recalllbls)
        f1lbls_str = ", ".join(str(v) for v in f1lbls)
        if verbose:
            print ("Label order: {0}. Precision: {1}. Recall: {2}. F1: {3}.".format(labelorder_str, precisionlbls_str, recalllbls_str, f1lbls_str))
    else:
        raise Exception("Unrecognized outcome status")
    return (numpple, accuracy, precision, recall, f1, ma_error)


def runForOneModel(SENSOR, modeltype, modeli, outcome_status, pre_status, serverDataPath, printtruepreds = False, verbose = False):
    limflag = ""
    # folderpath = serverDataPath + "/models/{0}".format(SENSOR) + "_{3}_results{0}{2}prestatus{1}".format(limflag, pre_status, modeltype, outcome_status) + "_rlog_{0}/".format(modeli)
    folderpath = serverDataPath+"/models/{0}".format(SENSOR_NAME)+"_{1}_results{0}_10Fold".format(modelname, outcome_status)+"_rlog_{0}/".format(modeli)
    
    # print (folderpath)
    numpple, accuracy, precision, recall, f1, ma_error = runForFolder(folderpath, modeltype, modeli, outcome_status, printtruepreds = printtruepreds, verbose = verbose)
    return (numpple, accuracy, precision, recall, f1, ma_error)
