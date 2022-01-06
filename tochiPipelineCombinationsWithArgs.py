from itertools import combinations
import os
import numpy as np
import pandas as pd
from tochiCombinationsHelper import combineSensors
import sys

classesPerOutcome = {
    u'phq9_score_lbl': [0,1],
    u'mfis_m_lbl': [0,1],
    u'psqi_total_lbl': [0,1],
    u'pss_m_lbl': [0,1],
    u'msrsr_m_lbl': [0,1]
}

def getAllCombinations():
#     if bc and not bv:
#         sensors_to_combine = ["f_blue_BC", "f_call_BC", "f_loc_BC", "f_locMap_BC", "f_screen_BC", "f_slp_BC", "f_steps_BC"]
#     elif not bc and not bv:
#         sensors_to_combine = ["f_blue", "f_call", "f_loc", "f_locMap", "f_screen", "f_slp", "f_steps"]
#     else:
#         raise Exception("Unrecognized bc and bv boolean values specified")
#     sensors_to_combine = ["blue", "calls", "hr", "loc", "scr", "slp", "steps"]
    sensors_to_combine = ["calls", "hr", "loc", "scr", "slp", "steps"]

    ## calculating total combinations
    totalcombs = 0
    comblist = []
    therange = range(2, len(sensors_to_combine)+1)
    for r in reversed(therange):
        comb = list(combinations(sensors_to_combine, r))
        comblist = comblist + comb
        totalcombs = totalcombs + len(comb)
    print ("Total combinations to run = {0}".format(totalcombs))
    return (comblist, totalcombs)

# def getSelect7andBest_post_bdi_2():
#     comb1 = ["f_blue", "f_call", "f_loc", "f_locMap", "f_screen", "f_slp", "f_steps"]
#     comb2 = ["f_blue", "f_call", "f_screen", "f_steps"]
#     comblist = [comb1, comb2]
#     totalcombs = 2
#     return (comblist, totalcombs)
#
# def getSelect7andBest_change_bdi_2():
#     comb1 = ["f_blue", "f_call", "f_loc", "f_locMap", "f_screen", "f_slp", "f_steps"]
#     comb2 = ["f_locMap", "f_screen"]
#     comblist = [comb1, comb2]
#     totalcombs = 2
#     return (comblist, totalcombs)
#
# def getSelect7andBest_change_bdi_2_bc():
#     comb1 = ["f_blue_BC", "f_call_BC", "f_loc_BC", "f_locMap_BC", "f_screen_BC", "f_slp_BC", "f_steps_BC"]
#     comb2 = ["f_call_BC", "f_screen_BC", "f_steps_BC", "f_loc_BC"]
#     comblist = [comb1, comb2]
#     totalcombs = 2
#     return (comblist, totalcombs)

# def getSelect7andBest_change_bdi_2_levelsC_layer2(): # NEED TO CHECK THIS
#     comb1 = ["f_blue", "f_call", "f_loc", "f_locMap", "f_screen", "f_slp", "f_steps"]
#     comb2 = ["f_blue", "f_locMap", "f_screen", "f_slp"]
#     comblist = [comb1, comb2]
#     totalcombs = 2
#     return (comblist, totalcombs)


def str2bool(astr):
    if astr == "True":
        return (True)
    else:
        return (False)
def combinationMain():
#     print (sys.argv)
    serverDataPath = sys.argv[1]
    outcome_in = sys.argv[2]
    scenario_in = sys.argv[3]
    n_estimators_gbc_in = int(sys.argv[4])
    scaling_in = str2bool(sys.argv[5])
    toOutputPreds = str2bool(sys.argv[6])
    automaticPipeline = str2bool(sys.argv[7])
    combineFolder = sys.argv[8]
    combineFolderFileSuffix = sys.argv[9]
    tenFolds = str2bool(sys.argv[10])
#     print ("serverDataPath: {0}, outcome_in: {1}, n_estimators_gbc_in: {2}".format(serverDataPath, outcome_in, n_estimators_gbc_in))
#     print ("scaling_in: {0}, toOutputPreds: {1}, automaticPipeline: {2}".format(scaling_in, toOutputPreds, automaticPipeline))
#     print ("combineFolder: {0}, combineFolderFileSuffix: {1}, tenFolds: {2}".format(combineFolder, combineFolderFileSuffix, tenFolds))

    if not tenFolds:
        outbase = serverDataPath+"/SENSOR_COMBINATIONS_{0}_{1}/".format(outcome_in, scenario_in)
    else:
        outbase = serverDataPath+"/SENSOR_COMBINATIONS_{0}_{1}_10Folds/".format(outcome_in, scenario_in)
    if not (os.path.isdir(outbase)):
        os.mkdir(outbase)
    ## output accuracies, etc - not predictions
    outpath = outbase+"results_sensor_combinations_nestgbc{0}_scaling{1}.csv".format(n_estimators_gbc_in, scaling_in)
    ## output predictions - formatting is done below
    predoutpath = outbase+"preds_from_{0}sensors_{1}_nestgbc{2}_scaling{3}.csv"
    print (outpath)

    comblist, totalcombs = getAllCombinations()

    ## running for each combination
    tl = len(comblist)
    for ci in range(0, tl):
        sensors_to_combine_curr = list(comblist[ci])
        print ("Combination {0} of {1}".format(ci+1, totalcombs))
        print (sensors_to_combine_curr)
        thressensors_in = len(sensors_to_combine_curr)
        #print (thressensors_in)
        p_idx, p_true, p_preds = combineSensors(sensors_to_combine_curr, thressensors_in, outcome_in, scenario_in, classesPerOutcome[outcome_in], scaling_in, n_estimators_gbc_in, outbase, outpath, serverDataPath, automaticPipeline, combineFolder, combineFolderFileSuffix, tenFolds)
        if toOutputPreds:
            dfOutPreds = pd.DataFrame({"device_id": p_idx, "Y_true": p_true, "Y_pred": p_preds})
            predoutpath_curr = predoutpath.format(len(sensors_to_combine_curr), ",".join(sensors_to_combine_curr), n_estimators_gbc_in, scaling_in)
            dfOutPreds.to_csv(predoutpath_curr, columns=["device_id", "Y_true", "Y_pred"], header=True)

# call main
if __name__ == '__main__':
    combinationMain()