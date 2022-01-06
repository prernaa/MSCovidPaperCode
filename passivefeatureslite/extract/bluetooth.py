import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import time

def getOwnDevices(labeledFreq):
    labeledFreq = labeledFreq[labeledFreq["OwnOrNot"] == 1]
    return labeledFreq["bt_address"].values

def getMaxFreqClusterIdx(centers):
    mmax = -1000
    midx = None
    for ci in range(0, len(centers)):
        c = centers[ci]
        if c>mmax:
            mmax = c
            midx = ci
    return midx

def cluster_address_freq(df): # not for feature extraction directly
    for col in ["avgfreq", "numdays", "freq"]:
        col_zscore = col + '_z'
        df[col_zscore] = (df[col] - df[col].mean()) / df[col].std(ddof=0)
    # USE BELOW TO USE FREQ ONLY
    # freq = df["freq"]
    # score = freq
    # USE BELOW TO USE EQUALLY WEIGHTED AVGFREQ (TOTAL FREQ/ NUMBER OF DAYS THE ADDRESS APPEARS AT LEAST ONCE) AND NUMDAYS (NUMBER OF DAYS THE ADDRESS APPEARS AT LEAST ONCE)
    df = df.dropna(how='any')
    if len(df) < 3:
        return (df, None)

    avgfreq_z = df["avgfreq_z"]
    numdays_z = df["numdays_z"]
    score = avgfreq_z + numdays_z

    score.sort_values(ascending=False) # sorts in place (descending)
    maxscore = score.iloc[0]
    minscore = score.iloc[-1]
    midscore = (maxscore + minscore) / 2
    initial_k2 = np.array([[maxscore], [minscore]], np.int32)
    initial_k3 = np.array([[maxscore], [midscore], [minscore]], np.int32)
    X_array = score.values
    X = np.reshape(X_array, (len(score), 1))

    # K = 2, devices I own VS devices other people own
    kmeans_k2 = KMeans(n_clusters=2, init = initial_k2, n_init = 1).fit(X)
    labels_k2 = kmeans_k2.labels_
    centers_k2 = [c[0] for c in kmeans_k2.cluster_centers_]
    diff_k2 = [(X_array[xi] - centers_k2[labels_k2[xi]])**2 for xi in range(0, len(X_array))]
    sum_dist_k2 = sum(diff_k2)
    # K = 2, devices I own VS devices my partner/roommate owns (can also be other devices I own though) VS devices other people own
    kmeans_k3 = KMeans(n_clusters=3, init=initial_k3,  n_init = 1).fit(X)
    labels_k3 = kmeans_k3.labels_
    centers_k3 = [c[0] for c in kmeans_k3.cluster_centers_]
    diff_k3 = [(X_array[xi] - centers_k3[labels_k3[xi]])**2 for xi in range(0, len(X_array))]
    sum_dist_k3 = sum(diff_k3)
    if sum_dist_k2 < sum_dist_k3: # K = 2 is better
        lbls = labels_k2
        centers = centers_k2
        numclust = 2
    else:
        lbls = labels_k3
        centers = centers_k3
        numclust = 3
    # debug_df = pd.DataFrame({"Address":df["bt_address"], "X":score, "LBL":lbls})
    # print (debug_df)
    maxcluster = np.where(lbls == getMaxFreqClusterIdx(centers), 1, 0)
    df["OwnOrNot"] = maxcluster
    return (df, numclust)


def number_samples_bluetooth(g):
    if g is None:
        return None
    return len(g)

def number_unique_devices(g):
    if g is None or len(g) == 0:
        return None
    return g["bt_address"].nunique()

def number_unique_devices_of_others(g, args=None):
    if g is None or len(g) == 0:
        return None
    owndevices = args[0]
    g = g[~g["bt_address"].isin(owndevices)]
    if len(g) == 0:
        return 0
    return number_unique_devices(g)

def number_unique_devices_of_self(g, args=None):
    if g is None or len(g) == 0:
        return None
    owndevices = args[0]
    g = g[g["bt_address"].isin(owndevices)]
    if len(g) == 0:
        return 0
    return number_unique_devices(g)

def num_scans_of_most_frequent_device(g):
    if g is None or len(g) == 0:
        return None
    return g['bt_address'].value_counts().max()

def num_scans_of_least_frequent_device(g):
    if g is None or len(g) == 0:
        return None
    return g['bt_address'].value_counts().min()

def num_scans_of_most_frequent_device_of_others(g, args=None):
    if g is None or len(g) == 0:
        return None
    owndevices = args[0]
    g = g[~g["bt_address"].isin(owndevices)]
    if len(g) == 0:
        return 0
    return num_scans_of_most_frequent_device(g)

def num_scans_of_least_frequent_device_of_others(g, args=None):
    if g is None or len(g) == 0:
        return None
    owndevices = args[0]
    g = g[~g["bt_address"].isin(owndevices)]
    if len(g) == 0:
        return 0
    return num_scans_of_least_frequent_device(g)

def num_scans_of_most_frequent_device_of_self(g, args=None):
    if g is None or len(g) == 0:
        return None
    owndevices = args[0]
    g = g[g["bt_address"].isin(owndevices)]
    if len(g) == 0:
        return 0
    return num_scans_of_most_frequent_device(g)

def num_scans_of_least_frequent_device_of_self(g, args=None):
    if g is None or len(g) == 0:
        return None
    owndevices = args[0]
    g = g[g["bt_address"].isin(owndevices)]
    if len(g) == 0:
        return 0
    return num_scans_of_least_frequent_device(g)

def sum_num_scans_of_all_devices_of_self(g, args=None):
    if g is None or len(g) == 0:
        return None
    owndevices = args[0]
    g = g[g["bt_address"].isin(owndevices)]
    if len(g) == 0:
        return 0
    return len(g)

def sum_num_scans_of_all_devices_of_others(g, args=None):
    if g is None or len(g) == 0:
        return None
    owndevices = args[0]
    g = g[~g["bt_address"].isin(owndevices)]
    if len(g) == 0:
        return 0
    return len(g)

def avg_num_scans_of_all_devices_of_self(g, args=None):
    if g is None or len(g) == 0:
        return None
    owndevices = args[0]
    g = g[g["bt_address"].isin(owndevices)]
    if len(g) == 0:
        return 0
    return g['bt_address'].value_counts().mean()

def avg_num_scans_of_all_devices_of_others(g, args=None):
    if g is None or len(g) == 0:
        return None
    owndevices = args[0]
    g = g[~g["bt_address"].isin(owndevices)]
    if len(g) == 0:
        return 0
    return g['bt_address'].value_counts().mean()

def std_num_scans_of_all_devices_of_self(g, args=None):
    if g is None or len(g) == 0:
        return None
    owndevices = args[0]
    g = g[g["bt_address"].isin(owndevices)]
    if len(g) == 0:
        return 0
    return g['bt_address'].value_counts().std()

def std_num_scans_of_all_devices_of_others(g, args=None):
    if g is None or len(g) == 0:
        return None
    owndevices = args[0]
    g = g[~g["bt_address"].isin(owndevices)]
    if len(g) == 0:
        return 0
    return g['bt_address'].value_counts().std()
