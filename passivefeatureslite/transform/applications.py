import numpy as np
from google_play_scraper import app
APP_FOREGROUND_EXCLD = ["'com.samsung.android.app.cocktailbarservice'", "'com.touchtype.swiftkey'"]


def add_time_spent_per_app(df, timestamp_ms_col, subject_col):
    df = df.sort_values(by=[subject_col, timestamp_ms_col])
    df[timestamp_ms_col] = df[timestamp_ms_col].astype(np.int64)
    df["next_{0}".format(timestamp_ms_col)] = (df[timestamp_ms_col].shift(periods=-1))
    df["time_spent_secs"] = (df["next_{0}".format(timestamp_ms_col)] - df[timestamp_ms_col])/1000.0
    return df

def add_time_spent_per_app(df, timestamp_ms_col, subject_col, pkg_col, app_col):
    df = df.sort_values(by=[subject_col, timestamp_ms_col])
    df[timestamp_ms_col] = df[timestamp_ms_col].astype(np.int64)
    ## remove some packages that aren't real apps 
    df[app_col] = np.where(df[app_col].isnull(), df[pkg_col], df[app_col])
    try:
        df = df[~(df[pkg_col].isin(APP_FOREGROUND_EXCLD))]
        df = df[~df[pkg_col].str.contains("inputmethod")]
        df = df[((~(df[pkg_col].str.contains("inputmethod"))) & (~(df[app_col].str.contains("inputmethod"))))]
        df = df[((~(df[pkg_col].str.contains("keyboard"))) & (~(df[app_col].str.contains("keyboard"))))]
    except Exception as e:
#         print (len(df)-df[pkg_col].count())
#         print (len(df)-df[app_col].count())
#         print (df[(df[app_col].isnull())])
        raise Exception(e)
    ## calculate time spent
    df["next_{0}".format(timestamp_ms_col)] = (df[timestamp_ms_col].shift(periods=-1))
    df["next_{0}".format(subject_col)] = (df[subject_col].shift(periods=-1))
    df["next_{0}".format(timestamp_ms_col)] = np.where( (df["next_{0}".format(subject_col)] != df[subject_col]), np.nan, df["next_{0}".format(timestamp_ms_col)])
    df["time_spent_secs"] = (df["next_{0}".format(timestamp_ms_col)] - df[timestamp_ms_col])/1000.0
    return df



APP_CATEGORY_TIME_SAVER_DICT = {}
def apply_app_category(row, pkg_col, total_apps):
    pkg = row[pkg_col]
    cat = np.nan
#     if (row["sno"]%200) == 0:
#         print ("---- {0} out of {1} apps".format(row["sno"], total_apps))
    if pkg in APP_CATEGORY_TIME_SAVER_DICT:
        cat = APP_CATEGORY_TIME_SAVER_DICT[pkg]
    else:
        try:
            res = app(pkg)
            cat = res["genreId"]
            if "GAME" in cat:
                cat = "GAME"
        except Exception as e: 
            cat = "404"
            #print ("{0}: {1}".format(pkg, e))
            pass
        APP_CATEGORY_TIME_SAVER_DICT[pkg] = cat
    return cat

def add_app_category(df, pkg_col, cat_col_out):
    total_apps = len(df)
    df["sno"] = list(range(0, total_apps))
    df[cat_col_out] = df.apply((lambda x: apply_app_category(x, pkg_col, total_apps)), axis=1)
    df = df.drop(columns=["sno"])
    return df
