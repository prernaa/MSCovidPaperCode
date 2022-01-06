import pandas as pd

def number_outgoing_calls(g):
#     if g is None or len(g) == 0:
    if g is None:
        return None
    return g[g["call_type"] == 2]["call_type"].count()

def number_incoming_calls(g):
#     if g is None or len(g) == 0:
    if g is None:
        return None
    return  g[g["call_type"] == 1]["call_type"].count()

def number_missed_calls(g):
#     if g is None or len(g) == 0:
    if g is None:
        return None
    return  g[g["call_type"] == 3]["call_type"].count()

def duration_outgoing_calls_seconds(g):
#     if g is None or len(g) == 0:
    if g is None:
        return None
    return  int(g[g["call_type"] == 2]["call_duration"].sum())

def duration_incoming_calls_seconds(g):
#     if g is None or len(g) == 0:
    if g is None:
        return None
    return  int(g[g["call_type"] == 1]["call_duration"].sum())

def number_of_correspondents_phone(g):
#     if g is None or len(g) == 0:
    if g is None:
        return None
    return g["trace"].nunique()
