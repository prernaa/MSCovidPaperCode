import pandas as pd
import numpy as np


def number_of_outgoing_messages(g):
#     if g is None or len(g) == 0:
    if g is None:
        return None
    return g["message_type"][g["message_type"] == 2].count()

def number_of_incoming_messages(g):
#     if g is None or len(g) == 0:
    if g is None:
        return None
    return g["message_type"][g["message_type"] == 1].count()

# def most_frequent_correspondent(g):
#     if g is None or len(g) == 0:
#         return None
#     if g.empty:
#         return None
#     result = g["trace"].mode()
#     if result.empty:
#         return g["trace"].iloc[0]
#     else:
#         return result[0]

def number_of_correspondents(g):
#     if g is None or len(g) == 0:
    if g is None:
        return None
    return g["trace"].nunique()
