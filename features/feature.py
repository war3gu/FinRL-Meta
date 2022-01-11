"""
will read in from 1 file of format | "time" | tech_ind_1 | ... | tech_ind_2 | "tic" |
and parse to user train_start - train_end, test_start - test_end and combine with
ohclv data source for analysis

user must ensure col_0 is "time", and col_n is "tic"
"""
import pandas as pd
import numpy as np


def read_custom_data(file):
    features = pd.read_csv(file)
    inds = features.columns[1:-1]

    unique_ticker = features.tic.unique()
    price_array = np.column_stack([features[features.tic==tic].close for tic in unique_ticker])
    tech_array = np.hstack([features.loc[(features.tic==tic), inds] for tic in unique_ticker])       
    assert price_array.shape[0] == tech_array.shape[0]
    return price_array, tech_array, np.array([])