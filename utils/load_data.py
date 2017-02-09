# coding: utf-8

# Author: Hussein AL-NATSHEH <hussein.al-natsheh@ish-lyon.cnrs.fr>
# License: BSD 3 clause
# 2016, 2017

import pandas as pd
import numpy as np

def load_dataset(data_file, verbose=0):
    data = pd.read_csv(data_file, sep='~', dtype={'Score': np.float32})
    if verbose == 2:
        print data.shape
        print data.head(n=10)
    elif verbose == 1:
        print data.shape
    X = data.as_matrix(columns=["Sent1", "Sent2"])
    y = data['Score'].values
    return X, y
