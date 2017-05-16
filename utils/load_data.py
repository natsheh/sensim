# coding: utf-8

# Author: Hussein AL-NATSHEH <hussein.al-natsheh@ish-lyon.cnrs.fr>
# License: BSD 3 clause
# 2016, 2017

import pandas as pd
import numpy as np


def read_tsv(path_to_file):
    df = pd.DataFrame()
    line_nb = 0
    with open(path_to_file) as file:
        for line in file.readlines():
            for num, cell in enumerate(line.split('\t')):
                df.set_value(line_nb, 'column_{}'.format(num),cell.strip())
            line_nb = line_nb + 1
    return df

def df_2_dset(dframe, sent1_col="Sent1", sent2_col="Sent2", score_col="Score"):
    X = dframe.as_matrix(columns=[sent1_col, sent2_col])
    y = dframe[score_col].values
    return X, y	

def load_dataset(data_file, verbose=0, sep='~'):
    data = pd.read_csv(data_file, sep=sep, dtype={'Score': np.float32})
    if verbose == 2:
        print data.shape
        print data.head(n=10)
    elif verbose == 1:
        print data.shape
    X, y = df_2_dset(data)
    return X, y
