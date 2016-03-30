# coding: utf-8

# Author: Hussein AL-NATSHEH <hussein.al-natsheh@ish-lyon.cnrs.fr>
# License: BSD 3 clause
# 2016

import pandas as pd
import argparse
import numpy as np

def load_dataset(data_file, verbose=0):
    data = pd.read_csv(data_file, dtype={'Score': np.float32})
    if verbose == 2:
        print data.shape
        print data.head(n=10)
    elif verbose == 1:
        print data.shape
    X = data.as_matrix(columns=["Sent1", "Sent2"])
    y = data['Score'].values
    return X, y

def load_glove(filepath, verbose=0):
	glove6b300d = pd.read_csv(filepath, sep=' ', skiprows=[8], index_col=0, header=None, encoding='utf-8')
	if verbose == 2:
		print glove6b300d.shape
		print glove6b300d.head(n=10)
	elif verbose == 1:
		print glove6b300d.shape
	return glove6b300d