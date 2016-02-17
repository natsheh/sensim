# coding: utf-8

# Author: Hussein AL-NATSHEH <hussein.al-natsheh@ish-lyon.cnrs.fr>
# License: BSD 3 clause
# 2016

import pandas as pd
import argparse
import numpy as np

def load_data(dataset, verbose=0):
	if dataset == "sts":
		#Load STS data (combined 2012-2014 and cleaned)
		data = pd.read_csv('data/sts_gs_all.csv', dtype={'Score': np.float32})
		if verbose == 2:
			print data.shape
			print data.head(n=10)
		elif verbose == 1:
			print data.shape
		X = data.as_matrix(columns=["Sent1", "Sent2"])
		y = data.as_matrix(columns=["Score"])
		return X, y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--verbose", default=1, type=int)
    args = parser.parse_args()

    X, y = load_data (args.dataset, args.verbose)
