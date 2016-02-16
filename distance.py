# coding: utf-8

# Author: Hussein AL-NATSHEH <hussein.al-natsheh@ish-lyon.cnrs.fr>
# License: BSD 3 clause
# 2016

import pandas as pd
import argparse

def load_data(dataset, verbose=0):
	if dataset == "sts":
		#Load STS data (combined 2012-2014 and cleaned)
		data = pd.read_csv('data/sts_gs_all.csv')
		if verbose == 2:
			print data.shape
			print data.head(n=10)
		elif verbose == 1:
			print data.shape

		return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--verbose", default=1, type=int)
    args = parser.parse_args()

    sts_gs_data = load_data (args.dataset, args.verbose)