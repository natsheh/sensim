# coding: utf-8

# Author: Hussein AL-NATSHEH <hussein.al-natsheh@ish-lyon.cnrs.fr>
# License: BSD 3 clause
# 2016

import pandas as pd
import argparse
import numpy as np

def load_dataset(dataset, verbose=0):
	if dataset == 'sts':
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

def load_glove(filepath, verbose=0):
	data = pd.read_csv(filepath, sep=' ', compression='gzip', skiprows=9, index_col=0, header=None, encoding='utf-8')
	if verbose == 2:
		print data.shape
		print data.head(n=10)
	elif verbose == 1:
		print data.shape