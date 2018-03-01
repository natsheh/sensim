# coding: utf-8

# License: BSD 3 clause
# 2018
"""
.. coauthor:: Haitham Selawi <haitham.selawi@mawdoo3.com>
.. coauthor:: Hussein Al-Natsheh <hussein.al-natsheh@cnrs.fr>

"""
import keras.backend as K
from keras.models import load_model
from keras.models import model_from_yaml
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

import numpy as np
import pandas as pd
import pickle

from scipy.stats import pearsonr

path = "../sensim/utils/LSTM/"

tokenizer = pickle.load(open(path+'tokenizer.p','rb'))

yaml_file = open(path+'model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()

loaded_model = model_from_yaml(loaded_model_yaml)
loaded_model.load_weights(path+'model.h5')
loaded_model.compile(loss='mse', optimizer='adam')

maxlen = loaded_model.get_layer(index = 0).input_shape[1]

#A transformer function that takes 2 sentences (or vector of pairs)
# and returns the numpy array representation of them
def mlstm_transformer(X):
	Xt = np.zeros(len(X), dtype=np.object)
	for (i, x) in enumerate(X):
		xt = tokenizer.texts_to_sequences(x)
		xt = pad_sequences(xt, maxlen = maxlen)
		xt = loaded_model.predict(xt)
		Xt[i] = xt
	return Xt

#Element-wise version of mlstm_transformer
def mlstm_element_transformer(x):
	xt = tokenizer.texts_to_sequences(x)
	xt = pad_sequences(xt, maxlen = maxlen)
	xt = loaded_model.predict(xt)
	return xt

def estimate(X):
	Xt = mlstm_transformer(X)
	y = np.zeros(len(X), dtype=np.float32)
	for (i, xt) in enumerate(Xt):
		y[i] = 5*np.exp(-np.linalg.norm(xt[0] - xt[1], ord = 1))
	return y

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

def load_sts_benchmark_dataset(dframe_file):
	dframe = read_tsv(dframe_file)
	dframe["Score"] = np.array(dframe['column_4'], dtype=np.float32)
	X, y = df_2_dset(dframe, sent1_col="column_5", sent2_col="column_6")
	return X, y

def sts_score(X, y, decimals=2):
	y_est = estimate(X)
	y_est[np.where(y_est > 5)] = 5
	y_est[np.where(y_est < 0)] = 0
	if decimals is not None:
		y_est = np.round(y_est, decimals=decimals)
	pickle.dump(y_est, open("y_est.p", "wb"))
	print ("y_est is pickled at y_est.p")
	return pearsonr(y_est, y)[0]

X_test, y_test = load_sts_benchmark_dataset(path+'sts-test.csv')
score = sts_score(X_test, y_test)#, decimals=None)
print score
pickle.dump(score, open("score.p", "wb"))


"""
#Example
# from the test set : 3.5 resulted as 3.62
line1_1 = 'A man is riding an electric bicycle.'
line1_2 = 'A man is riding a bicycle.'

# from the test set : 2.0, resulted as 2.31
line2_1 = 'A man is slicing a bun.'
line2_2 = 'A man is slicing a tomato.'

X = [[line1_1, line1_2],
	 [line2_1, line2_2]
	 ]


Xt = mlstm_transformer(X)

for xt in Xt:
	print(5*np.exp(-np.linalg.norm(xt[0] - xt[1], ord = 1)))
"""