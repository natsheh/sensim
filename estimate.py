# coding: utf-8

# Author: Hussein AL-NATSHEH <hussein.al-natsheh@cnrs.fr>
# License: BSD 3 clause
# 2016, 2017

import argparse
import numpy as np
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimator", default='distance_model.pickle', type=str)
    parser.add_argument("--sent1", default='There was a rather ridiculous young man on it— indigo neck, cord round his hat', type=str)
    parser.add_argument("--sent2", default='There was a young man on this bus who was rather ridiculous, not because he wasn’t carrying a bayonet, but because he looked as if he was carrying one when all the time he wasn’t carrying one.', type=str)
    args = parser.parse_args()

    X = np.zeros(shape=(1,2), dtype=object)
    X[0][0] = args.sent1
    X[0][1] = args.sent2
    
    estimator = pickle.load(open(args.estimator,"r"))
    sensim = estimator.predict(X)[0]

    print ('Sentence 1: ', args.sent1)
    print ('Sentence 2: ', args.sent2)
    print ('Estimated semantic textual similarity', sensim)
