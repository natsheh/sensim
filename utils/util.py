# -*- coding: utf-8 -*-
#
# This file is part of sensim

"""Helpers for sentence semantic similarity model.

.. Author:: Hussein AL-NATSHEH <hussein.al-natsheh@ish-lyon.cnrs.fr>

"""
from scipy.stats import pearsonr

def to_numeric(s):
    res = []
    for n in s:
        if n.replace(".", "", 1).isdigit():
            res.append(float(n))
    if len(res) < 1:
        res.append(' ')
    return res

def sts_score(est, X, y):
    y_est = est.predict(X)
    return pearsonr(y_est, y)[0]

