# -*- coding: utf-8 -*-
#
# This file is part of sensim

"""Helpers for sentence semantic similarity model.

.. Author:: Hussein AL-NATSHEH <hussein.al-natsheh@ish-lyon.cnrs.fr>

"""

def to_numeric(s):
    res = []
    for n in s:
        if n.replace(".", "", 1).isdigit():
            res.append(float(n))
    if len(res) < 1:
        res.append(' ')
    return res
