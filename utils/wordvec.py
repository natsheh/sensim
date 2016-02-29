# -*- coding: utf-8 -*-
#
# This file is part of sensim

"""Helpers for sentence semantic similarity model.

.. Author:: Hussein AL-NATSHEH <hussein.al-natsheh@ish-lyon.cnrs.fr>

"""

import numpy as np


def word2glove(word):
    """Get the GloVe vector representation of the word.

    Parameters
    ----------
    :param dframe: Pandas DataFrame
        Pre-trained GloVe loaded dataframe
    
    :param word: string
        word

    Returns
    -------
    :returns: Vecotr
        Glove vector of the word
    """
    global glove6b300d
    
    word = word.lower()
    if word not in glove6b300d.index:
        return np.zeros(300, dtype=float, order='C').reshape(1, -1)
    else:
        return np.array(glove6b300d.loc[word]).reshape(1, -1)
