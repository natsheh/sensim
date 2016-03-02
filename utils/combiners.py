# coding: utf-8

# Author: Hussein AL-NATSHEH <hussein.al-natsheh@ish-lyon.cnrs.fr>
# License: BSD 3 clause
# 2016

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

class PairCosine(BaseEstimator, TransformerMixin):
    """Cosine similarity on paired data."""

    def fit(self, X, y=None):
        """(Do nothing).

        Parameters
        ----------
        :param X: array-like, shape (n_pair, 2)
            Input data.

        Returns
        -------
        :returns: self
        """
        return self

    def transform(self, X):
        """Compute the cosine similarity of all pairs in ``X``.

        To be added.

        Parameters
        ----------
        :param X: array-like, shape (n_pair, 2)
            Input paired data.

        Returns
        -------
        :returns Xt: array-like, shape (n_pair, 1)
            The transformed data.
        """
        n_pairs, two = X.shape
        Xt = np.zeros(n_pairs, dtype=float)
        i=0
        for x1, x2 in X:
            Xt[i] = cosine_similarity(x1, x2)
            i+=1

        return Xt.reshape(n_pairs, 1)
