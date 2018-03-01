# coding: utf-8

# Author: Hussein AL-NATSHEH <hussein.al-natsheh@ish-lyon.cnrs.fr>
# License: BSD 3 clause
# 2016

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

def _exp_manhatten(x1, x2):
    return np.exp(-np.linalg.norm(x1 - x2, ord = 1))

def _manhattan_distance(x,y):
    return sum(abs(a-b) for a,b in zip(x,y))

def _get_matches(r):
    matches = []
    for i in np.unique(r[:,0]):
        s_slice = r[r[:,0]==i]
        max_match = np.argmax(s_slice[:,2])
        matches.append(list(s_slice[max_match]))
    return np.array(matches)

def _sort_arr(arr, axis=0):
    return arr[arr[:,axis].argsort()]

def _solve_duplictes(mk, ids, th=None):
    res = []
    for id in ids:
        chk_dup = mk[mk[:,1]==id]
        if len(chk_dup) > 1:
            mn = np.argmin(chk_dup[:,2])
            chk_dup = np.delete(chk_dup, mn, axis=0)
        if chk_dup.shape[0] > 0:
            res.append(chk_dup[0])
    sorted_res = _sort_arr(np.array(res))
    if th is not None:
        res = []
        s = sorted_res
        if len (s[s[:,2] >= th])> 0:
            res.append((s[s[:,2] >= th]))
    sorted_res = np.array(res)

    return sorted_res

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

class PairMLSTM(BaseEstimator, TransformerMixin):
    """Exp. Manhattan similarity on paired data."""

    def fit(self, X, y=None):
        """(Do nothing).

        Parameters
        ----------
        :param X: array-like, shape (2*n_pair, 1)
            Input data.

        Returns
        -------
        :returns: self
        """
        return self

    def transform(self, X):
        """Compute the exponant Manhattan distance of all pairs in ``X``.

        To be added.

        Parameters
        ----------
        :param X: array-like, shape (2*n_pair, 1)
            Input paired data.

        Returns
        -------
        :returns Xt: array-like, shape (n_pair, 1)
            The transformed data.
        """
        n_samples = X.shape[0]

        #X1 = X[:,0]
        #X2 = X[:,1]
        #Xt = np.exp(-1 * pairwise_distances(X1, X2, metric='manhattan', n_jobs=1))
        Xt = np.zeros(n_samples, dtype=float)
        i = 0
        for x1, x2 in X:
            Xt[i] = 5 * np.exp(-np.linalg.norm(x1 - x2, ord = 1))
            i+=1

        return Xt.reshape(n_samples, 1)

class PairExpManhattan(BaseEstimator, TransformerMixin):
    """Exp. Manhattan similarity on paired data."""

    def fit(self, X, y=None):
        """(Do nothing).

        Parameters
        ----------
        :param X: array-like, shape (n_pair, features_of_example_pair)
            Input data.

        Returns
        -------
        :returns: self
        """
        return self

    def transform(self, X):
        """Compute the exponant Manhattan distance of all pairs in ``X``.

        To be added.

        Parameters
        ----------
        :param X: array-like, shape (n_pair, features_of_example_pair)
            Input paired data.

        Returns
        -------
        :returns Xt: array-like, shape (n_pair, 1)
            The transformed data.
        """
        n_samples, n_features_all = X.shape
        n_features = n_features_all // 2

        if sp.issparse(X):
            X = X.todense()

        X1 = X[:, :n_features]
        X2 = X[:, n_features:]
        #Xt = np.exp(-1 * pairwise_distances(X1, X2, metric='manhattan', n_jobs=1))
        Xt = np.zeros(n_samples, dtype=float)
        for i in range(n_samples):
            Xt[i] = np.exp(-np.linalg.norm(X1[i] - X2[i], ord = 1))

        return 5 * Xt.reshape(n_samples, 1)

class PairManhattan(BaseEstimator, TransformerMixin):
    """Manhattan distance on paired data."""

    def fit(self, X, y=None):
        """(Do nothing).

        Parameters
        ----------
        :param X: array-like, shape (n_pair, features_of_example_pair)
            Input data.

        Returns
        -------
        :returns: self
        """
        return self

    def transform(self, X):
        """Compute the Manhattan distance of all pairs in ``X``.

        To be added.

        Parameters
        ----------
        :param X: array-like, shape (n_pair, features_of_example_pair)
            Input paired data.

        Returns
        -------
        :returns Xt: array-like, shape (n_pair, 1)
            The transformed data.
        """
        n_samples, n_features_all = X.shape
        n_features = n_features_all // 2

        if sp.issparse(X):
            X = X.todense()

        X1 = X[:, :n_features]
        X2 = X[:, n_features:]
        Xt = np.zeros(n_samples, dtype=float)
        for i in range(n_samples):
            Xt[i] = np.linalg.norm(X1[i] - X2[i], ord = 1)

        return Xt.reshape(n_samples, 1)

class SmallerOtherParing(BaseEstimator, TransformerMixin):
    """Combine pairs of lists of words into list of paired words."""

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
        """Generate pairs of list elements of all pairs in ``X``.

        To be added.

        Parameters
        ----------
        :param X: array-like, shape (n_pair, 2)
            Input paired data where each pair element is a list of words

        Returns
        -------
        :returns Xt: array-like, shape (n_pair, 1)
            The transformed data where each element is a list like [(w_id, word), (w_id, word)]
        """
        
        n_pairs, two = X.shape
        Xt = np.zeros(n_pairs, dtype=object)
        p = 0
        for x1, x2 in X:
            if len(x1) < len(x2):
                smaller = x1
                other = x2
            else:
                smaller = x2
                other = x1
            n_pair_pairs = len(x1) * len(x2)
            X_pair = np.zeros(shape=(n_pair_pairs,2), dtype=object)
            hash_tbl_pair = np.zeros(shape=(n_pair_pairs,2), dtype=object)
            X_p = np.zeros(shape=(n_pair_pairs,2), dtype=object)

            smaller_dict = {}
            s_id = 0
            for w1 in smaller:
                s_id += 1
                smaller_dict[s_id] = w1

            o_id = 0
            other_dict = {}
            for w2 in other:
                    o_id += 1
                    other_dict[o_id] = w2
            k = 0
            for i in smaller_dict.keys():
                for j in other_dict.keys():
                    hash_tbl_pair[k] = [i, j]
                    k += 1

            i= 0
            for w1 in smaller:
                for w2 in other:
                    X_pair[i] = [w1, w2]
                    i += 1
            
            for i in range(n_pair_pairs):
                X_p[i] = (hash_tbl_pair[i][0], X_pair[i][0]), (hash_tbl_pair[i][1], X_pair[i][1])
            
            Xt[p] = X_p
            p += 1       
        
        return Xt

class RefGroupPairCosine(BaseEstimator, TransformerMixin):
    """Cosine similarity on paired data."""

    def fit(self, X, y=None):
        """(Do nothing).

        Parameters
        ----------
        :param X: array-like, shape (n_samples,)
            Input data 

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
        :param X: array-like, shape (n_samples, 2)
            Input paired data (output of PairGloveTransformer).

        Returns
        -------
        :returns Xt: array-like, shape (n_samples,)
            The transformed data: array of triples like (word_id, word_id, cosine_sim)
        """
        n_samples = len(X)
        Xt = np.zeros(n_samples, dtype=object)
        s_id = 0
        for sample in X:
            lst = []
            for tup in sample:
                w1, w2 = tup
                w1_id, w1_vec = w1
                w2_id, w2_vec = w2
                cos_sim = cosine_similarity(w1_vec, w2_vec)
                lst.append((w1_id, w2_id, cos_sim[0]))
            Xt[s_id] = lst
            s_id += 1
        
        return Xt

class GetMatches(BaseEstimator, TransformerMixin):
    """Get the best match for each word in smaller sentence from the other."""

    def fit(self, X, y=None):
        """(Do nothing).

        Parameters
        ----------
        :param X: array-like, shape (n_samples,)
            Input data 

        Returns
        -------
        :returns: self
        """
        return self

    def transform(self, X):
        """Get the best match for each word in smaller sentence from the other.

        To be added.

        Parameters
        ----------
        :param X: array-like, shape (n_samples, 2)
            Input paired data (output of RefGroupPairCosine).

        Returns
        -------
        :returns Xt: array-like, shape (n_samples,)
            The transformed data: array of triples like (word_id, word_id, cosine_sim)
        """
        n_samples = len(X)
        Xt = np.zeros(n_samples, dtype=object)
        s_id = 0
        for sample in X:
            r = np.array(map(list, sample))
            Xt[s_id] = _get_matches(r)
            s_id += 1
        return Xt

class SolveDuplicate(BaseEstimator, TransformerMixin):
    """Solve duplicates of words in the other sentence of the GetMatches result."""

    def fit(self, X, y=None):
        """(Do nothing).

        Parameters
        ----------
        :param X: array-like, shape (n_samples,)
            Input data 

        Returns
        -------
        :returns: self
        """
        return self

    def transform(self, X):
        """Solve duplicates of words in the other sentence of the GetMatches result.

        To be added.

        Parameters
        ----------
        :param X: array-like, shape (n_samples, 2)
            Input paired data (output of GetMatches transformer.

        Returns
        -------
        :returns Xt: array-like, shape (n_samples,)
            The transformed data: array of triples like (word_id, word_id, cosine_sim)
        """
        n_samples = len(X)
        Xt = np.zeros(n_samples, dtype=object)
        s_id = 0
        for sample in X:
            r = np.array(sample)
            ids= np.unique(r[:,1])
            Xt[s_id] = _solve_duplictes(sample, ids, th=0.01)
            s_id += 1
        return Xt

class AvgPOSCombiner(BaseEstimator, TransformerMixin):
    """Compute the average of word similarity of the selected pairs."""

    def fit(self, X, y=None):
        """(Do nothing).

        Parameters
        ----------
        :param X: array-like, shape (n_samples,)
            Input data 

        Returns
        -------
        :returns: self
        """
        return self

    def transform(self, X):
        """Compute the average of word similarity of the selected pairs.

        To be added.

        Parameters
        ----------
        :param X: array-like, shape (n_samples, 2)
            Input paired data (output of GetMatches transformer.

        Returns
        -------
        :returns Xt: array-like, shape (n_samples,)
            The transformed data: array of feature values
        """
        n_samples = len(X)
        Xt = np.zeros(n_samples, dtype=object)
        s_id = 0
        for sample in X:
            if len(sample) == 0:
                Xt[s_id] = 0.0
            else:
                r = np.array(sample)
                sim_vals = np.unique(r[0][:,2])
                Xt[s_id] = np.average(sim_vals)
            s_id += 1
        return Xt.reshape(n_samples, 1)

class NumCombiner(BaseEstimator, TransformerMixin):
    """to be added."""

    def fit(self, X, y=None):
        """(Do nothing).

        Parameters
        ----------
        :param X: array-like, shape (n_samples,)
            Input data 

        Returns
        -------
        :returns: self
        """
        return self

    def transform(self, X):
        """Compute the normalized absolute difference of the pairs.

        Compute the normalized absolute difference of the summation of 
        the numbers found in each pair element.

        Parameters
        ----------
        :param X: array-like, shape (n_samples, 2)
            Input paired data (output of num transformer.

        Returns
        -------
        :returns Xt: array-like, shape (n_samples,)
            The transformed data: array of feature values
        """
        n_samples = len(X)
        Xt = np.zeros(n_samples, dtype='float32')
        s_id = 0
        for sample in X:
            x1, x2 = sample
            if x1[0] == ' ' or x2[0] == ' ':
                Xt[s_id] = 1.0
            else:
                x1sum = np.sum(x1)
                x2sum = np.sum(x2)
                if x1sum+x2sum == 0.0:
                    Xt[s_id] == 0.0
                else:
                    Xt[s_id] = np.abs(x1sum-x2sum) / (x1sum+x2sum)
            s_id += 1
        return Xt.reshape(n_samples, 1)
