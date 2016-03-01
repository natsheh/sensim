# coding: utf-8

# Author: Hussein AL-NATSHEH <hussein.al-natsheh@ish-lyon.cnrs.fr>
# License: BSD 3 clause
# 2016

import argparse
import numpy as np
import pickle

from polyglot.base import TextFile
from polyglot.text import Text

from utils import load_glove

def _define_global():
    global glove6b300d
    glove6b300d = load_glove('data/glove.6B.300d.tar.gz', verbose=0)

def _word2glove(word):
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
    word = word.lower()
    if word not in glove6b300d.index:
        return np.zeros(300, dtype=float, order='C')
    else:
        return np.array(glove6b300d.loc[word])

def _get_matches(r, th):
    matches = []
    for i in np.unique(r[:,0]):
        s_slice = r[r[:,0]==i]
        max_match = np.argmax(s_slice[:,2])
        if np.amax(s_slice[:,2]) > th:
            matches.append(list(s_slice[max_match]))
    return np.array(matches)

def _sort_arr(arr, axis=0):
    return arr[arr[:,axis].argsort()]

def _solve_duplictes(mk, ids):
    res = []
    for id in ids:
        chk_dup = mk[mk[:,1]==id]
        if len(chk_dup) > 1:
            mn = np.argmin(chk_dup[:,2])
            chk_dup = np.delete(chk_dup, mn, axis=0)
        if chk_dup.shape[0] > 0:
            res.append(chk_dup[0])
    return _sort_arr(np.array(res))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimator", default='distance_model.pickle', type=str)
    parser.add_argument("--doc1", default='data/docs/Retrograde.txt', type=str)
    parser.add_argument("--doc2", default='data/docs/Dream.txt', type=str)
    parser.add_argument("--threshold", default=2.5, type=float)
    args = parser.parse_args()

    doc1 = TextFile(args.doc1).read()
    doc2 = TextFile(args.doc2).read()

    text1 = Text(doc1)
    text2 = Text(doc2)

    text1.language = 'en'
    text2.language = 'en'
    
    sentences1 = text1.sentences
    sentences2 = text2.sentences

    if len(sentences1) < len(sentences2):
        smaller = sentences1
        other = sentences2
    else:
        smaller = sentences2
        other = sentences1

    n_pairs = len(sentences1) * len(sentences2)

    X = np.zeros(shape=(n_pairs,2), dtype=object)
    hash_tbl = np.zeros(shape=(n_pairs,2), dtype=object)

    smaller_dict = {}
    s_id = 0
    for s1 in smaller:
        s_id += 1
        smaller_dict[s_id] = s1.string


    o_id = 0
    other_dict = {}
    for s2 in other:
            o_id += 1
            other_dict[o_id] = s2.string
    k = 0
    for i in smaller_dict.keys():
        for j in other_dict.keys():
            hash_tbl[k] = [i, j]
            k += 1

    i= 0
    for s1 in smaller:
        for s2 in other:
            X[i] = [s1.string, s2.string]
            i += 1

    _define_global()
    
    estimator = pickle.load(open(args.estimator,"r"))
    y = estimator.predict(X)

    r = np.column_stack((hash_tbl,y))
    matches_keys = _get_matches(r, args.threshold)
    matches_keys_resolved = _solve_duplictes(matches_keys, other_dict.keys())
    matches_sentences = []
    for a, b, s in matches_keys_resolved:
        matches_sentences.append((smaller_dict[a], other_dict[b], ("%.2f" % s)))
    
    print '------------------------------------'
    print ('number of matches: ', len(matches_keys_resolved))
    print ('number of max_matches: ', len(smaller_dict.keys()))
    print ('estimated doc_relatedness: ', 5 *(float(len(matches_keys_resolved))/len(smaller_dict.keys())), 'out of 5.0')
    print 'matches_sentences'
    print '------------------------------------'
    for pair in matches_sentences:
        print ('Sent1: ',pair[0])
        print ('Sent2: ',pair[1])
        print ('Sent_relatedness score: ',pair[2])
        print '------------------------------------'
