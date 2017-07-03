# coding: utf-8

# Author: Hussein AL-NATSHEH <hussein.al-natsheh@cnrs.fr>
# License: BSD 3 clause
# 2016, 2017

import argparse
import pickle
import spacy
import numpy as np


global en_parser
en_parser = spacy.load('en')

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

def _read_doc_file(file_path):
    s = open(file_path).read()
    s = unicode(s, 'utf-8')
    return en_parser(s)

def _sent_tokenizer(spacy_doc):
    sentences = []
    for sent in spacy_doc.sents:
        sentences.append(sent.orth_)
    return sentences

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimator", default='distance_model.pickle', type=str)
    parser.add_argument("--doc1", default='data/docs/Dream.txt', type=str)
    parser.add_argument("--doc2", default='data/docs/Double_Entry.txt', type=str)
    parser.add_argument("--threshold", default=2, type=float)
    parser.add_argument("--solve_dup", default=1, type=int)
    parser.add_argument("--glovefile", default='data/glove.6B.300d.txt', type=str)
    args = parser.parse_args()

    doc1 = _read_doc_file(args.doc1)
    doc2 = _read_doc_file(args.doc2)

    sentences1 = _sent_tokenizer(doc1)
    sentences2 = _sent_tokenizer(doc2)

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
        smaller_dict[s_id] = s1


    o_id = 0
    other_dict = {}
    for s2 in other:
            o_id += 1
            other_dict[o_id] = s2
    k = 0
    for i in smaller_dict.keys():
        for j in other_dict.keys():
            hash_tbl[k] = [i, j]
            k += 1

    i= 0
    for s1 in smaller:
        for s2 in other:
            s1 = s1.encode('ascii', 'ignore')
            s2 = s2.encode('ascii', 'ignore')
            X[i] = [str(s1), str(s2)]
            i += 1
    X = np.array(X, dtype=np.object)
    
    estimator = pickle.load(open(args.estimator,"rb"))
    y = estimator.predict(X)

    r = np.column_stack((hash_tbl,y))
    matches_keys = _get_matches(r, args.threshold)
    if args.solve_dup == 1 and len(matches_keys) > 0:
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
    else:
        matches_sentences = []
        for a, b, s in matches_keys:
            matches_sentences.append((smaller_dict[a], other_dict[b], ("%.2f" % s)))
        
        print '------------------------------------'
        print ('number of matches: ', len(matches_keys))
        print ('number of max_matches: ', len(smaller_dict.keys()))
        print ('estimated doc_relatedness: ', 5 *(float(len(matches_keys))/len(smaller_dict.keys())), 'out of 5.0')
        print 'matches_sentences'
        print '------------------------------------'
        for pair in matches_sentences:
            print ('Sent1: ',pair[0])
            print ('Sent2: ',pair[1])
            print ('Sent_relatedness score: ',pair[2])
            print '------------------------------------'
