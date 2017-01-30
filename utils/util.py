# -*- coding: utf-8 -*-
#
# This file is part of sensim

"""Helpers for sentence semantic similarity model.

.. Author:: Hussein AL-NATSHEH <hussein.al-natsheh@ish-lyon.cnrs.fr>

"""
from scipy.stats import pearsonr
import numpy as np
import json

domaim_class = {'FNWN': 'DEF',
                'MSRpar': 'PARA',
                'MSRvid': 'IMG',
                'OnWN': 'DEF',
                'SMT': 'NEWS',
                'SMTeuroparl': 'NEWS',
                'answer-answer': 'QA',
                'answers-forums': 'QA',
                'answers-students': 'QA',
                'belief': 'PARA',
                'deft-forum': 'QA',
                'deft-news': 'NEWS',
                'headlines': 'NEWS',
                'images': 'IMG',
                'plagiarism': 'PARA',
                'postediting': 'PARA',
                'question-question': 'QA',
                'surprise.OnWN': 'DEF',
                'surprise.SMTnews': 'NEWS',
                'tweet-news': 'NEWS'}

class_numeric_value = {'DEF': 1,
                'PARA': 2,
                'IMG': 3,
                'NEWS': 4,
                'QA': 5}

domain_class_list = np.unique(domaim_class.values())
domain_class_vec_size = len(domain_class_list)

def get_domain_class(s):
    return domaim_class[s]

def domain_class_numeric_value(s):
    return class_numeric_value[s]

def domain_class2vec(s):
    class_vec = np.zeros(domain_class_vec_size, dtype=np.int)
    class_vec[np.where(domain_class_list== s)] = 1
    return class_vec

def get_text(s):
    return s

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

def group_by_sentence(r):
    """Grouping function for ``PairTransformer``.

    Parameters
    ----------
    :param r: iterable
        sentence in a singleton.

    Returns
    -------
    :returns: string
        Sentence id
    """
    return r[0]["sentence_id"]

def contractionsEN():
    # copywrite source: https://github.com/cipriantruica/CATS/blob/master/cats/nlplib/static.py
    # slightly modified dict than the source
    contractions_en2 = { 
    "'s": " is",
    "'ve": " have",
    "'d": " had",
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he has",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how has",
    "i'd": "i had",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it had",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that had",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when has",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where has",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who has",
    "who've": "who have",
    "why's": "why has",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
    }

    contractions_en = {}
    for key in contractions_en2.iterkeys():
        contractions_en[key.capitalize()] = contractions_en2[key].capitalize()
        contractions_en[key] = contractions_en2[key]
    return contractions_en

def de_contraction(word):
    contraction_dict = contractionsEN()
    if word in contraction_dict.keys():
        return contraction_dict[word]
    else:
        return word
