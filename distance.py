# coding: utf-8

# Author: Hussein AL-NATSHEH <hussein.al-natsheh@ish-lyon.cnrs.fr>
# License: BSD 3 clause
# 2016

import pandas as pd
import argparse
import numpy as np
import pickle

from utils import get_text
from utils import get_words
from utils import get_nouns
from utils import get_proper_nouns
from utils import get_pronouns
from utils import get_verbs
from utils import get_auxiliary_verbs
from utils import get_adjectives
from utils import get_adverbs
from utils import get_numbers
from utils import get_punctuation
from utils import get_particle
from utils import get_determiner
from utils import get_interjection
from utils import get_coordinating_conjunction
from utils import get_symbol
from utils import group_by_sentence
from utils import FuncTransformer
from utils import Shaper
from utils import load_dataset
from utils import load_glove
from utils import PairCosine
from utils import SmallerOtherParing
from utils import RefGroupPairCosine
from utils import GetMatches
from utils import SolveDuplicate
from utils import AvgPOSCombiner
from utils import NumCombiner
from utils import to_numeric
from utils import sts_score

from digify import replace_spelled_numbers

from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoLarsCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.grid_search import GridSearchCV

from beard.similarity import AbsoluteDifference
from beard.similarity import CosineSimilarity
from beard.similarity import PairTransformer
from beard.similarity import StringDistance
from beard.similarity import EstimatorTransformer


def _define_global(glove_file):
    global glove6b300d
    glove6b300d = load_glove(glove_file, verbose=1)

from polyglot.text import Text
import numpy as np

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

class PairGloveTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        n_samples = len(X)
        Xt = np.zeros(n_samples, dtype=object)
        s_id = 0
        for sample in X:
            lst = []
            for tup in sample:
                w1, w2 = tup
                w1_id, w1_text = w1
                w2_id, w2_text = w2
                w1_vec = _word2glove(w1_text)
                w2_vec = _word2glove(w2_text)
                lst.append(((w1_id, w1_vec), (w2_id, w2_vec)))
            Xt[s_id] = lst
            s_id += 1
        return Xt

def _build_distance_estimator(X, y, verbose=1):

    """Build a vector reprensation of a pair of signatures."""
    transformer = FeatureUnion([
        ("nouns_glove", Pipeline(steps=[
        	('pairtransformer', PairTransformer(element_transformer=
                FuncTransformer(dtype=None, func=get_nouns),
        groupby=None)), 
        	('sop', SmallerOtherParing()),
            ('pgt', PairGloveTransformer()),
            ('rgpc', RefGroupPairCosine()),
            ('gm', GetMatches()),
            ('sd', SolveDuplicate()),
            ('ac', AvgPOSCombiner()),
            ])),
        ("get_proper_nouns", Pipeline(steps=[
            ('pairtransformer', PairTransformer(element_transformer=
                FuncTransformer(dtype=None, func=get_proper_nouns),
        groupby=None)), 
            ('sop', SmallerOtherParing()),
            ('pgt', PairGloveTransformer()),
            ('rgpc', RefGroupPairCosine()),
            ('gm', GetMatches()),
            ('sd', SolveDuplicate()),
            ('ac', AvgPOSCombiner()),
            ])),
        ("get_pronouns", Pipeline(steps=[
            ('pairtransformer', PairTransformer(element_transformer=
                FuncTransformer(dtype=None, func=get_pronouns),
        groupby=None)), 
            ('sop', SmallerOtherParing()),
            ('pgt', PairGloveTransformer()),
            ('rgpc', RefGroupPairCosine()),
            ('gm', GetMatches()),
            ('sd', SolveDuplicate()),
            ('ac', AvgPOSCombiner()),
            ])),
        ("get_verbs", Pipeline(steps=[
            ('pairtransformer', PairTransformer(element_transformer=
                FuncTransformer(dtype=None, func=get_verbs),
        groupby=None)), 
            ('sop', SmallerOtherParing()),
            ('pgt', PairGloveTransformer()),
            ('rgpc', RefGroupPairCosine()),
            ('gm', GetMatches()),
            ('sd', SolveDuplicate()),
            ('ac', AvgPOSCombiner()),
            ])),
        ("get_words", Pipeline(steps=[
            ('pairtransformer', PairTransformer(element_transformer=
                FuncTransformer(dtype=None, func=get_words),
        groupby=None)), 
            ('sop', SmallerOtherParing()),
            ('pgt', PairGloveTransformer()),
            ('rgpc', RefGroupPairCosine()),
            ('gm', GetMatches()),
            ('sd', SolveDuplicate()),
            ('ac', AvgPOSCombiner()),
            ])),
        ("get_auxiliary_verbs", Pipeline(steps=[
            ('pairtransformer', PairTransformer(element_transformer=
                FuncTransformer(dtype=None, func=get_auxiliary_verbs),
        groupby=None)), 
            ('sop', SmallerOtherParing()),
            ('pgt', PairGloveTransformer()),
            ('rgpc', RefGroupPairCosine()),
            ('gm', GetMatches()),
            ('sd', SolveDuplicate()),
            ('ac', AvgPOSCombiner()),
            ])),
        ("adjectives_glove", Pipeline(steps=[
            ('pairtransformer', PairTransformer(element_transformer=
                FuncTransformer(dtype=None, func=get_adjectives),
        groupby=None)), 
            ('sop', SmallerOtherParing()),
            ('pgt', PairGloveTransformer()),
            ('rgpc', RefGroupPairCosine()),
            ('gm', GetMatches()),
            ('sd', SolveDuplicate()),
            ('ac', AvgPOSCombiner()),
            ])),
        ("adverbs_glove", Pipeline(steps=[
            ('pairtransformer', PairTransformer(element_transformer=
                FuncTransformer(dtype=None, func=get_adverbs),
        groupby=None)), 
            ('sop', SmallerOtherParing()),
            ('pgt', PairGloveTransformer()),
            ('rgpc', RefGroupPairCosine()),
            ('gm', GetMatches()),
            ('sd', SolveDuplicate()),
            ('ac', AvgPOSCombiner()),
            ])),
        ("get_punctuation", Pipeline(steps=[
            ('pairtransformer', PairTransformer(element_transformer=
                FuncTransformer(dtype=None, func=get_punctuation),
        groupby=None)), 
            ('sop', SmallerOtherParing()),
            ('pgt', PairGloveTransformer()),
            ('rgpc', RefGroupPairCosine()),
            ('gm', GetMatches()),
            ('sd', SolveDuplicate()),
            ('ac', AvgPOSCombiner()),
            ])),
        ("get_particle", Pipeline(steps=[
            ('pairtransformer', PairTransformer(element_transformer=
                FuncTransformer(dtype=None, func=get_particle),
        groupby=None)), 
            ('sop', SmallerOtherParing()),
            ('pgt', PairGloveTransformer()),
            ('rgpc', RefGroupPairCosine()),
            ('gm', GetMatches()),
            ('sd', SolveDuplicate()),
            ('ac', AvgPOSCombiner()),
            ])),
        ("get_determiner", Pipeline(steps=[
            ('pairtransformer', PairTransformer(element_transformer=
                FuncTransformer(dtype=None, func=get_determiner),
        groupby=None)), 
            ('sop', SmallerOtherParing()),
            ('pgt', PairGloveTransformer()),
            ('rgpc', RefGroupPairCosine()),
            ('gm', GetMatches()),
            ('sd', SolveDuplicate()),
            ('ac', AvgPOSCombiner()),
            ])),
        ("get_interjection", Pipeline(steps=[
            ('pairtransformer', PairTransformer(element_transformer=
                FuncTransformer(dtype=None, func=get_interjection),
        groupby=None)), 
            ('sop', SmallerOtherParing()),
            ('pgt', PairGloveTransformer()),
            ('rgpc', RefGroupPairCosine()),
            ('gm', GetMatches()),
            ('sd', SolveDuplicate()),
            ('ac', AvgPOSCombiner()),
            ])),
        ("get_coordinating_conjunction", Pipeline(steps=[
            ('pairtransformer', PairTransformer(element_transformer=
                FuncTransformer(dtype=None, func=get_coordinating_conjunction),
        groupby=None)), 
            ('sop', SmallerOtherParing()),
            ('pgt', PairGloveTransformer()),
            ('rgpc', RefGroupPairCosine()),
            ('gm', GetMatches()),
            ('sd', SolveDuplicate()),
            ('ac', AvgPOSCombiner()),
            ])),
        ("get_symbol", Pipeline(steps=[
            ('pairtransformer', PairTransformer(element_transformer=
                FuncTransformer(dtype=None, func=get_symbol),
        groupby=None)), 
            ('sop', SmallerOtherParing()),
            ('pgt', PairGloveTransformer()),
            ('rgpc', RefGroupPairCosine()),
            ('gm', GetMatches()),
            ('sd', SolveDuplicate()),
            ('ac', AvgPOSCombiner()),
            ])),
        ("sent_tfidf", Pipeline([
            ("pairs", PairTransformer(element_transformer=Pipeline([
                ("1st_verb", FuncTransformer(func=get_text)),
                ("shaper", Shaper(newshape=(-1,))),
                ("tf-idf", TfidfVectorizer(analyzer="char_wb",
                                           ngram_range=(2, 4),
                                           dtype=np.float32,
                                           decode_error="replace",
                                           stop_words="english"))
            ]))),
            ("combiner", CosineSimilarity())
        ])),
        ("sent_len_diff", Pipeline(steps=[
            ('pairtransformer', PairTransformer(element_transformer=
                FuncTransformer(dtype=None, func=len),
        groupby=None)),
            ('abs_diff', AbsoluteDifference()),
            ])),
        ("1st_num_diff", Pipeline(steps=[
            ('pairtransformer', PairTransformer(element_transformer= Pipeline([
                        ("rsn", FuncTransformer(func=replace_spelled_numbers)),
                        ("get_num", FuncTransformer(func=get_numbers)),
                        ("to_num", FuncTransformer(func=to_numeric)),
        ]),groupby=None)),
            ('1st_nm_comb', NumCombiner()),
            ])),
    ])

    # Train a classifier on these vectors
    #classifier = RandomForestRegressor(n_jobs=-1, max_depth=9, n_estimators=1800)
    classifier = LassoLarsCV(cv=5, max_iter=1000, n_jobs=-1)

    # Return the whole pipeline
    estimator = Pipeline([("transformer", transformer),
                          ("classifier", classifier)]).fit(X, y)

    return estimator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_set", default='data/sts_gs_all.csv', type=str)
    parser.add_argument("--training_set", default='data/sts_except_2015_gs.csv', type=str)
    parser.add_argument("--test_set_headlines", default='data/sts_2015_headlines.csv', type=str)
    parser.add_argument("--test_set_images", default='data/sts_2015_images.csv', type=str)
    parser.add_argument("--test_set_answers_students", default='data/sts_2015_answers-students.csv', type=str)
    parser.add_argument("--verbose", default=1, type=int)
    parser.add_argument("--evaluate", default=1, type=int)
    parser.add_argument("--glovefile", default='data/glove.6B.300d.txt', type=str)
    args = parser.parse_args()

    _define_global(args.glovefile)

    if args.evaluate:

        X, y = load_dataset (args.training_set, args.verbose)
      
        distance_estimator = _build_distance_estimator(
            X, y, verbose=1)

        pickle.dump(distance_estimator,
                    open("traning_distance_model.pickle", "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)
        
        score = dict()
        X_test, y_test = load_dataset(args.test_set_headlines, verbose=1)
        score['headlines_score'] = sts_score(distance_estimator,X_test, y_test)
        X_test, y_test = load_dataset(args.test_set_images, verbose=1)
        score['images_score'] = sts_score(distance_estimator,X_test, y_test)
        X_test, y_test = load_dataset(args.test_set_answers_students, verbose=1)
        score['answers_students_score'] = sts_score(distance_estimator,X_test, y_test)

        if args.verbose == 1:
            print score

        pickle.dump(score,
                open("score.pickle", "wb"),
                protocol=pickle.HIGHEST_PROTOCOL)

    else:
        X, y = load_dataset (args.data_set, args.verbose)
      
        distance_estimator = _build_distance_estimator(
            X, y, verbose=1)

        pickle.dump(distance_estimator,
                    open("distance_model.pickle", "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)
