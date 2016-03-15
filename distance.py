# coding: utf-8

# Author: Hussein AL-NATSHEH <hussein.al-natsheh@ish-lyon.cnrs.fr>
# License: BSD 3 clause
# 2016

import pandas as pd
import argparse
import numpy as np
import pickle

from utils import get_text
from utils import get_nouns
from utils import get_proper_nouns
from utils import get_pronouns
from utils import get_verbs
from utils import get_auxiliary_verbs
from utils import get_adjectives
from utils import get_adverbs
from utils import get_numbers
from utils import get_1st_noun
from utils import get_2nd_noun
from utils import get_1st_proper_noun
from utils import get_2nd_proper_noun
from utils import get_1st_pronoun
from utils import get_2nd_pronoun
from utils import get_1st_verb
from utils import get_2nd_verb
from utils import get_1st_auxiliary_verb
from utils import get_2nd_auxiliary_verb
from utils import get_1st_adjective
from utils import get_2nd_adjective
from utils import get_1st_adverb
from utils import get_2nd_adverb
from utils import get_1st_number
from utils import get_2nd_number
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

from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from beard.similarity import AbsoluteDifference
from beard.similarity import CosineSimilarity
from beard.similarity import PairTransformer
from beard.similarity import StringDistance
from beard.similarity import EstimatorTransformer

from digify import replace_spelled_numbers

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
        ("nouns_glove", Pipeline(steps=[
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
        ("nouns_glove", Pipeline(steps=[
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

    classifier = RandomForestRegressor(n_estimators=500,
                                        verbose=verbose,
                                        n_jobs=8)

    # Return the whole pipeline
    estimator = Pipeline([("transformer", transformer),
                          ("classifier", classifier)]).fit(X, y)

    return estimator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='sts', type=str)
    parser.add_argument("--verbose", default=1, type=int)
    parser.add_argument("--glovefile", default='data/glove.6B.300d.txt', type=str)
    args = parser.parse_args()

    X, y = load_dataset (args.dataset, args.verbose)

    _define_global(args.glovefile)
    
    distance_estimator = _build_distance_estimator(
        X, y, verbose=1)

    pickle.dump(distance_estimator,
                open("distance_model.pickle", "wb"),
                protocol=pickle.HIGHEST_PROTOCOL)
