# coding: utf-8

# Author: Hussein AL-NATSHEH <hussein.al-natsheh@ish-lyon.cnrs.fr>
# License: BSD 3 clause
# 2016

import pandas as pd
import argparse
import numpy as np
import pickle

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

from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

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


def _build_distance_estimator(X, y, verbose=1):

    """Build a vector reprensation of a pair of signatures."""
    transformer = FeatureUnion([
        ("1st_verb_glove", Pipeline(steps=[
        	('pairtransformer', PairTransformer(element_transformer=Pipeline(steps=[
        		('functransformer-1', FuncTransformer(dtype=None, func=get_1st_verb)),
        		('functransformer-2', FuncTransformer(dtype=None, func=_word2glove))]),
        groupby=None)), 
        	('paircosine', PairCosine())])),
        
        ("1st_verb_glove", Pipeline(steps=[
        	('pairtransformer', PairTransformer(element_transformer=Pipeline(steps=[
        		('functransformer-1', FuncTransformer(dtype=None, func=get_2nd_verb)),
        		('functransformer-2', FuncTransformer(dtype=None, func=_word2glove))]),
        groupby=None)), 
        	('paircosine', PairCosine())])),
        
        ("1st_verb_glove", Pipeline(steps=[
        	('pairtransformer', PairTransformer(element_transformer=Pipeline(steps=[
        		('functransformer-1', FuncTransformer(dtype=None, func=get_1st_noun)),
        		('functransformer-2', FuncTransformer(dtype=None, func=_word2glove))]),
        groupby=None)), 
        	('paircosine', PairCosine())])),
        
        ("1st_verb_glove", Pipeline(steps=[
        	('pairtransformer', PairTransformer(element_transformer=Pipeline(steps=[
        		('functransformer-1', FuncTransformer(dtype=None, func=get_2nd_noun)),
        		('functransformer-2', FuncTransformer(dtype=None, func=_word2glove))]),
        groupby=None)), 
        	('paircosine', PairCosine())])),
        
        ("1st_verb_glove", Pipeline(steps=[
        	('pairtransformer', PairTransformer(element_transformer=Pipeline(steps=[
        		('functransformer-1', FuncTransformer(dtype=None, func=get_1st_pronoun)),
        		('functransformer-2', FuncTransformer(dtype=None, func=_word2glove))]),
        groupby=None)), 
        	('paircosine', PairCosine())])),
        
        ("1st_verb_glove", Pipeline(steps=[
        	('pairtransformer', PairTransformer(element_transformer=Pipeline(steps=[
        		('functransformer-1', FuncTransformer(dtype=None, func=get_2nd_pronoun)),
        		('functransformer-2', FuncTransformer(dtype=None, func=_word2glove))]),
        groupby=None)), 
        	('paircosine', PairCosine())])),
        
        ("1st_verb_glove", Pipeline(steps=[
        	('pairtransformer', PairTransformer(element_transformer=Pipeline(steps=[
        		('functransformer-1', FuncTransformer(dtype=None, func=get_1st_proper_noun)),
        		('functransformer-2', FuncTransformer(dtype=None, func=_word2glove))]),
        groupby=None)), 
        	('paircosine', PairCosine())])),
        
        ("1st_verb_glove", Pipeline(steps=[
        	('pairtransformer', PairTransformer(element_transformer=Pipeline(steps=[
        		('functransformer-1', FuncTransformer(dtype=None, func=get_2nd_proper_noun)),
        		('functransformer-2', FuncTransformer(dtype=None, func=_word2glove))]),
        groupby=None)), 
        	('paircosine', PairCosine())])),
        
        ("1st_verb_glove", Pipeline(steps=[
        	('pairtransformer', PairTransformer(element_transformer=Pipeline(steps=[
        		('functransformer-1', FuncTransformer(dtype=None, func=get_1st_auxiliary_verb)),
        		('functransformer-2', FuncTransformer(dtype=None, func=_word2glove))]),
        groupby=None)), 
        	('paircosine', PairCosine())])),
        
        ("1st_verb_glove", Pipeline(steps=[
        	('pairtransformer', PairTransformer(element_transformer=Pipeline(steps=[
        		('functransformer-1', FuncTransformer(dtype=None, func=get_2nd_auxiliary_verb)),
        		('functransformer-2', FuncTransformer(dtype=None, func=_word2glove))]),
        groupby=None)), 
        	('paircosine', PairCosine())])),
        
        ("1st_verb_glove", Pipeline(steps=[
        	('pairtransformer', PairTransformer(element_transformer=Pipeline(steps=[
        		('functransformer-1', FuncTransformer(dtype=None, func=get_1st_adjective)),
        		('functransformer-2', FuncTransformer(dtype=None, func=_word2glove))]),
        groupby=None)), 
        	('paircosine', PairCosine())])),
        
        ("1st_verb_glove", Pipeline(steps=[
        	('pairtransformer', PairTransformer(element_transformer=Pipeline(steps=[
        		('functransformer-1', FuncTransformer(dtype=None, func=get_2nd_adjective)),
        		('functransformer-2', FuncTransformer(dtype=None, func=_word2glove))]),
        groupby=None)), 
        	('paircosine', PairCosine())])),
        
        ("1st_verb_glove", Pipeline(steps=[
        	('pairtransformer', PairTransformer(element_transformer=Pipeline(steps=[
        		('functransformer-1', FuncTransformer(dtype=None, func=get_1st_adverb)),
        		('functransformer-2', FuncTransformer(dtype=None, func=_word2glove))]),
        groupby=None)), 
        	('paircosine', PairCosine())])),
        
        ("1st_verb_glove", Pipeline(steps=[
        	('pairtransformer', PairTransformer(element_transformer=Pipeline(steps=[
        		('functransformer-1', FuncTransformer(dtype=None, func=get_2nd_adverb)),
        		('functransformer-2', FuncTransformer(dtype=None, func=_word2glove))]),
        groupby=None)), 
        	('paircosine', PairCosine())])),
        
        ("1st_verb_glove", Pipeline(steps=[
        	('pairtransformer', PairTransformer(element_transformer=Pipeline(steps=[
        		('functransformer-1', FuncTransformer(dtype=None, func=get_symbol)),
        		('functransformer-2', FuncTransformer(dtype=None, func=_word2glove))]),
        groupby=None)), 
        	('paircosine', PairCosine())])),
        
        ("1st_verb_glove", Pipeline(steps=[
        	('pairtransformer', PairTransformer(element_transformer=Pipeline(steps=[
        		('functransformer-1', FuncTransformer(dtype=None, func=get_interjection)),
        		('functransformer-2', FuncTransformer(dtype=None, func=_word2glove))]),
        groupby=None)), 
        	('paircosine', PairCosine())])),
        
        ("1st_verb_glove", Pipeline(steps=[
        	('pairtransformer', PairTransformer(element_transformer=Pipeline(steps=[
        		('functransformer-1', FuncTransformer(dtype=None, func=get_determiner)),
        		('functransformer-2', FuncTransformer(dtype=None, func=_word2glove))]),
        groupby=None)), 
        	('paircosine', PairCosine())])),
        
        ("1st_verb_glove", Pipeline(steps=[
        	('pairtransformer', PairTransformer(element_transformer=Pipeline(steps=[
        		('functransformer-1', FuncTransformer(dtype=None, func=get_particle)),
        		('functransformer-2', FuncTransformer(dtype=None, func=_word2glove))]),
        groupby=None)), 
        	('paircosine', PairCosine())])),
        
        ("1st_verb_glove", Pipeline(steps=[
        	('pairtransformer', PairTransformer(element_transformer=Pipeline(steps=[
        		('functransformer-1', FuncTransformer(dtype=None, func=get_punctuation)),
        		('functransformer-2', FuncTransformer(dtype=None, func=_word2glove))]),
        groupby=None)), 
        	('paircosine', PairCosine())])),
        
        ("1st_noun_glove", Pipeline([
            ("pairs", PairTransformer(element_transformer=Pipeline([
                ("1st_verb", FuncTransformer(func=get_1st_noun)),
                ("shaper", Shaper(newshape=(-1,))),
                ("tf-idf", TfidfVectorizer(analyzer="char_wb",
                                           ngram_range=(2, 4),
                                           dtype=np.float32,
                                           decode_error="replace"))
            ]))),
            ("combiner", CosineSimilarity())
        ])),
    ])

    # Train a classifier on these vectors

    classifier = RandomForestRegressor(n_estimators=200,
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
