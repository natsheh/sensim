# coding: utf-8

# Author: Hussein AL-NATSHEH <hussein.al-natsheh@ish-lyon.cnrs.fr>
# License: BSD 3 clause
# 2016

import pandas as pd
import argparse
import numpy as np
import pickle

from utils import polyglot_words
from utils import polyglot_nouns
from utils import polyglot_proper_nouns
from utils import polyglot_pronouns
from utils import polyglot_verbs
from utils import polyglot_auxiliary_verbs
from utils import polyglot_adjectives
from utils import polyglot_adverbs
from utils import polyglot_numbers
from utils import polyglot_punctuation
from utils import polyglot_particle
from utils import polyglot_determiner
from utils import polyglot_interjection
from utils import polyglot_coordinating_conjunction
from utils import polyglot_symbol
from utils import polyglot_organizations
from utils import polyglot_persons
from utils import polyglot_locations
from utils import polyglot_adpositions
from utils import polyglot_others
from utils import polyglot_subordinating_conjunctions

from utils import spacy_organizations
from utils import spacy_persons
from utils import spacy_locations
from utils import spacy_groups
from utils import spacy_facilities
from utils import spacy_geo_locations
from utils import spacy_products
from utils import spacy_events
from utils import spacy_work_of_arts
from utils import spacy_laws
from utils import spacy_languages
from utils import PairSpacyVecTransformer
from utils import spacy_tokens
from utils import spacy_adj
from utils import spacy_adp
from utils import spacy_adv
from utils import spacy_aux
from utils import spacy_conj
from utils import spacy_det
from utils import spacy_intj
from utils import spacy_noun
from utils import spacy_num
from utils import spacy_part
from utils import spacy_pron
from utils import spacy_propn
from utils import spacy_punct
from utils import spacy_sconj
from utils import spacy_sym
from utils import spacy_verb
from utils import spacy_x
from utils import spacy_eol
from utils import spacy_space

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


def get_text(s):
    return s

def _define_global(glove_file):
    global glove
    glove = load_glove(glove_file, verbose=1)


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
    if word not in glove.index:
        return np.zeros(300, dtype=float, order='C')
    else:
        return np.array(glove.loc[word])

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

def _build_distance_estimator(X, y, w2v, PoS, NER, regressor, verbose=1):
    """Build a vector reprensation of a pair of signatures."""
    if w2v == 'glove':
        PairVecTransformer = PairGloveTransformer
    elif w2v == 'spacy':
        PairVecTransformer = PairSpacyVecTransformer
    else:
        print('error passing w2v argument value')

    if PoS == 'polyglot':
        get_nouns = polyglot_nouns
        get_verbs = polyglot_verbs
        get_words = polyglot_words
        get_particle = polyglot_particle
        get_interjection = polyglot_interjection
        get_symbol = polyglot_symbol
        get_numbers = polyglot_numbers
        get_proper_nouns = polyglot_proper_nouns
        get_pronouns = polyglot_pronouns
        get_auxiliary_verbs = polyglot_auxiliary_verbs
        get_adjectives = polyglot_adjectives
        get_adverbs = polyglot_adverbs
        get_punctuation = polyglot_punctuation
        get_determiner = polyglot_determiner
        get_coordinating_conjunction = polyglot_coordinating_conjunction
        get_adpositions = polyglot_adpositions
        get_others = polyglot_others
        get_subordinating_conjunctions = polyglot_subordinating_conjunctions
    elif PoS == 'spacy':
        get_nouns = spacy_noun
        get_verbs = spacy_verb
        get_words = spacy_tokens
        get_particle = spacy_part
        get_interjection = spacy_intj
        get_symbol = spacy_sym
        get_numbers = spacy_num
        get_proper_nouns = spacy_propn
        get_pronouns = spacy_pron
        get_auxiliary_verbs = spacy_aux
        get_adjectives = spacy_adj
        get_adverbs = spacy_adv
        get_punctuation = spacy_punct
        get_determiner = spacy_det
        get_coordinating_conjunction = spacy_conj
        get_adpositions = spacy_adp
        get_others = spacy_x
        get_subordinating_conjunctions = spacy_sconj
    else:
        print('error passing PoS argument value')

    transformer = FeatureUnion([
        ("nouns_glove", Pipeline(steps=[
        	('pairtransformer', PairTransformer(element_transformer=
                FuncTransformer(dtype=None, func=get_nouns),
        groupby=None)), 
        	('sop', SmallerOtherParing()),
            ('pgt', PairVecTransformer()),
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
            ('pgt', PairVecTransformer()),
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
            ('pgt', PairVecTransformer()),
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
            ('pgt', PairVecTransformer()),
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
            ('pgt', PairVecTransformer()),
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
            ('pgt', PairVecTransformer()),
            ('rgpc', RefGroupPairCosine()),
            ('gm', GetMatches()),
            ('sd', SolveDuplicate()),
            ('ac', AvgPOSCombiner()),
            ])),

        ("num_diff", Pipeline(steps=[
            ('pairtransformer', PairTransformer(element_transformer= Pipeline([
                        ("rsn", FuncTransformer(func=replace_spelled_numbers)),
                        ("get_num", FuncTransformer(func=get_numbers)),
                        ("to_num", FuncTransformer(func=to_numeric)),
        ]),groupby=None)),
            ('1st_nm_comb', NumCombiner()),
            ])),
        ("get_proper_nouns", Pipeline(steps=[
            ('pairtransformer', PairTransformer(element_transformer=
                FuncTransformer(dtype=None, func=get_proper_nouns),
        groupby=None)), 
            ('sop', SmallerOtherParing()),
            ('pgt', PairVecTransformer()),
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
            ('pgt', PairVecTransformer()),
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
            ('pgt', PairVecTransformer()),
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
            ('pgt', PairVecTransformer()),
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
            ('pgt', PairVecTransformer()),
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
            ('pgt', PairVecTransformer()),
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
            ('pgt', PairVecTransformer()),
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
            ('pgt', PairVecTransformer()),
            ('rgpc', RefGroupPairCosine()),
            ('gm', GetMatches()),
            ('sd', SolveDuplicate()),
            ('ac', AvgPOSCombiner()),
            ])),
        ("get_adpositions", Pipeline(steps=[
            ('pairtransformer', PairTransformer(element_transformer=
                FuncTransformer(dtype=None, func=get_adpositions),
        groupby=None)), 
            ('sop', SmallerOtherParing()),
            ('pgt', PairVecTransformer()),
            ('rgpc', RefGroupPairCosine()),
            ('gm', GetMatches()),
            ('sd', SolveDuplicate()),
            ('ac', AvgPOSCombiner()),
            ])),
        ("get_others", Pipeline(steps=[
            ('pairtransformer', PairTransformer(element_transformer=
                FuncTransformer(dtype=None, func=get_others),
        groupby=None)), 
            ('sop', SmallerOtherParing()),
            ('pgt', PairVecTransformer()),
            ('rgpc', RefGroupPairCosine()),
            ('gm', GetMatches()),
            ('sd', SolveDuplicate()),
            ('ac', AvgPOSCombiner()),
            ])),
        ("get_subordinating_conjunctions", Pipeline(steps=[
            ('pairtransformer', PairTransformer(element_transformer=
                FuncTransformer(dtype=None, func=get_subordinating_conjunctions),
        groupby=None)), 
            ('sop', SmallerOtherParing()),
            ('pgt', PairVecTransformer()),
            ('rgpc', RefGroupPairCosine()),
            ('gm', GetMatches()),
            ('sd', SolveDuplicate()),
            ('ac', AvgPOSCombiner()),
            ])),
        ("spacy_eol", Pipeline(steps=[
            ('pairtransformer', PairTransformer(element_transformer=
                FuncTransformer(dtype=None, func=spacy_eol),
        groupby=None)), 
            ('sop', SmallerOtherParing()),
            ('pgt', PairVecTransformer()),
            ('rgpc', RefGroupPairCosine()),
            ('gm', GetMatches()),
            ('sd', SolveDuplicate()),
            ('ac', AvgPOSCombiner()),
            ])),
        ("spacy_space", Pipeline(steps=[
            ('pairtransformer', PairTransformer(element_transformer=
                FuncTransformer(dtype=None, func=spacy_space),
        groupby=None)), 
            ('sop', SmallerOtherParing()),
            ('pgt', PairVecTransformer()),
            ('rgpc', RefGroupPairCosine()),
            ('gm', GetMatches()),
            ('sd', SolveDuplicate()),
            ('ac', AvgPOSCombiner()),
            ])),
        ("spacy_organizations", Pipeline(steps=[
            ('pairtransformer', PairTransformer(element_transformer=
                FuncTransformer(dtype=None, func=spacy_organizations),
        groupby=None)), 
            ('sop', SmallerOtherParing()),
            ('pgt', PairVecTransformer()),
            ('rgpc', RefGroupPairCosine()),
            ('gm', GetMatches()),
            ('sd', SolveDuplicate()),
            ('ac', AvgPOSCombiner()),
            ])),
        ("spacy_persons", Pipeline(steps=[
            ('pairtransformer', PairTransformer(element_transformer=
                FuncTransformer(dtype=None, func=spacy_persons),
        groupby=None)), 
            ('sop', SmallerOtherParing()),
            ('pgt', PairVecTransformer()),
            ('rgpc', RefGroupPairCosine()),
            ('gm', GetMatches()),
            ('sd', SolveDuplicate()),
            ('ac', AvgPOSCombiner()),
            ])),
        ("spacy_locations", Pipeline(steps=[
            ('pairtransformer', PairTransformer(element_transformer=
                FuncTransformer(dtype=None, func=spacy_locations),
        groupby=None)), 
            ('sop', SmallerOtherParing()),
            ('pgt', PairVecTransformer()),
            ('rgpc', RefGroupPairCosine()),
            ('gm', GetMatches()),
            ('sd', SolveDuplicate()),
            ('ac', AvgPOSCombiner()),
            ])),
        ("spacy_groups", Pipeline(steps=[
            ('pairtransformer', PairTransformer(element_transformer=
                FuncTransformer(dtype=None, func=spacy_groups),
        groupby=None)), 
            ('sop', SmallerOtherParing()),
            ('pgt', PairVecTransformer()),
            ('rgpc', RefGroupPairCosine()),
            ('gm', GetMatches()),
            ('sd', SolveDuplicate()),
            ('ac', AvgPOSCombiner()),
            ])),
        ("spacy_facilities", Pipeline(steps=[
            ('pairtransformer', PairTransformer(element_transformer=
                FuncTransformer(dtype=None, func=spacy_facilities),
        groupby=None)), 
            ('sop', SmallerOtherParing()),
            ('pgt', PairVecTransformer()),
            ('rgpc', RefGroupPairCosine()),
            ('gm', GetMatches()),
            ('sd', SolveDuplicate()),
            ('ac', AvgPOSCombiner()),
            ])),
        ("spacy_geo_locations", Pipeline(steps=[
            ('pairtransformer', PairTransformer(element_transformer=
                FuncTransformer(dtype=None, func=spacy_geo_locations),
        groupby=None)), 
            ('sop', SmallerOtherParing()),
            ('pgt', PairVecTransformer()),
            ('rgpc', RefGroupPairCosine()),
            ('gm', GetMatches()),
            ('sd', SolveDuplicate()),
            ('ac', AvgPOSCombiner()),
            ])),
        ("spacy_products", Pipeline(steps=[
            ('pairtransformer', PairTransformer(element_transformer=
                FuncTransformer(dtype=None, func=spacy_products),
        groupby=None)), 
            ('sop', SmallerOtherParing()),
            ('pgt', PairVecTransformer()),
            ('rgpc', RefGroupPairCosine()),
            ('gm', GetMatches()),
            ('sd', SolveDuplicate()),
            ('ac', AvgPOSCombiner()),
            ])),
        ("spacy_events", Pipeline(steps=[
            ('pairtransformer', PairTransformer(element_transformer=
                FuncTransformer(dtype=None, func=spacy_events),
        groupby=None)), 
            ('sop', SmallerOtherParing()),
            ('pgt', PairVecTransformer()),
            ('rgpc', RefGroupPairCosine()),
            ('gm', GetMatches()),
            ('sd', SolveDuplicate()),
            ('ac', AvgPOSCombiner()),
            ])),
        ("spacy_work_of_arts", Pipeline(steps=[
            ('pairtransformer', PairTransformer(element_transformer=
                FuncTransformer(dtype=None, func=spacy_work_of_arts),
        groupby=None)), 
            ('sop', SmallerOtherParing()),
            ('pgt', PairVecTransformer()),
            ('rgpc', RefGroupPairCosine()),
            ('gm', GetMatches()),
            ('sd', SolveDuplicate()),
            ('ac', AvgPOSCombiner()),
            ])),
        ("spacy_laws", Pipeline(steps=[
            ('pairtransformer', PairTransformer(element_transformer=
                FuncTransformer(dtype=None, func=spacy_laws),
        groupby=None)), 
            ('sop', SmallerOtherParing()),
            ('pgt', PairVecTransformer()),
            ('rgpc', RefGroupPairCosine()),
            ('gm', GetMatches()),
            ('sd', SolveDuplicate()),
            ('ac', AvgPOSCombiner()),
            ])),
        ("spacy_languages", Pipeline(steps=[
            ('pairtransformer', PairTransformer(element_transformer=
                FuncTransformer(dtype=None, func=spacy_languages),
        groupby=None)), 
            ('sop', SmallerOtherParing()),
            ('pgt', PairVecTransformer()),
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
    ])

    # Train a classifier on these vectors
    if regressor == 'lasso':
        classifier = LassoLarsCV(cv=10, max_iter=1024, n_jobs=-1)
    elif regressor == 'RF':
        classifier = RandomForestRegressor(n_jobs=-1, max_depth=8, n_estimators=1024)
    else:
        print('Error passing the regressor type')

    # Return the whole pipeline
    estimator = Pipeline([("transformer", transformer),
                          ("classifier", classifier)]).fit(X, y)

    return estimator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectorization_method", default='spacy', type=str)    #glove, spacy, gensim
    parser.add_argument("--PoS_method", default='spacy', type=str)   #spacy, polyglot
    parser.add_argument("--NER_method", default='spacy', type=str)  #spacy, polyglot
    parser.add_argument("--regressor", default='lasso', type=str)   #lasso, RF
    parser.add_argument("--data_set", default='data/sts_gs_all.csv', type=str)
    parser.add_argument("--training_set", default='data/sts_except_2015_gs.csv', type=str)
    parser.add_argument("--test_set_headlines", default='data/sts_2015_headlines.csv', type=str)
    parser.add_argument("--test_set_images", default='data/sts_2015_images.csv', type=str)
    parser.add_argument("--test_set_answers_students", default='data/sts_2015_answers-students.csv', type=str)
    parser.add_argument("--verbose", default=1, type=int)
    parser.add_argument("--evaluate", default=1, type=int)
    parser.add_argument("--glovefile", default='data/glove.6B.300d.txt', type=str)
    args = parser.parse_args()

    w2v = args.vectorization_method
    PoS = args.PoS_method
    NER = args.NER_method
    regressor = args.regressor

    if w2v == 'glove':
        _define_global(args.glovefile)

    if args.evaluate:

        X, y = load_dataset (args.training_set, args.verbose)
      
        distance_estimator = _build_distance_estimator(
            X, y, w2v, PoS, NER, regressor, verbose=1)

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
            X, y, w2v, PoS, NER, regressor, verbose=1)

        pickle.dump(distance_estimator,
                    open("distance_model.pickle", "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)
