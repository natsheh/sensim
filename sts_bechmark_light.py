# coding: utf-8

# This is the code used for UdL team at SemEval 2017 STS task EN-EN track
# Author: Hussein AL-NATSHEH <hussein.al-natsheh@cnrs.fr>
# License: BSD 3 clause
# 2016, 2017

import pandas as pd
import argparse
import numpy as np
import pickle

from polyglot.downloader import downloader
downloader.download("embeddings2.en")
downloader.download ("pos2.en")

from utils.polyglot import polyglot_words
from utils.polyglot import polyglot_nouns
from utils.polyglot import polyglot_proper_nouns
from utils.polyglot import polyglot_pronouns
from utils.polyglot import polyglot_verbs
from utils.polyglot import polyglot_auxiliary_verbs
from utils.polyglot import polyglot_adjectives
from utils.polyglot import polyglot_adverbs
from utils.polyglot import polyglot_numbers
from utils.polyglot import polyglot_punctuation
from utils.polyglot import polyglot_particle
from utils.polyglot import polyglot_determiner
from utils.polyglot import polyglot_interjection
from utils.polyglot import polyglot_coordinating_conjunction
from utils.polyglot import polyglot_symbol
from utils.polyglot import polyglot_organizations
from utils.polyglot import polyglot_persons
from utils.polyglot import polyglot_locations
from utils.polyglot import polyglot_adpositions
from utils.polyglot import polyglot_others
from utils.polyglot import polyglot_subordinating_conjunctions
from utils.polyglot import PairPolyglotVecTransformer

from utils.spacy import spacy_organizations
from utils.spacy import spacy_persons
from utils.spacy import spacy_locations
from utils.spacy import spacy_groups
from utils.spacy import spacy_facilities
from utils.spacy import spacy_geo_locations
from utils.spacy import spacy_products
from utils.spacy import spacy_events
from utils.spacy import spacy_work_of_arts
from utils.spacy import spacy_laws
from utils.spacy import spacy_languages
from utils.spacy import PairSpacyVecTransformer
from utils.spacy import spacy_tokens
from utils.spacy import spacy_adj
from utils.spacy import spacy_adp
from utils.spacy import spacy_adv
from utils.spacy import spacy_aux
from utils.spacy import spacy_conj
from utils.spacy import spacy_det
from utils.spacy import spacy_intj
from utils.spacy import spacy_noun
from utils.spacy import spacy_num
from utils.spacy import spacy_part
from utils.spacy import spacy_pron
from utils.spacy import spacy_propn
from utils.spacy import spacy_punct
from utils.spacy import spacy_sconj
from utils.spacy import spacy_sym
from utils.spacy import spacy_verb
from utils.spacy import spacy_x
from utils.spacy import spacy_eol
from utils.spacy import spacy_space

from utils import get_text
from utils import group_by_sentence
from utils import FuncTransformer
from utils import Shaper
from utils import read_tsv
from utils import df_2_dset
from utils import load_dataset
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
from polyglot.mapping import Embedding

def _load_sts_benchmark_dataset(dframe_file):
    dframe = read_tsv(dframe_file)
    dframe["Score"] = np.array(dframe['column_4'], dtype=np.float32)
    X, y = df_2_dset(dframe, sent1_col="column_5", sent2_col="column_6")
    return X, y

def _load_glove(glove_file, verbose=1):
    global glove
    glove = Embedding.from_glove(glove_file)
    if verbose == 2:
        print 'GloVe shape:', glove.shape
        print 'GloVe first 10:', glove.head(n=10)
    elif verbose == 1:
        print 'GloVe shape:', glove.shape
    return glove

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
    if word not in glove.vocabulary:
        word_vec = np.zeros(300, dtype=float, order='C')
        return word_vec.reshape(1, -1)
    else:
        word_vec = np.array(glove[word])
        return word_vec.reshape(1, -1)

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
    elif w2v == 'polyglot':
        PairVecTransformer = PairPolyglotVecTransformer
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
        ("get_nouns", Pipeline(steps=[
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
        ("sent_tfidf_cosine", Pipeline([
            ("pairs", PairTransformer(element_transformer=Pipeline([
                ("get_text", FuncTransformer(func=get_text)),
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
        classifier = LassoLarsCV(cv=5, max_iter=512, n_jobs=-1)
    elif regressor == 'RF':
        classifier = RandomForestRegressor(n_jobs=-1, max_depth=8, n_estimators=500)
    else:
        print('Error passing the regressor type')

    # Return the whole pipeline
    estimator = Pipeline([("transformer", transformer),
                          ("classifier", classifier)]).fit(X, y)

    return estimator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectorization_method", default='spacy', type=str) #glove, spacy, polygolt
    parser.add_argument("--PoS_method", default='polyglot', type=str) #spacy, polyglot
    parser.add_argument("--NER_method", default='spacy', type=str) #spacy, polyglot
    parser.add_argument("--regressor", default='RF', type=str) #lasso, RF
    parser.add_argument("--data_set", default='data/cleaned_2017_all.csv', type=str)
    parser.add_argument("--training_set", default='data/stsbenchmark/sts-train.csv', type=str)
    parser.add_argument("--dev_set", default='data/stsbenchmark/sts-dev.csv', type=str)
    parser.add_argument("--test_set", default='data/stsbenchmark/sts-test.csv', type=str)
    parser.add_argument("--companion_other_set", default='data/stscompanion/sts-other.csv', type=str)
    parser.add_argument("--predict_task", default='data/predict_task.csv', type=str)
    parser.add_argument("--verbose", default=1, type=int)
    parser.add_argument("--evaluate", default=1, type=int)
    parser.add_argument("--training_estimator", default=None, type=str)
    parser.add_argument("--dev_estimator", default=None, type=str)
    parser.add_argument("--estimator", default=None, type=str)
    parser.add_argument("--decimals", default=None, type=int)
    parser.add_argument("--bounded", default=1, type=str)
    parser.add_argument("--glovefile", default='data/glove.6B.300d.txt', type=str)
    args = parser.parse_args()

    w2v = args.vectorization_method
    PoS = args.PoS_method
    NER = args.NER_method
    regressor = args.regressor

    if w2v == 'glove':
        _load_glove(args.glovefile, verbose=args.verbose)

    X_train, y_train = _load_sts_benchmark_dataset(args.training_set)
    X_dev, y_dev = _load_sts_benchmark_dataset(args.dev_set)
    X_test, y_test = _load_sts_benchmark_dataset(args.test_set)
    rest_dframe = read_tsv(args.companion_other_set)
    rest_dframe["Score"] = np.array(rest_dframe['column_3'], dtype=np.float32)
    X_rest, y_rest = df_2_dset(rest_dframe, sent1_col="column_4", sent2_col="column_5")

    if args.evaluate:
        if args.training_estimator is None:
            training_estimator = _build_distance_estimator(
                X_train, y_train, w2v, PoS, NER, regressor, verbose=1)

            pickle.dump(training_estimator,
                        open("traning_distance_model"+regressor+".pickle", "wb"),
                        protocol=pickle.HIGHEST_PROTOCOL)
        else:
            training_estimator = pickle.load(open(args.training_estimator,'rb'))
        
        score = dict()
        score['dev_score'] = sts_score(training_estimator,X_dev, y_dev, args.decimals)

        if args.dev_estimator is None:
            X_train_dev = np.vstack((X_train,X_dev))
            y_train_dev = np.hstack((y_train,y_dev))
            dev_estimator = _build_distance_estimator(
                X_train_dev, y_train_dev, w2v, PoS, NER, regressor, verbose=1)

            pickle.dump(dev_estimator,
                        open("traning_dev_distance_model"+regressor+".pickle", "wb"),
                        protocol=pickle.HIGHEST_PROTOCOL)
        else:
            dev_estimator = pickle.load(open(args.dev_estimator,'rb'))

        score['test_score'] = sts_score(dev_estimator,X_test, y_test, args.decimals)

        if args.verbose == 1:
            print score
            # recent run on default parameters values: 
            #{'test_score': 0.73496312643370554, 'dev_score': 0.79295106912391955}
        pickle.dump(score,
                open("score.pickle", "wb"),
                protocol=pickle.HIGHEST_PROTOCOL)

    else:
        if args.estimator is None:
            X = np.vstack((X_train, X_dev, X_test, X_rest))
            y = np.hstack((y_train, y_dev, y_test, y_rest))
          
            distance_estimator = _build_distance_estimator(
                X, y, w2v, PoS, NER, regressor, verbose=1)

            pickle.dump(distance_estimator,
                        open("distance_model.pickle", "wb"),
                        protocol=pickle.HIGHEST_PROTOCOL)
        else:
            distance_estimator = pickle.load(open(args.estimator, 'rb'))
        X_predict, _ = load_dataset(args.predict_task, verbose=1)
        y_predict = distance_estimator.predict(X_predict)
        prediction = y_predict.reshape(-1,1)
        if args.decimals is not None:
            prediction = np.round(prediction, decimals=args.decimals)
        if args.bounded:
            prediction[np.where(prediction > 5)] = 5
            prediction[np.where(prediction < 0)] = 0
        res = pd.DataFrame()

        results = []
        for r in prediction:
            p = r[0]
            results.append(p)

        res['Score'] = results
        res.to_csv('STS.sys.track5.en-en.txt', index=False, header=False)
        if args.verbose == 1:
            y_predict = pd.read_csv("STS.sys.track5.en-en.txt", header=None)
            y_gs = pd.read_csv("data/STS.gs.track5.en-en.txt", header=None)
            print pearsonr(y_predict, y_gs)[0][0]
