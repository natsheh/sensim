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
#from utils import word2glove
from utils import FuncTransformer
from utils import Shaper
from utils import load_dataset
from utils import load_glove

from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

from beard.similarity import AbsoluteDifference
from beard.similarity import CosineSimilarity
from beard.similarity import PairTransformer
from beard.similarity import StringDistance
from beard.similarity import EstimatorTransformer

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.metrics.pairwise import cosine_similarity
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

def _define_global():
    global glove6b300d
    glove6b300d = load_glove('data/glove.6B.300d.tar.gz', verbose=1)

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

def get_pos(s):
    """Get dictionary of list POS_tags words from the sentence.

    Parameters
    ----------
    :param s: string
        Sentence

    Returns
    -------
    :returns: Dictionay
        Dictionary of list POS_tags words
    """
    def get_pos(pos, pos_codes):
        for w in text.words:
            if w.pos_tag == pos_code:
                pos.append(w)
        if len(pos) > 0:
            return pos
        else:
            return [' ']
    text = Text(s)
    text.language = 'en'
    text.pos_tags
    POS = {}
    pos_lst = []
    adjectives = []
    pos_lst.append((adjectives, 'ADJ'))
    adpositions = []
    pos_lst.append((adpositions, 'ADP'))
    adverbs = []
    pos_lst.append((adverbs, 'ADV'))
    auxiliary_verbs = []
    pos_lst.append((auxiliary_verbs, 'AUX'))
    coordinating_conjunctions = []
    pos_lst.append((coordinating_conjunctions, 'CONJ'))
    determiners = []
    pos_lst.append((determiners, 'DET'))
    interjections = []
    pos_lst.append((interjections, 'INTJ'))
    nouns = []
    pos_lst.append((nouns, 'NOUN'))
    numerals = []
    pos_lst.append((numerals, 'NUM'))
    particles = []
    pos_lst.append((particles, 'PART'))
    pronouns = []
    pos_lst.append((pronouns, 'PRON'))
    proper_nouns = []
    pos_lst.append((proper_nouns, 'PROPN'))
    punctuations = []
    pos_lst.append((punctuations, 'PUNCT'))
    subordinating_conjunctions = []
    pos_lst.append((subordinating_conjunctions, 'SCONJ'))
    symbols = []
    pos_lst.append((symbols, 'SYM'))
    verbs = []
    pos_lst.append((verbs, 'VERB'))
    others = []
    pos_lst.append((others, 'X'))
    for pos, pos_code in pos_lst:
        POS[pos_code] = get_pos(pos, pos_code)
    return POS

def get_verbs(s):
    """Get list of verbs from the sentence.

    Parameters
    ----------
    :param s: string
        Sentence

    Returns
    -------
    :returns: list of strings
        list of verbs
    """
    return (get_pos(s)['VERB'])

def _get_1st_verb(s):
    """Get first verb from the sentence.

    Parameters
    ----------
    :param s: string
        Sentence

    Returns
    -------
    :returns: string
        first verb
    """
    verbs = get_verbs(s)
    word = verbs[0]
    return _word2glove(word)

def _build_distance_estimator(X, y, verbose=1):

    """Build a vector reprensation of a pair of signatures."""
    transformer = FeatureUnion([
        ("1st_verb_glove", Pipeline(steps=[
        	('pairtransformer', PairTransformer(element_transformer=Pipeline(steps=[
        		('functransformer-1', FuncTransformer(dtype=None, func=get_1st_verb)),
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

    classifier = RandomForestRegressor(n_estimators=100,
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
    parser.add_argument("--glovefile", default='data/glove.6B.300d.tar.gz', type=str)
    args = parser.parse_args()

    X, y = load_dataset (args.dataset, args.verbose)

    _define_global()
    
    distance_estimator = _build_distance_estimator(
        X, y, verbose=1)

    pickle.dump(distance_estimator,
                open("distance_model.pickle", "wb"),
                protocol=pickle.HIGHEST_PROTOCOL)
