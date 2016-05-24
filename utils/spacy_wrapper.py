# coding: utf-8

# Author: Hussein AL-NATSHEH <hussein.al-natsheh@ish-lyon.cnrs.fr>
# Wrapper for SpaCy NER, PoS and Word Vector
# License: BSD 3 clause
# 2016

import numpy as np
import spacy

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

global en_parser
global vocab
en_parser = spacy.load('en')
vocab = en_parser.vocab

def spacy_name_entities(s):
    text = en_parser(unicode(s, 'utf-8'))
    entities = list(text.ents)
    org = list(' ')
    loc = list(' ')
    per = list(' ')
    norp = list(' ')
    fac = list(' ')
    gpe = list(' ')
    product = list(' ')
    event = list(' ')
    work_of_art = list(' ')
    law = list(' ')
    language = list(' ')
    for entity in entities:
        if entity.label_ == 'ORG':
            org.append(entity.orth_)
        if entity.label_ == 'PERSON':
            per.append(entity.orth_)
        if entity.label_ == 'LOC':
            loc.append(entity.orth_)
        if entity.label_ == 'NORP':
            norp.append(entity.orth_)
        if entity.label_ == 'FAC':
            fac.append(entity.orth_)
        if entity.label_ == 'GPE':
            gpe.append(entity.orth_)
        if entity.label_ == 'PRODUCT':
            product.append(entity.orth_)
        if entity.label_ == 'EVENT':
            event.append(entity.orth_)
        if entity.label_ == 'WORK_OF_ART':
            work_of_art.append(entity.orth_)
        if entity.label_ == 'LAW':
            law.append(entity.orth_)
        if entity.label_ == 'LANGUAGE':
            language.append(entity.orth_)
    return org, per, loc, norp, fac, gpe, product, event, work_of_art, law, language

def spacy_organizations(s):
    org, _, _, _, _, _, _, _, _, _, _ = spacy_name_entities(s)
    return org

def spacy_persons(s):
    _, per, _, _, _, _, _, _, _, _, _ = spacy_name_entities(s)
    return per

def spacy_locations(s):
    _, _, loc, _, _, _, _, _, _, _, _ = spacy_name_entities(s)
    return loc

def spacy_groups(s):
    _, _, _, norp, _, _, _, _, _, _, _ = spacy_name_entities(s)
    return norp

def spacy_facilities(s):
    _, _, _, _, fac, _, _, _, _, _, _ = spacy_name_entities(s)
    return fac

def spacy_geo_locations(s):
    _, _, _, _, _, gpe, _, _, _, _, _ = spacy_name_entities(s)
    return gpe

def spacy_products(s):
    _, _, _, _, _, _, product, _, _, _, _ = spacy_name_entities(s)
    return product

def spacy_events(s):
    _, _, _, _, _, _, _, event, _, _, _ = spacy_name_entities(s)
    return event

def spacy_work_of_arts(s):
    _, _, _, _, _, _, _, _, work_of_art, _, _ = spacy_name_entities(s)
    return work_of_art

def spacy_laws(s):
    _, _, _, _, _, _, _, _, _, law, _ = spacy_name_entities(s)
    return law

def spacy_languages(s):
    _, _, _, _, _, _, _, _, _, _, language = spacy_name_entities(s)
    return language

# Get the vector representation of the word using SpaCy
def _spacy_vec(word):
    word = word.encode('ascii','ignore').decode('ascii')
    if word not in vocab:
        return np.zeros(300, dtype=float, order='C').reshape(1, -1)
    else:
        w = vocab[word]
        w_vec = w.vector
        return np.array(w_vec).reshape(1, -1)


class PairSpacyVecTransformer(BaseEstimator, TransformerMixin):
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
                w1_vec = _spacy_vec(w1_text)
                w2_vec = _spacy_vec(w2_text)
                lst.append(((w1_id, w1_vec), (w2_id, w2_vec)))
            Xt[s_id] = lst
            s_id += 1
        return Xt