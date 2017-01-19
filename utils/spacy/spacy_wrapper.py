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

def spacy_tokens(s):
    text = en_parser(unicode(s, 'utf-8'))
    tokens = np.arange(len(text),dtype=np.object)
    i = 0
    for token in text:
        tokens[i] = token.orth_
        i += 1
    return tokens

def non_empty(l):
    if len(l) < 1:
        l.append(' ')
    return l
def spacy_pos(s):
    text = en_parser(unicode(s, 'utf-8'))
    adj = []
    adp = []
    adv = []
    aux = []
    conj = []
    det = []
    intj = []
    noun = []
    num = []
    part = []
    pron = []
    propn = []
    punct = []
    sconj = []
    sym = []
    verb = []
    x = []
    eol = []
    space = []
    for token in text:
        if token.pos_ == 'ADJ':
            adj.append(token.orth_)
        if token.pos_ == 'ADP':
            adp.append(token.orth_)
        if token.pos_ == 'ADV':
            adv.append(token.orth_)
        if token.pos_ == 'AUX':
            aux.append(token.orth_)
        if token.pos_ == 'CONJ':
            conj.append(token.orth_)
        if token.pos_ == 'DET':
            det.append(token.orth_)
        if token.pos_ == 'INTJ':
            intj.append(token.orth_)
        if token.pos_ == 'NOUN':
            noun.append(token.orth_)
        if token.pos_ == 'NUM':
            num.append(token.orth_)
        if token.pos_ == 'PART':
            part.append(token.orth_)
        if token.pos_ == 'PRON':
            pron.append(token.orth_)
        if token.pos_ == 'PROPN':
            propn.append(token.orth_)
        if token.pos_ == 'PUNCT':
            punct.append(token.orth_)
        if token.pos_ == 'SCONJ':
            sconj.append(token.orth_)
        if token.pos_ == 'SYM':
            sym.append(token.orth_)
        if token.pos_ == 'VERB':
            verb.append(token.orth_)
        if token.pos_ == 'X':
            x.append(token.orth_)
        if token.pos_ == 'EOL':
            eol.append(token.orth_)
        if token.pos_ == 'SPACE':
            space.append(token.orth_)

    return non_empty(adj), non_empty(adp), non_empty(adv), non_empty(aux), non_empty(conj), non_empty(det), non_empty(intj), non_empty(noun), non_empty(num), non_empty(part), non_empty(pron), non_empty(propn), non_empty(punct), non_empty(sconj), non_empty(sym), non_empty(verb), non_empty(x), non_empty(eol), non_empty(space)

def spacy_adj(s):
    adj, _, _, _, _, _, _, _, _, _, _, _ , _, _, _, _, _, _, _ = spacy_pos(s)
    return adj

def spacy_adp(s):
    _, adp, _, _, _, _, _, _, _, _, _, _ , _, _, _, _, _, _, _ = spacy_pos(s)
    return adp

def spacy_adv(s):
    _, _, adv, _, _, _, _, _, _, _, _, _ , _, _, _, _, _, _, _ = spacy_pos(s)
    return adv

def spacy_aux(s):
    _, _, _, aux, _, _, _, _, _, _, _, _ , _, _, _, _, _, _, _ = spacy_pos(s)
    return aux

def spacy_conj(s):
    _, _, _, _, conj, _, _, _, _, _, _, _ , _, _, _, _, _, _, _ = spacy_pos(s)
    return conj

def spacy_det(s):
    _, _, _, _, _, det, _, _, _, _, _, _ , _, _, _, _, _, _, _ = spacy_pos(s)
    return det

def spacy_intj(s):
    _, _, _, _, _, _, intj, _, _, _, _, _ , _, _, _, _, _, _, _ = spacy_pos(s)
    return intj

def spacy_noun(s):
    _, _, _, _, _, _, _, noun, _, _, _, _ , _, _, _, _, _, _, _ = spacy_pos(s)
    return noun

def spacy_num(s):
    _, _, _, _, _, _, _, _, num, _, _, _ , _, _, _, _, _, _, _ = spacy_pos(s)
    return num

def spacy_part(s):
    _, _, _, _, _, _, _, _, _, part, _, _ , _, _, _, _, _, _, _ = spacy_pos(s)
    return part

def spacy_pron(s):
    _, _, _, _, _, _, _, _, _, _, pron, _ , _, _, _, _, _, _, _ = spacy_pos(s)
    return pron

def spacy_propn(s):
    _, _, _, _, _, _, _, _, _, _, _, propn , _, _, _, _, _, _, _ = spacy_pos(s)
    return propn

def spacy_punct(s):
    _, _, _, _, _, _, _, _, _, _, _, _ , punct, _, _, _, _, _, _ = spacy_pos(s)
    return punct

def spacy_sconj(s):
    _, _, _, _, _, _, _, _, _, _, _, _ , _, sconj, _, _, _, _, _ = spacy_pos(s)
    return sconj

def spacy_sym(s):
    _, _, _, _, _, _, _, _, _, _, _, _ , _, _, sym, _, _, _, _ = spacy_pos(s)
    return sym

def spacy_verb(s):
    _, _, _, _, _, _, _, _, _, _, _, _ , _, _, _, verb, _, _, _ = spacy_pos(s)
    return verb

def spacy_x(s):
    _, _, _, _, _, _, _, _, _, _, _, _ , _, _, _, _, x, _, _ = spacy_pos(s)
    return x

def spacy_eol(s):
    _, _, _, _, _, _, _, _, _, _, _, _ , _, _, _, _, _, eol, _ = spacy_pos(s)
    return eol

def spacy_space(s):
    _, _, _, _, _, _, _, _, _, _, _, _ , _, _, _, _, _, _, space = spacy_pos(s)
    return space

def spacy_name_entities(s):
    text = en_parser(unicode(s, 'utf-8'))
    entities = list(text.ents)
    org = []
    loc = []
    per = []
    norp = []
    fac = []
    gpe = []
    product = []
    event = []
    work_of_art = []
    law = []
    language = []
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
    return non_empty(org), non_empty(per), non_empty(loc), non_empty(norp), non_empty(fac), non_empty(gpe), non_empty(product), non_empty(event), non_empty(work_of_art), non_empty(law), non_empty(language)

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

LABELS = {
	u'ENT': u'ENT',
	u'PERSON': u'PERSON',
	u'NORP': u'NORP',
	u'FAC': u'FAC',
	u'ORG': u'ORG',
	u'GPE': u'GPE',
	u'LOC': u'LOC',
	u'LAW': u'LAW',
	u'PRODUCT': u'PRODUCT',
	u'EVENT': u'EVENT',
	u'WORK_OF_ART': u'WORK_OF_ART',
	u'LANGUAGE': u'LANGUAGE',
	u'DATE': u'DATE',
	u'TIME': u'TIME',
	u'PERCENT': u'PERCENT',
	u'MONEY': u'MONEY',
	u'QUANTITY': u'QUANTITY',
	u'ORDINAL': u'ORDINAL',
	u'CARDINAL': u'CARDINAL'
}

# Some of the functions below are re-used from sense2vec:
#						https://github.com/spacy-io/sense2vec/blob/master/bin/merge_text.py
#						Copyright (C) 2016 spaCy GmbH, MIT License 

def represent_word(word):
	if word.like_url:
		return '%%URL|X'
	text = re.sub(r'\s', '_', word.text)
	tag = LABELS.get(word.ent_type_, word.pos_)
	if not tag:
		tag = '?'
	return text + '|' + tag

def transform_doc(doc):
	for ent in doc.ents:
		ent.merge(ent.root.tag_, ent.text, LABELS[ent.label_])
	for np in doc.noun_chunks:
		while len(np) > 1 and np[0].dep_ not in ('advmod', 'amod', 'compound'):
			np = np[1:]
		np.merge(np.root.tag_, np.text, np.root.ent_type_)
	strings = []
	for sent in doc.sents:
		if sent.text.strip():
			strings.append(' '.join(represent_word(w) for w in sent if not w.is_space))
	if strings:
		return '\n'.join(strings) + '\n'
	else:
		return ''

def sense_tokens(s):
	# Takes a string (i.e. sentence) and returns list of senses
	try:
		s = s.encode('utf-8','ignore').decode('utf-8')
	except:
		return [' ']
	return transform_doc(en_parser(s)).lower().split()

