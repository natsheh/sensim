# coding: utf-8

# Author: Hussein AL-NATSHEH <hussein.al-natsheh@ish-lyon.cnrs.fr>
# Affiliation: EA 3083 - University of Lyon, USR 3385 - CNRS, France
# Wrapper for sense2vec: Some functions are used from:
#						https://github.com/spacy-io/sense2vec/blob/master/bin/merge_text.py
#						Copyright (C) 2016 spaCy GmbH, MIT License 
# License: BSD 3 clause
# 2016

import re
import spacy as sp
parser = sp.load('en')

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
	return transform_doc(parser(s)).lower().split()
