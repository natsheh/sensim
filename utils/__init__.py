# -*- coding: utf-8 -*-
#
# This file is part of sensim

"""Helpers for sentence semantic similarity model.

.. Author:: Hussein AL-NATSHEH <hussein.al-natsheh@ish-lyon.cnrs.fr>

"""

"""Helper functions."""

from .polyglot_wrapper import polyglot_nouns
from .polyglot_wrapper import polyglot_proper_nouns
from .polyglot_wrapper import polyglot_pronouns
from .polyglot_wrapper import polyglot_verbs
from .polyglot_wrapper import polyglot_auxiliary_verbs
from .polyglot_wrapper import polyglot_adjectives
from .polyglot_wrapper import polyglot_adverbs
from .polyglot_wrapper import polyglot_numbers
from .polyglot_wrapper import polyglot_1st_noun
from .polyglot_wrapper import polyglot_2nd_noun
from .polyglot_wrapper import polyglot_1st_proper_noun
from .polyglot_wrapper import polyglot_2nd_proper_noun
from .polyglot_wrapper import polyglot_1st_pronoun
from .polyglot_wrapper import polyglot_2nd_pronoun
from .polyglot_wrapper import polyglot_1st_verb
from .polyglot_wrapper import polyglot_2nd_verb
from .polyglot_wrapper import polyglot_1st_auxiliary_verb
from .polyglot_wrapper import polyglot_2nd_auxiliary_verb
from .polyglot_wrapper import polyglot_1st_adjective
from .polyglot_wrapper import polyglot_2nd_adjective
from .polyglot_wrapper import polyglot_1st_adverb
from .polyglot_wrapper import polyglot_2nd_adverb
from .polyglot_wrapper import polyglot_1st_number
from .polyglot_wrapper import polyglot_2nd_number
from .polyglot_wrapper import polyglot_punctuation
from .polyglot_wrapper import polyglot_particle
from .polyglot_wrapper import polyglot_determiner
from .polyglot_wrapper import polyglot_interjection
from .polyglot_wrapper import polyglot_coordinating_conjunction
from .polyglot_wrapper import polyglot_symbol
from .polyglot_wrapper import polyglot_adpositions
from .polyglot_wrapper import polyglot_others
from .polyglot_wrapper import polyglot_subordinating_conjunctions
from .polyglot_wrapper import polyglot_words
from .polyglot_wrapper import polyglot_organizations
from .polyglot_wrapper import polyglot_persons
from .polyglot_wrapper import polyglot_locations
from .spacy_wrapper import spacy_organizations
from .spacy_wrapper import spacy_persons
from .spacy_wrapper import spacy_locations
from .spacy_wrapper import spacy_groups
from .spacy_wrapper import spacy_facilities
from .spacy_wrapper import spacy_geo_locations
from .spacy_wrapper import spacy_products
from .spacy_wrapper import spacy_events
from .spacy_wrapper import spacy_work_of_arts
from .spacy_wrapper import spacy_laws
from .spacy_wrapper import spacy_languages
from .spacy_wrapper import PairSpacyVecTransformer
from .spacy_wrapper import spacy_tokens
from .spacy_wrapper import spacy_adj
from .spacy_wrapper import spacy_adp
from .spacy_wrapper import spacy_adv
from .spacy_wrapper import spacy_aux
from .spacy_wrapper import spacy_conj
from .spacy_wrapper import spacy_det
from .spacy_wrapper import spacy_intj
from .spacy_wrapper import spacy_noun
from .spacy_wrapper import spacy_num
from .spacy_wrapper import spacy_part
from .spacy_wrapper import spacy_pron
from .spacy_wrapper import spacy_propn
from .spacy_wrapper import spacy_punct
from .spacy_wrapper import spacy_sconj
from .spacy_wrapper import spacy_sym
from .spacy_wrapper import spacy_verb
from .spacy_wrapper import spacy_x
from .spacy_wrapper import spacy_eol
from .spacy_wrapper import spacy_space
from .util import group_by_sentence
from .util import to_numeric
from .util import sts_score
from .wordvec import word2glove
from .transformers import FuncTransformer
from .transformers import Shaper
from .load_data import load_dataset
from .load_data import load_glove
from .combiners import PairCosine
from .combiners import SmallerOtherParing
from .combiners import RefGroupPairCosine
from .combiners import GetMatches
from .combiners import SolveDuplicate
from .combiners import AvgPOSCombiner
from .combiners import NumCombiner


__all__ = ("polyglot_nouns",
           "polyglot_proper_nouns",
           "polyglot_pronouns",
           "polyglot_verbs",
           "polyglot_auxiliary_verbs",
           "polyglot_adjectives",
           "polyglot_adverbs",
           "polyglot_numbers",
           "polyglot_1st_noun",
           "polyglot_2nd_noun",
           "polyglot_1st_proper_noun",
           "polyglot_2nd_proper_noun",
           "polyglot_1st_pronoun",
           "polyglot_2nd_pronoun",
           "polyglot_1st_verb",
           "polyglot_2nd_verb",
           "polyglot_1st_auxiliary_verb",
           "polyglot_2nd_auxiliary_verb",
           "polyglot_1st_adjective",
           "polyglot_2nd_adjective",
           "polyglot_1st_adverb",
           "polyglot_2nd_adverb",
           "polyglot_1st_number",
           "polyglot_2nd_number",
           "polyglot_punctuation",
           "polyglot_particle",
           "polyglot_determiner",
           "polyglot_interjection",
           "polyglot_coordinating_conjunction",
           "polyglot_symbol",
           "polyglot_adpositions",
           "polyglot_others",
           "polyglot_subordinating_conjunctions",
           "polyglot_words",
           "polyglot_organizations",
           "polyglot_persons",
           "polyglot_locations",
           "spacy_organizations",
           "spacy_persons",
           "spacy_locations",
           "spacy_groups",
           "spacy_facilities",
           "spacy_geo_locations",
           "spacy_products",
           "spacy_events",
           "spacy_work_of_arts",
           "spacy_laws",
           "spacy_languages",
           "PairSpacyVecTransformer",
           "spacy_tokens",
           "spacy_adj",
           "spacy_adp",
           "spacy_adv",
           "spacy_aux",
           "spacy_conj",
           "spacy_det",
           "spacy_intj",
           "spacy_noun",
           "spacy_num",
           "spacy_part",
           "spacy_pron",
           "spacy_propn",
           "spacy_punct",
           "spacy_sconj",
           "spacy_sym",
           "spacy_verb",
           "spacy_x",
           "spacy_eol",
           "spacy_space"
           "group_by_sentence",
           "to_numeric",
           "sts_score",
           "word2glove",
           "FuncTransformer",
           "Shaper",
           "load_dataset",
           "load_glove",
           "PairCosine",
           "SmallerOtherParing",
           "RefGroupPairCosine",
           "GetMatches",
           "SolveDuplicate",
           "AvgPOSCombiner",
           "NumCombiner")
