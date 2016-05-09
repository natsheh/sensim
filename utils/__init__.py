# -*- coding: utf-8 -*-
#
# This file is part of sensim

"""Helpers for sentence semantic similarity model.

.. Author:: Hussein AL-NATSHEH <hussein.al-natsheh@ish-lyon.cnrs.fr>

"""

"""Helper functions."""

from .pos import get_text
from .pos import get_nouns
from .pos import get_proper_nouns
from .pos import get_pronouns
from .pos import get_verbs
from .pos import get_auxiliary_verbs
from .pos import get_adjectives
from .pos import get_adverbs
from .pos import get_numbers
from .pos import get_1st_noun
from .pos import get_2nd_noun
from .pos import get_1st_proper_noun
from .pos import get_2nd_proper_noun
from .pos import get_1st_pronoun
from .pos import get_2nd_pronoun
from .pos import get_1st_verb
from .pos import get_2nd_verb
from .pos import get_1st_auxiliary_verb
from .pos import get_2nd_auxiliary_verb
from .pos import get_1st_adjective
from .pos import get_2nd_adjective
from .pos import get_1st_adverb
from .pos import get_2nd_adverb
from .pos import get_1st_number
from .pos import get_2nd_number
from .pos import get_punctuation
from .pos import get_particle
from .pos import get_determiner
from .pos import get_interjection
from .pos import get_coordinating_conjunction
from .pos import get_symbol
from .pos import get_words
from .pos import get_organizations
from .pos import get_persons
from .pos import get_locations
from .pos import spacy_organizations
from .pos import spacy_persons
from .pos import spacy_locations
from .pos import spacy_groups
from .pos import spacy_facilities
from .pos import spacy_geo_locations
from .pos import spacy_products
from .pos import spacy_events
from .pos import spacy_work_of_arts
from .pos import spacy_laws
from .pos import spacy_languages
from .pos import group_by_sentence
from .wordvec import word2glove
from .util import to_numeric
from .util import sts_score
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


__all__ = ("get_text",
           "get_nouns",
           "get_proper_nouns",
           "get_pronouns",
           "get_verbs",
           "get_auxiliary_verbs",
           "get_adjectives",
           "get_adverbs",
           "get_numbers",
           "get_1st_noun",
           "get_2nd_noun",
           "get_1st_proper_noun",
           "get_2nd_proper_noun",
           "get_1st_pronoun",
           "get_2nd_pronoun",
           "get_1st_verb",
           "get_2nd_verb",
           "get_1st_auxiliary_verb",
           "get_2nd_auxiliary_verb",
           "get_1st_adjective",
           "get_2nd_adjective",
           "get_1st_adverb",
           "get_2nd_adverb",
           "get_1st_number",
           "get_2nd_number",
           "get_punctuation",
           "get_particle",
           "get_determiner",
           "get_interjection",
           "get_coordinating_conjunction",
           "get_symbol",
           "get_words",
           "get_organizations",
           "get_persons",
           "get_locations",
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
           "group_by_sentence",
           "word2glove",
           "to_numeric",
           "sts_score",
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
