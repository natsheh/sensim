# -*- coding: utf-8 -*-
#
# This file is part of sensim

"""Helpers for sentence semantic similarity model.

.. Author:: Hussein AL-NATSHEH <hussein.al-natsheh@ish-lyon.cnrs.fr>

"""

"""Helper functions."""

from .util import get_text
from .util import group_by_sentence
from .util import to_numeric
from .util import sts_score
from .util import de_contraction
from .wordvec import word2glove
from .transformers import FuncTransformer
from .transformers import Shaper
from .load_data import read_tsv
from .load_data import df_2_dset
from .load_data import load_dataset
from .combiners import PairCosine
from .combiners import SmallerOtherParing
from .combiners import RefGroupPairCosine
from .combiners import GetMatches
from .combiners import SolveDuplicate
from .combiners import AvgPOSCombiner
from .combiners import NumCombiner
from .combiners import PairMLSTM
from .combiners import PairExpManhattan
from .combiners import PairManhattan


__all__ = ("get_text",
           "group_by_sentence",
           "to_numeric",
           "sts_score",
           "de_contraction",
           "word2glove",
           "FuncTransformer",
           "Shaper",
           "read_tsv",
           "df_2_dset",
           "load_dataset",
           "PairCosine",
           "SmallerOtherParing",
           "RefGroupPairCosine",
           "GetMatches",
           "SolveDuplicate",
           "AvgPOSCombiner",
           "NumCombiner",
           "PairMLSTM",
           "PairExpManhattan",
           "PairManhattan")
