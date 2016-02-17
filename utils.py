# -*- coding: utf-8 -*-
#
# This file is part of sensim

"""Helpers for sentence semantic similarity model.

.. Author:: Hussein AL-NATSHEH <hussein.al-natsheh@ish-lyon.cnrs.fr>

"""

from polyglot.text import Text


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



def get_nouns(s):
    """Get list of nouns from the sentence.

    Parameters
    ----------
    :param s: string
        Sentence

    Returns
    -------
    :returns: list of strings
        list of nouns
    """
    return (get_pos(s)['NOUN'])


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


def get_adjectives(s):
    """Get list of adjectives from the sentence.

    Parameters
    ----------
    :param s: string
        Sentence

    Returns
    -------
    :returns: list of strings
        list of adjectives
    """
    return (get_pos(s)['ADJ'])


def get_adverbs(s):
    """Get list of adverbs from the sentence.

    Parameters
    ----------
    :param s: string
        Sentence

    Returns
    -------
    :returns: list of strings
        list of adverbs
    """
    return (get_pos(s)['ADV'])


def get_numbers(s):
    """Get list of numbers from the sentence.

    Parameters
    ----------
    :param s: string
        Sentence

    Returns
    -------
    :returns: list of strings
        list of numbers
    """
    return (get_pos(s)['NUM'])


def group_by_sentence(r):
    """Grouping function for ``PairTransformer``.

    Parameters
    ----------
    :param r: iterable
        sentence in a singleton.

    Returns
    -------
    :returns: string
        Sentence id
    """
    return r[0]["sentence_id"]
