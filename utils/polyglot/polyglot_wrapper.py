# -*- coding: utf-8 -*-
#
# This file is part of sensim

"""Helpers for sentence semantic similarity model.

.. Author:: Hussein AL-NATSHEH <hussein.al-natsheh@ish-lyon.cnrs.fr>

"""

from polyglot.text import Text
import numpy as np

def polyglot_words(s):
    text = Text(s)
    text.language = 'en'
    return text.words

def polyglot_name_entities(s):
    text = Text(s)
    text.language = 'en'
    entities = text.entities
    org = list(' ')
    loc = list(' ')
    per = list(' ')
    for entity in entities:
        if entity.tag == 'I-ORG':
            org = list(entity)
        elif entity.tag == 'I-PER':
            per = list(entity)
        else:
            loc = list(entity)
    return org, per, loc

def polyglot_organizations(s):
    org, _, _ = polyglot_name_entities(s)
    return org

def polyglot_persons(s):
    _, per, _ = polyglot_name_entities(s)
    return per

def polyglot_locations(s):
    _, _, loc = polyglot_name_entities(s)
    return loc

def polyglot_pos(s):
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
    def polyglot_pos(pos, pos_codes):
        for w in text.words:
            if w.pos_tag == pos_code:
                pos.append(w)
        if len(pos) > 0:
            return pos
        else:
            return ([' '])
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
        POS[pos_code] = polyglot_pos(pos, pos_code)
    return POS

def polyglot_adpositions(s):
    """Get the set of th adpositions tags from the sentence.

    Parameters
    ----------
    :param s: string
        Sentence

    Returns
    -------
    :returns: array of string
        adpositions
    """
    adpositions = polyglot_pos(s)['ADP']
    return (adpositions)

def polyglot_others(s):
    """Get the set of other PoS tags from the sentence.

    Parameters
    ----------
    :param s: string
        Sentence

    Returns
    -------
    :returns: array of string
        other PoS tags
    """
    others = polyglot_pos(s)['X']
    return (others)

def polyglot_subordinating_conjunctions(s):
    """Get the subordinating_conjunctions from the sentence.

    Parameters
    ----------
    :param s: string
        Sentence

    Returns
    -------
    :returns: array of string
        subordinating_conjunctions
    """
    subordinating_conjunctions = polyglot_pos(s)['SCONJ']
    return (subordinating_conjunctions)

def polyglot_punctuation(s):
    """Get the punctuation from the sentence.

    Parameters
    ----------
    :param s: string
        Sentence

    Returns
    -------
    :returns: array of string
        punctuation
    """
    punctuations = polyglot_pos(s)['PUNCT']
    return (punctuations)

def polyglot_particle(s):
    """Get the particle from the sentence.

    Parameters
    ----------
    :param s: string
        Sentence

    Returns
    -------
    :returns: array of string
        particle
    """
    particles = polyglot_pos(s)['PART']
    return (particles)

def polyglot_determiner(s):
    """Get the determiner from the sentence.

    Parameters
    ----------
    :param s: string
        Sentence

    Returns
    -------
    :returns: array of string
        determiner
    """
    determiners = polyglot_pos(s)['DET']
    return (determiners)

def polyglot_interjection(s):
    """Get the interjection from the sentence.

    Parameters
    ----------
    :param s: string
        Sentence

    Returns
    -------
    :returns: array of string
        interjection
    """
    interjections = polyglot_pos(s)['INTJ']
    return (interjections)

def polyglot_coordinating_conjunction(s):
    """Get the coordinating_conjunction from the sentence.

    Parameters
    ----------
    :param s: string
        Sentence

    Returns
    -------
    :returns: array of string
        coordinating_conjunction
    """
    coordinating_conjunctions = polyglot_pos(s)['CONJ']
    return (coordinating_conjunctions)

def polyglot_symbol(s):
    """Get the symbol from the sentence.

    Parameters
    ----------
    :param s: string
        Sentence

    Returns
    -------
    :returns: array of string
        symbol
    """
    symbols = polyglot_pos(s)['SYM']
    return (symbols)

def polyglot_nouns(s):
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
    return (polyglot_pos(s)['NOUN'])

def polyglot_1st_noun(s):
    """Get first noun from the sentence.

    Parameters
    ----------
    :param s: string
        Sentence

    Returns
    -------
    :returns: string
        first noun
    """
    nouns = polyglot_nouns(s)
    return nouns[0]

def polyglot_2nd_noun(s):
    """Get second noun from the sentence.

    Parameters
    ----------
    :param s: string
        Sentence

    Returns
    -------
    :returns: string
        second noun
    """
    nouns = polyglot_nouns(s)
    if len(nouns) > 1:
        return nouns[1]
    else:
        return (' ')

def polyglot_proper_nouns(s):
    """Get list of proper_nouns from the sentence.

    Parameters
    ----------
    :param s: string
        Sentence

    Returns
    -------
    :returns: list of strings
        list of proper_nouns
    """
    return (polyglot_pos(s)['PROPN'])

def polyglot_1st_proper_noun(s):
    """Get first proper_noun from the sentence.

    Parameters
    ----------
    :param s: string
        Sentence

    Returns
    -------
    :returns: string
        first proper_noun
    """
    proper_nouns = polyglot_proper_nouns(s)
    return proper_nouns[0]

def polyglot_2nd_proper_noun(s):
    """Get second proper_noun from the sentence.

    Parameters
    ----------
    :param s: string
        Sentence

    Returns
    -------
    :returns: string
        second proper_noun
    """
    proper_nouns = polyglot_proper_nouns(s)
    if len(proper_nouns) > 1:
        return proper_nouns[1]
    else:
        return (' ')

def polyglot_pronouns(s):
    """Get list of pronouns from the sentence.

    Parameters
    ----------
    :param s: string
        Sentence

    Returns
    -------
    :returns: list of strings
        list of pronouns
    """
    return (polyglot_pos(s)['PRON'])

def polyglot_1st_pronoun(s):
    """Get first pronoun from the sentence.

    Parameters
    ----------
    :param s: string
        Sentence

    Returns
    -------
    :returns: string
        first pronoun
    """
    pronouns = polyglot_pronouns(s)
    return pronouns[0]

def polyglot_2nd_pronoun(s):
    """Get second pronoun from the sentence.

    Parameters
    ----------
    :param s: string
        Sentence

    Returns
    -------
    :returns: string
        second pronoun
    """
    pronouns = polyglot_pronouns(s)
    if len(pronouns) > 1:
        return pronouns[1]
    else:
        return (' ')

def polyglot_verbs(s):
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
    return (polyglot_pos(s)['VERB'])

def polyglot_1st_verb(s):
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
    verbs = polyglot_verbs(s)
    return verbs[0]

def polyglot_2nd_verb(s):
    """Get second verb from the sentence.

    Parameters
    ----------
    :param s: string
        Sentence

    Returns
    -------
    :returns: string
        second verb
    """
    verbs = polyglot_verbs(s)
    if len(verbs) > 1:
        return verbs[1]
    else:
        return (' ')

def polyglot_auxiliary_verbs(s):
    """Get list of auxiliary_verbs from the sentence.

    Parameters
    ----------
    :param s: string
        Sentence

    Returns
    -------
    :returns: list of strings
        list of auxiliary_verbs
    """
    return (polyglot_pos(s)['AUX'])

def polyglot_1st_auxiliary_verb(s):
    """Get first auxiliary_verb from the sentence.

    Parameters
    ----------
    :param s: string
        Sentence

    Returns
    -------
    :returns: string
        first auxiliary_verb
    """
    auxiliary_verbs = polyglot_auxiliary_verbs(s)
    return auxiliary_verbs[0]

def polyglot_2nd_auxiliary_verb(s):
    """Get second auxiliary_verb from the sentence.

    Parameters
    ----------
    :param s: string
        Sentence

    Returns
    -------
    :returns: string
        second auxiliary_verb
    """
    auxiliary_verbs = polyglot_auxiliary_verbs(s)
    if len(auxiliary_verbs) > 1:
        return auxiliary_verbs[1]
    else:
        return (' ')

def polyglot_adjectives(s):
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
    return (polyglot_pos(s)['ADJ'])

def polyglot_1st_adjective(s):
    """Get first adjective from the sentence.

    Parameters
    ----------
    :param s: string
        Sentence

    Returns
    -------
    :returns: string
        first adjective
    """
    adjectives = polyglot_adjectives(s)
    return adjectives[0]

def polyglot_2nd_adjective(s):
    """Get second adjective from the sentence.

    Parameters
    ----------
    :param s: string
        Sentence

    Returns
    -------
    :returns: string
        second adjective
    """
    adjectives = polyglot_adjectives(s)
    if len(adjectives) > 1:
        return adjectives[1]
    else:
        return (' ')

def polyglot_adverbs(s):
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
    return (polyglot_pos(s)['ADV'])

def polyglot_1st_adverb(s):
    """Get first adverb from the sentence.

    Parameters
    ----------
    :param s: string
        Sentence

    Returns
    -------
    :returns: string
        first adverb
    """
    adverbs = polyglot_adverbs(s)
    return adverbs[0]

def polyglot_2nd_adverb(s):
    """Get second adverb from the sentence.

    Parameters
    ----------
    :param s: string
        Sentence

    Returns
    -------
    :returns: string
        second adverb
    """
    adverbs = polyglot_adverbs(s)
    if len(adverbs) > 1:
        return adverbs[1]
    else:
        return (' ')


def polyglot_numbers(s):
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
    return (polyglot_pos(s)['NUM'])

def polyglot_1st_number(s):
    """Get first number from the sentence.

    Parameters
    ----------
    :param s: string
        Sentence

    Returns
    -------
    :returns: string
        first number
    """
    numbers = polyglot_numbers(s)
    return numbers[0]

def polyglot_2nd_number(s):
    """Get second number from the sentence.

    Parameters
    ----------
    :param s: string
        Sentence

    Returns
    -------
    :returns: string
        second number
    """
    numbers = polyglot_numbers(s)
    if len(numbers) > 1:
        return numbers[1]
    else:
        return (' ')
