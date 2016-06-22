# coding: utf-8

# Author: Hussein AL-NATSHEH <hussein.al-natsheh@ish-lyon.cnrs.fr>
# Affiliation: EA 3083 - University of Lyon, USR 3385 - CNRS, France
# Wrapper for PyEnchant spell-cheker
# License: BSD 3 clause
# 2016

import enchant
d = enchant.Dict("en_US")

def spell_check(word):
	# takes a string and return it if it's correct, first suggestion otherwise.
	if d.check(word):
		return word
	else:
		return d.suggest(word)[0]
