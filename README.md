# SenSim

Sentence Similarity Estimator (SenSim)

Dependancies
------------
	This repository currently supports Python 2.7
	For the used default values in sts.py/sts_light.py, you need the following:
	sklearn==0.18
	polyglot==16.07.04 
		Dependencies: (python-numpy libicu-dev)
		(to use in ubuntu/debian) sudo apt-get install python-numpy libicu-dev
	beard==0.2
	digify==0.2
	enchant==1.6.8
	spacy==0.100.5
		Needed models: python -m spacy.en.download glove

Usage to reproduce the results in the paper
-------------------------------------------
	After cloning the repositpry, use sts.py with its documented arguments


Usage to reproduce the results against the STS Benchmark
--------------------------------------------------------
	After cloning the repositpry, use sts_benchmark.py with its default param


Please cite using the following BibTex entry
--------------------------------------------

```
@InProceedings{alnatsheh-EtAl:2017:SemEval,
  author    = {Al-Natsheh, Hussein T.  and  Martinet, Lucie  and  Muhlenbach, Fabrice  and  ZIGHED, Djamel Abdelkader},
  title     = {UdL at SemEval-2017 Task 1: Semantic Textual Similarity Estimation of English Sentence Pairs Using Regression Model over Pairwise Features},
  booktitle = {Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017)},
  month     = {August},
  year      = {2017},
  address   = {Vancouver, Canada},
  publisher = {Association for Computational Linguistics},
  pages     = {115--119},
  abstract  = {This paper describes the model UdL we proposed to solve the semantic textual
	similarity task of SemEval 2017 workshop. The track we participated in was
	estimating the semantics relatedness of a given set of sentence pairs in
	English. The best run out of three submitted runs of our model achieved a
	Pearson correlation score of 0.8004 compared to a hidden human annotation of
	250~pairs. We used random forest ensemble learning to map an expandable set of
	extracted pairwise features into a semantic similarity estimated value bounded
	between 0 and 5. Most of these features were calculated using word embedding
	vectors similarity to align Part of Speech (PoS) and Name Entities (NE) tagged
	tokens of each sentence pair. Among other pairwise features, we experimented a
	classical tf-idf weighted Bag of Words (BoW) vector model but with
	character-based range of n-grams instead of words. This sentence vector
	BoW-based feature gave a relatively high importance value percentage in the
	feature importances analysis of the ensemble learning.},
  url       = {http://www.aclweb.org/anthology/S17-2013}
}

```
