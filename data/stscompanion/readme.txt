
	      STS Benchmark: Companion English datasets
			    
	    Semantic Textual Similarity 2012-2017 Dataset

		    http://ixa2.si.ehu.es/stswiki
				   

The companion datasets to the STS Benchmark comprise the rest of the
English datasets used in the STS tasks organized by us in the context
of SemEval between 2012 and 2017.

We collated two datasets, one with pairs of sentences related to
machine translation evaluation. Another one with the rest of datasets,
which can be used for domain adaptation studies.

For reference, this is the breakdown according to the original names
and task years of the datasets:


  MT-related datasets: sts-mt.csv
  
     file              years pairs
     -----------------------------
     SMTnews            2012   399
     SMTeuroparl        2012  1293
     postediting        2016   244

  Other datasets: sts-other.csv
  
     file              years pairs
     -----------------------------
     OnWN               2012   750
     OnWN               2013   561
     OnWN               2014   750
     FNWN               2013   189
     tweet-news         2014   750
     belief             2015   375
     plagiarism         2016   230
     question-question  2016   209

Note, the 2013 SMT dataset is available through LDC only.


Introduction
------------

Given two sentences of text, s1 and s2, the systems need to compute
how similar s1 and s2 are, returning a similarity score between 0 and
5. The dataset comprises naturally occurring pairs of sentences drawn
from several domains and genres, annotated by crowdsourcing. See
papers by Agirre et al. (2012; 2013; 2014; 2015; 2016).


Format
------

Each file is encoded in utf-8 (a superset of ASCII), and has the
following tab separated fields:

  filename year score sentence1 sentence2

optionally there might be some license-related fields after sentence2.

NOTE: Given that some sentence pairs have been reused here and
elsewhere, systems should NOT use the following datasets to develop or
train their systems (see below for more details on datasets):

- Any of the datasets in Semeval STS competitions
- The data from the evaluation tasks at any WMT (all years are
  forbidden)
- DARPA GALE HTER and HyTER datasets.
- Ontonotes - Wordnet sense aligned definitions.
- FrameNet - Wordnet sense aligned definitions.
- The Linking-Tweets-to-News data set (Guo et al., 2013)


Evaluation script
-----------------

The official evaluation is the Pearson correlation coefficient. Given
an output file comprising the system scores (one per line) in a file
called sys.txt, you can use the evaluation script as follows:

$ perl correlation.pl sts-dev.txt sys.txt


Notes on datasets and licenses
------------------------------

If using this data in your research please cite (Agirre et al. 2017)
and the STS website: http://ixa2.si.ehu.eus/stswiki.

Please see LICENSE.txt



Other
-----

Please check http://ixa2.si.ehu.eus/stswiki

We recommend that interested researchers join the (low traffic)
mailing list:

 http://groups.google.com/group/STS-semeval


Organizers of tasks by year
---------------------------

2012 Eneko Agirre, Daniel Cer, Mona Diab, Aitor Gonzalez-Agirre

2013 Eneko Agirre, Daniel Cer, Mona Diab, Aitor Gonzalez-Agirre,
     WeiWei Guo

2014 Eneko Agirre, Carmen Banea, Claire Cardie, Daniel Cer, Mona Diab,
     Aitor Gonzalez-Agirre, Weiwei Guo, Rada Mihalcea, German Rigau,
     Janyce Wiebe

2015 Eneko Agirre, Carmen Banea, Claire Cardie, Daniel Cer, Mona Diab,
     Aitor Gonzalez-Agirre, Weiwei Guo, Inigo Lopez-Gazpio, Montse
     Maritxalar, Rada Mihalcea, German Rigau, Larraitz Uria, Janyce
     Wiebe

2016 Eneko Agirre, Carmen Banea, Daniel Cer, Mona Diab, Aitor
     Gonzalez-Agirre, Rada Mihalcea, German Rigau, Janyce
     Wiebe

2017 Eneko Agirre, Daniel Cer, Mona Diab, Iñigo Lopez-Gazpio, Lucia
     Specia



References
----------

Eneko Agirre, Daniel Cer, Mona Diab, Aitor Gonzalez-Agirre. Task 6: A
   Pilot on Semantic Textual Similarity. Procceedings of Semeval 2012

Eneko Agirre, Daniel Cer, Mona Diab, Aitor Gonzalez-Agirre, WeiWei
   Guo. *SEM 2013 shared task: Semantic Textual
   Similarity. Procceedings of *SEM 2013

Eneko Agirre; Carmen Banea; Claire Cardie; Daniel Cer; Mona Diab;
   Aitor Gonzalez-Agirre; Weiwei Guo; Rada Mihalcea; German Rigau;
   Janyce Wiebe. Task 10: Multilingual Semantic Textual
   Similarity. Proceedings of SemEval 2014.

Eneko Agirre; Carmen Banea; Claire Cardie; Daniel Cer; Mona Diab;
   Aitor Gonzalez-Agirre; Weiwei Guo; Inigo Lopez-Gazpio; Montse
   Maritxalar; Rada Mihalcea; German Rigau; Larraitz Uria; Janyce
   Wiebe. Task 2: Semantic Textual Similarity, English, Spanish and
   Pilot on Interpretability. Proceedings of SemEval 2015.

Eneko Agirre; Carmen Banea; Daniel Cer; Mona Diab; Aitor
   Gonzalez-Agirre; Rada Mihalcea; German Rigau; Janyce
   Wiebe. Semeval-2016 Task 1: Semantic Textual Similarity,
   Monolingual and Cross-Lingual Evaluation. Proceedings of SemEval
   2016.

Eneko Agirre, Daniel Cer, Mona Diab, Iñigo Lopez-Gazpio, Lucia
   Specia. Semeval-2017 Task 1: Semantic Textual Similarity
   Multilingual and Crosslingual Focused Evaluation. Proceedings of
   SemEval 2017.

Paul Clough and Mark Stevenson. 2011. Developing a corpus of
   plagiarised short answers. Language Resources and Evaluation,
   45(1):5-24
   http://ir.shef.ac.uk/cloughie/resources/plagiarism_corpus.html

Weiwei Guo, Hao Li, Heng Ji and Mona Diab. 2013.  Linking Tweets to
   News: A Framework to Enrich Online Short Text Data in Social Media.
   In Proceedings of the 51th Annual Meeting of the Association for
   Computational Linguistics

Eduard Hovy, Mitchell Marcus, Martha Palmer, Lance Ramshaw, and Ralph
   Weischedel. 2006. Ontonotes: The 90% solution. In Proceedings of
   the Human Language Technology Conference of the North American
   Chapter of the ACL.

Lucia Specia. 2011. Exploiting Objective Annotations for Measuring
  Trans lation Post-editing Effort. In Proceedings of the 15th
  Conference of the European Association from Machine Translation
  (EAMT 2011).
  http://staffwww.dcs.shef.ac.uk/people/L.Specia/resources



