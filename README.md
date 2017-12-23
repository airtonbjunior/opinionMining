# sentimentPY
Automated creation of an Opinion Mining/Sentiment Analysis Classifier Model using Genetic Programming

# Opinion Mining

"Sentiment analysis, also called opinion mining, is the field of study that analyzes people’s opinions, sentiments, evaluations, appraisals, attitudes, and emotions towards entities such as products, services, organizations, individuals, issues, events, topics, and their attributes." ([Liu, 2012](https://www.cs.uic.edu/~liub/FBS/SentimentAnalysis-and-OpinionMining.pdf))

# Genetic Programming

"Genetic programming (GP) is an evolutionary computation (EC) technique that automatically solves problems without requiring the user to know or specify the form or structure of the solution in advance. At the most abstract level GP is a systematic, domain-independent method for getting computers to solve problems automatically starting from a high-level statement of what needs to be done." ([Poli, Langdon, McPhee](http://www.gp-field-guide.org.uk/))

# Dependencies
* [tkinter](https://docs.python.org/3/library/tkinter.html) (Standard Python interface to the Tk GUI toolkit)
* [DEAP](https://github.com/DEAP/deap) (Distributed Evolutionary Algorithms in Python)
* [numpy](http://www.numpy.org/) (Fundamental package for scientific computing with Python)
* [matplotlib](https://github.com/matplotlib/matplotlib) (Python 2D plotting library)
* [stemming 1.0](https://pypi.python.org/pypi/stemming/1.0) (Python implementations of various stemming algorithms)
* [yagmail](https://github.com/kootenpv/yagmail) (Gmail/SMTP client)
* [nltk](https://github.com/nltk/nltk) (Natural Language Toolkit)
  * ``` >>> import nltk ```
  * ``` >>> nltk.download('stopwords') ```

# References
* Lexicons
  * [Bing Liu Lexicon](https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon)
  * [SentiWordNet](http://sentiwordnet.isti.cnr.it/)
  * [AFINN](http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=6010)
  * [Vader Lexicon](https://github.com/cjhutto/vaderSentiment) (Valence Aware Dictionary and sEntiment Reasoner)
  * [SlangSD](http://slangsd.com/) (Sentiment Dictionary for Slang Words)
  * [MPQA Effect Lexicon](http://mpqa.cs.pitt.edu/lexicons/effect_lexicon/)
  * SemEval2015 Lexicon

# Old repository
* [Related Works](https://airtonbjunior.github.io/mestrado/sentiment-analysis/presentation/related-works.pdf)
* [Paper](https://airtonbjunior.github.io/mestrado/computational-intelligence/final-project/article/main.pdf)
* [Presentation](https://airtonbjunior.github.io/mestrado/sentiment-analysis/presentation/project-presentation.pdf)
* [Closed issues](https://github.com/airtonbjunior/mestrado/issues?q=is%3Aissue+is%3Aclosed)
