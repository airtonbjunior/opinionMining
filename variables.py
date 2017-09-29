# Airton Bordin Junior
# airtonbjunior@gmail.com
# Federal University of Goias (UFG)
# Computer Science Master's Degree

from nltk.corpus import stopwords

SEMEVAL_TRAIN_FILE = 'datasets/twitter-train-cleansed-B.txt'
SEMEVAL_TEST_FILE  = 'datasets/test/SemEval2014-task9-test-B-all-tweets.txt'
DICTIONARY_POSITIVE_WORDS = 'dictionaries/positive-words.txt'
DICTIONARY_NEGATIVE_WORDS = 'dictionaries/negative-words.txt'
DICTIONARY_POSITIVE_HASHTAGS  = 'dictionaries/positive-hashtags.txt'
DICTIONARY_NEGATIVE_HASHTAGS  = 'dictionaries/negative-hashtags.txt'
DICTIONARY_POSITIVE_EMOTICONS = 'dictionaries/positive-emoticons.txt'
DICTIONARY_NEGATIVE_EMOTICONS = 'dictionaries/negative-emoticons.txt'
DICTIONARY_NEGATING_WORDS = 'dictionaries/negating-word-list.txt'
DICTIONARY_BOOSTER_WORDS = 'dictionaries/boosterWords.txt'
DICTIONARY_SENTIWORDNET = 'dictionaries/SentiWordNet_3.0.0_20130122.txt'
DICTIONARY_AFFIN = 'dictionaries/affin.txt'
DICTIONARY_SLANG = 'dictionaries/slangSD.txt'
DICTIONARY_VADER = 'dictionaries/vaderLexicon.txt'
DICTIONARY_SENTIMENT140 = 'dictionaries/sentiment140_unigram.txt'

FILE_RESULTS = 'test-results.txt'
FILE_RESULTS_30 = 'test-results-30_50pop100generations.txt'

model_results = []

use_dic_liu          = False
use_dic_sentiwordnet = False
use_dic_affin        = False
use_dic_vader        = False
use_dic_sentiment140 = False
use_dic_slang        = False
use_dic_effect       = False


MAX_POSITIVES_TWEETS = 1400
MAX_NEGATIVES_TWEETS = 1400
MAX_NEUTRAL_TWEETS   = 1400

CROSSOVER = 0.9
MUTATION = 0.1
GENERATIONS = 100
POPULATION  = 50
generations_unchanged = 0
max_unchanged_generations = 10000

TOTAL_MODELS = 30

tweets_semeval       = []
tweets_semeval_score = []
tweet_semeval_index  = 0

dic_positive_words       = []
dic_negative_words       = []

dic_positive_words_affin = []
dic_negative_words_affin = []
dic_positive_value_affin = []
dic_negative_value_affin = []

dic_positive_words_vader = []
dic_negative_words_vader = []
dic_positive_value_vader = []
dic_negative_value_vader = []

dic_positive_words_s140 = []
dic_negative_words_s140 = []
dic_positive_value_s140 = []
dic_negative_value_s140 = []

dic_positive_hashtags  = []
dic_negative_hashtags  = []
dic_positive_emoticons = []
dic_negative_emoticons = []
dic_negation_words     = []
dic_booster_words      = []

positive_tweets = 0
negative_tweets = 0
neutral_tweets  = 0

fitness_positive = 0
fitness_negative = 0
fitness_neutral  = 0

best_fitness = 0
best_fitness_history  = []
best_fitness_per_generation_history = []
all_fitness_history   = []

best_accuracy = 0

best_precision_positive = 0
best_precision_negative = 0
best_precision_neutral  = 0
best_precision_avg      = 0

best_recall_positive = 0
best_recall_negative = 0
best_recall_neutral  = 0
best_recall_avg      = 0

best_f1_positive = 0
best_f1_negative = 0
best_f1_neutral  = 0
best_f1_avg      = 0
best_f1_positive_negative_avg = 0

best_precision_avg_function = ""
best_recall_avg_function    = ""
best_f1_avg_function        = ""

precision_positive_history = []
precision_negative_history = []
precision_neutral_history  = []
recall_positive_history    = []
recall_negative_history    = []
recall_neutral_history     = []
f1_positive_history        = []
f1_negative_history        = []
f1_neutral_history         = []

tweets_2013       = []
tweets_2013_score = []
tweets_2013_positive = 0
tweets_2013_negative = 0
tweets_2013_neutral  = 0

tweets_2014       = []
tweets_2014_score = []
tweets_2014_positive = 0
tweets_2014_negative = 0
tweets_2014_neutral  = 0

tweets_liveJournal2014       = []
tweets_liveJournal2014_score = []
tweets_liveJournal2014_positive = 0
tweets_liveJournal2014_negative = 0
tweets_liveJournal2014_neutral  = 0

tweets_2014_sarcasm       = []
tweets_2014_sarcasm_score = []
tweets_2014_sarcasm_positive = 0
tweets_2014_sarcasm_negative = 0
tweets_2014_sarcasm_neutral  = 0

sms_2013       = []
sms_2013_score = []
sms_2013_positive = 0
sms_2013_negative = 0
sms_2013_neutral  = 0

log_all_messages = False
MAX_ANALYSIS_TWEETS = 10000

false_neutral_log  = 0
false_negative_log = 0
false_positive_log = 0

log_parcial_results = True

save_file_results = True

stop_words = set(stopwords.words('english'))