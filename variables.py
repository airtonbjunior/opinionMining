# Airton Bordin Junior
# airtonbjunior@gmail.com
# Federal University of Goias (UFG)
# Computer Science Master's Degree

from nltk.corpus import stopwords
from datetime import datetime

# Paths
SEMEVAL_TRAIN_FILE            = 'datasets/train/twitter-train-cleansed-B.txt'
SEMEVAL_TRAIN_FILE_SPELLCHECK = 'datasets/train/twitter-train-cleansed-B_spell.txt'
#SEMEVAL_TEST_FILE            = 'datasets/test/SemEval2014-task9-test-B-all-tweets_withSVMValues.txt'
#SEMEVAL_TEST_FILE            = 'datasets/test/SemEval2014_SVM_Naive.txt'
#SEMEVAL_TEST_FILE            = 'datasets/test/SemEval2014_SVM_Naive_MS.txt'
#SEMEVAL_TEST_FILE            = 'datasets/test/SemEval2014_SVM_Naive_MS_Lreg.txt'
SEMEVAL_TEST_FILE             = 'datasets/test/SemEval2014_SVM_Naive_MS_Lreg_S140.txt'
SEMEVAL_TEST_FILE_SPELLCHECK  = 'datasets/test/SemEval2014-task9-test-B-all-tweets_withSVMValues_spell.txt'
SEMEVAL_TEST_FILE_ONLY_POS    = 'datasets/test/SemEval2014-task9-test-B-tweets-only-some-POS-classes.txt'
SEMEVAL_2017_TEST_FILE 	      = 'datasets/test/SemEval2017-task4-test.subtask-A.english.txt'
STS_TRAIN_FILE                = 'datasets/train/STS/STStrain.1600000.processed.noemoticon.EDITED.csv'
STS_TEST_FILE                 = 'datasets/test/STS/STS_Gold_All.txt'
DICTIONARY_POSITIVE_WORDS     = 'dictionaries/positive-words.txt'
DICTIONARY_NEGATIVE_WORDS     = 'dictionaries/negative-words.txt'
DICTIONARY_POSITIVE_HASHTAGS  = 'dictionaries/positive-hashtags.txt'
DICTIONARY_NEGATIVE_HASHTAGS  = 'dictionaries/negative-hashtags.txt'
DICTIONARY_POSITIVE_EMOTICONS = 'dictionaries/positive-emoticons.txt'
DICTIONARY_NEGATIVE_EMOTICONS = 'dictionaries/negative-emoticons.txt'
DICTIONARY_NEGATING_WORDS     = 'dictionaries/negating-word-list.txt'
DICTIONARY_BOOSTER_WORDS      = 'dictionaries/boosterWords.txt'
DICTIONARY_SENTIWORDNET       = 'dictionaries/SentiWordNet_3.0.0_20130122.txt'
DICTIONARY_AFFIN              = 'dictionaries/affin.txt'
DICTIONARY_SLANG              = 'dictionaries/slangSD.txt'
DICTIONARY_VADER              = 'dictionaries/vaderLexicon.txt'
DICTIONARY_SEMEVAL2015        = 'dictionaries/SemEval2015-English-Twitter-Lexicon.txt'
DICTIONARY_EFFECT             = 'dictionaries/EffectWordNet.tff'
DICTIONARY_NRC                = 'dictionaries/nrc_words.txt'
DICTIONARY_GENERAL_INQUIRER   = 'dictionaries/general-inquirer.txt'
DICTIONARY_S140               = 'dictionaries/sentiment140_unigram.txt'
DICTIONARY_MPQA               = 'dictionaries/mpqa.txt'

USE_SPELLCHECKED_WORDS      = False # set True if want to use the spellchecked words
USE_ONLY_POS_WORDS          = False

TRAIN_WORDS 		        = 'datasets/train/words_train/words_train.txt'
TRAIN_WORDS_SPELLCHECK      = 'datasets/train/words_train/words_train_spell.txt'
TRAIN_WORDS_POS_TAGGED      = 'datasets/train/words_train/words_train_spell_pos-tagged.txt'
TRAIN_WORDS_POS_TAGGED_W    = 'datasets/train/words_train/words_train_spell_pos-tagged_w.txt' # tagged words but without the tags, only words

TEST_WORDS                  = 'datasets/test/words_test.txt'
TEST_WORDS_SPELLCHECK       = 'datasets/test/words_test_spell.txt'
TEST_WORDS_POS_TAGGED_W     = 'datasets/test/words_test_spell_pos-tagged_W.txt' # tagged words but without the tags, only words

BEST_INDIVIDUAL 		    = 'partial-best-'          + str(datetime.now())[11:13] + str(datetime.now())[14:16] + str(datetime.now())[17:19] + '.txt'
BEST_INDIVIDUAL_AG 		    = 'ag-partial-best-'       + str(datetime.now())[11:13] + str(datetime.now())[14:16] + str(datetime.now())[17:19] + '.txt'
BEST_INDIVIDUAL_GP_ENSEMBLE = 'gp-ens-part-best-'      + str(datetime.now())[11:13] + str(datetime.now())[14:16] + str(datetime.now())[17:19] + '.txt'
BEST_INDIVIDUAL_2CLASSES    = 'partial-best-2classes-' + str(datetime.now())[11:13] + str(datetime.now())[14:16] + str(datetime.now())[17:19] + '.txt'

SAVE_INCORRECT_EVALUATIONS  = False
INCORRECT_EVALUATIONS       = 'incorrect-evaluations-' + str(datetime.now())[11:13] + str(datetime.now())[14:16] + str(datetime.now())[17:19] + '.txt'

INVERT_SARCASM = False

# According Penn Treebank Project
USE_POS_CLASSES = ['VB', 'VBD', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'NN'] # use NN? I don't know

model_results   = []

all_train_words = []
all_test_words  = []

train_using_folds = True
train_using_folds_gp_ensemble = True

use_svm_neutral 	  		= False
use_url_to_neutral	  		= False
use_date_to_neutral         = False
use_url_and_date_to_neutral = False
use_emoticon_analysis 		= True
use_only_emoticons          = False
use_only_gp 	      		= False
use_only_svm		  		= False
use_hashtag_analysis        = False
use_only_textblob_no_train  = False
use_only_naive_bayes        = False
use_only_MS_classifier      = False
use_only_LReg_classifier    = False
use_only_S140_classifier    = False
use_only_RForest_classifier = False
use_all_classifiers         = False

use_gp_ensemble = False # if False, all the models will be executed separated. If True, an ensemble of the models will be executed

neutral_inferior_range = 0
neutral_superior_range = 0

inferior_range_gp_ensemble = -1
superior_range_gp_ensemble = 1

liu_weight       	= 1
sentiwordnet_weight = 1
affin_weight        = 1
vader_weight        = 1
slang_weight        = 1
effect_weight       = 1
semeval2015_weight  = 1
nrc_weight          = 1
gi_weight           = 1
s140_weight         = 1
mpqa_weight         = 1

neutral_values = []

calling_by_test_file = False
calling_by_ag_file   = False

# True: load the dictionary
use_dic_liu          = True
use_dic_sentiwordnet = True
use_dic_affin        = True
use_dic_vader        = True
use_dic_slang        = True
use_dic_effect       = True
use_dic_semeval2015  = True
use_dic_nrc			 = True
use_dic_gi           = True
use_dic_s140         = True
use_dic_mpqa         = True

# Check if the dictionary was loaded
dic_liu_loaded 			= False
dic_sentiwordnet_loaded = False
dic_affin_loaded		= False
dic_vader_loaded		= False
dic_slang_loaded		= False
dic_effect_loaded		= False
dic_semeval2015_loaded	= False
dic_nrc_loaded	        = False
dic_gi_loaded	        = False
dic_s140_loaded         = False
dic_mpqa_loaded         = False

dic_loaded_total = 0

# Balance the train tweets
MAX_POSITIVES_TWEETS = 1500
MAX_NEGATIVES_TWEETS = 1500
MAX_NEUTRAL_TWEETS   = 1500

# GP/GA Parameters
CROSSOVER        = 0.9
AG_CROSSOVER     = 0.9

MUTATION         = 0.1
AG_MUTATION      = 0.3
MUTATE_EPHEMERAL = 0.85

GENERATIONS      = 100
AG_GENERATIONS   = 50

POPULATION       = 150
AG_POPULATION    = 100

cicles_unchanged = 0
generations_unchanged     = 0
max_unchanged_generations = 250
max_unchanged_cicles      = 9999999999

TOTAL_MODELS = 2

# Constraints
root_constraint = False
root_function = "polaritySumAVGUsingWeights"
#root_function = "polaritySumAVG"
#root_functions = ["polaritySumAVGUsingWeights", "if_then_else"]
root_decreased_value = 0.2

massive_functions_constraint = True
massive_function = "polaritySumAVGUsingWeights"
massive_functions_max = 1

neutral_range_constraint = False

generations_unchanged_reached_msg = False

FILE_RESULTS             = 'sandbox/results/test_results-'         + str(datetime.now())[11:13] + str(datetime.now())[14:16] + str(datetime.now())[17:19] + '.txt'
FILE_RESULTS_2CLASSES    = 'sandbox/results/test_results-2classes' + str(datetime.now())[11:13] + str(datetime.now())[14:16] + str(datetime.now())[17:19] + '.txt'

TRAIN_RESULTS            = 'sandbox/results/train-' + str(TOTAL_MODELS) + 'mod'              + str(POPULATION) + 'p'+ str(GENERATIONS) +'g_' + str(datetime.now())[11:13] + str(datetime.now())[14:16] + str(datetime.now())[17:19] + '.txt'
TRAIN_RESULTS_2CLASSES   = 'sandbox/results/train-' + str(TOTAL_MODELS) + 'models_2classes_' + str(POPULATION) + 'p'+ str(GENERATIONS) +'g_' + str(datetime.now())[11:13] + str(datetime.now())[14:16] + str(datetime.now())[17:19] + '.txt'

TRAIN_RESULTS_IMG        = 'sandbox/results/train-' + str(TOTAL_MODELS) + 'mod' + str(POPULATION) + 'p'+ str(GENERATIONS) +'g_' + str(datetime.now())[11:13] + str(datetime.now())[14:16] + str(datetime.now())[17:19] + '-eachFitness'
BEST_RESULTS_IMG         = 'sandbox/results/train-' + str(TOTAL_MODELS) + 'mod' + str(POPULATION) + 'p'+ str(GENERATIONS) +'g_' + str(datetime.now())[11:13] + str(datetime.now())[14:16] + str(datetime.now())[17:19] + 'onlyBests'

tweets_semeval        = []
tweets_semeval_score  = []
svm_values_tweets     = []
svm_is_neutral        = []
svm_normalized_values = []
tweet_semeval_index   = 0

naive_normalized_values = []
MS_normalized_values    = []

tweets_sts       = []
tweets_sts_score = []

tweets_sts_test       = []
tweets_sts_score_test = []

tweets_sts_positive = 0
tweets_sts_negative = 0

all_positive_words = []
all_negative_words = []

# All are converted into sets because we don't need to keep the order
dic_positive_words     = []
dic_negative_words     = []
dic_positive_hashtags  = []
dic_negative_hashtags  = []
dic_positive_emoticons = []
dic_negative_emoticons = []
dic_negation_words     = []
dic_booster_words      = []

# Using python dictionary to improve the search performance
dic_positive_semeval2015  = {}
dic_negative_semeval2015  = {}

dic_positive_slang        = {}
dic_negative_slang        = {}

dic_positive_affin        = {}
dic_negative_affin        = {}

dic_positive_sentiwordnet = {}
dic_negative_sentiwordnet = {}

dic_positive_effect       = {}
dic_negative_effect       = {}

dic_positive_vader        = {}
dic_negative_vader        = {}

dic_positive_nrc          = {}
dic_negative_nrc          = {}

dic_positive_gi           = {}
dic_negative_gi           = {}

dic_positive_s140         = {}
dic_negative_s140         = {}

dic_positive_mpqa         = {}
dic_negative_mpqa         = {}

# Counters
positive_tweets = 0
negative_tweets = 0
neutral_tweets  = 0

fitness_positive = 0
fitness_negative = 0
fitness_neutral  = 0

# Save the best values 
best_AG_weights_combination = []

best_fitness = 0
best_fitness_history  = []
best_fitness_history_dict = {}
best_fitness_per_generation_history = []
all_fitness_history   = []
best_fitness_per_generation_history_dict  = {}

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

tweets_mukh          = []
tweets_mukh_score    = []
tweets_mukh_positive = 0
tweets_mukh_negative = 0

# Test databases - SemEval 2014 (twitter2013, twitter2014, sms, livejournal, sarcasm)
tweets_2013             = []
tweets_2013_score       = []
tweets_2013_score_svm   = []
tweets_2013_score_naive = []
tweets_2013_score_MS    = []
tweets_2013_score_LReg  = []
tweets_2013_score_S140  = []
tweets_2013_score_RFor  = []
tweets_2013_positive    = 0
tweets_2013_negative    = 0
tweets_2013_neutral     = 0

tweets_2014             = []
tweets_2014_score       = []
tweets_2014_score_svm   = []
tweets_2014_score_naive = []
tweets_2014_score_MS    = []
tweets_2014_score_LReg  = []
tweets_2014_score_S140  = []
tweets_2014_score_RFor  = []
tweets_2014_positive    = 0
tweets_2014_negative    = 0
tweets_2014_neutral     = 0

tweets_liveJournal2014             = []
tweets_liveJournal2014_score       = []
tweets_liveJournal2014_score_svm   = []
tweets_liveJournal2014_score_naive = []
tweets_liveJournal2014_score_MS    = []
tweets_liveJournal2014_score_LReg  = []
tweets_liveJournal2014_score_S140  = []
tweets_liveJournal2014_score_RFor  = []
tweets_liveJournal2014_positive    = 0
tweets_liveJournal2014_negative    = 0
tweets_liveJournal2014_neutral     = 0

tweets_2014_sarcasm             = []
tweets_2014_sarcasm_score       = []
tweets_2014_sarcasm_score_svm   = []
tweets_2014_sarcasm_score_naive = []
tweets_2014_sarcasm_score_MS    = []
tweets_2014_sarcasm_score_LReg  = []
tweets_2014_sarcasm_score_S140  = []
tweets_2014_sarcasm_score_RFor  = []
tweets_2014_sarcasm_positive    = 0
tweets_2014_sarcasm_negative    = 0
tweets_2014_sarcasm_neutral     = 0

sms_2013             = []
sms_2013_score       = []
sms_2013_score_svm   = []
sms_2013_score_naive = []
sms_2013_score_MS    = []
sms_2013_score_LReg  = []
sms_2013_score_S140  = []
sms_2013_score_RFor  = []
sms_2013_positive    = 0
sms_2013_negative    = 0
sms_2013_neutral     = 0

all_messages_in_file_order         = []
all_polarities_in_file_order       = []
all_polarities_in_file_order_svm   = []
all_polarities_in_file_order_naive = []
all_polarities_in_file_order_MS    = []
all_polarities_in_file_order_LReg  = []
all_polarities_in_file_order_S140  = []
all_polarities_in_file_order_RFor  = []

MAX_ANALYSIS_TWEETS = 100000

false_neutral_log  = 0
false_negative_log = 0
false_positive_log = 0

log_all_metrics_each_cicle = False
log_all_messages           = False
log_parcial_results        = True
log_times           	   = True
log_loads                  = True

w1  = []
w2  = []
w3  = []
w4  = []
w5  = []
w6  = []
w7  = []
w8  = []
w9  = []
w10 = []
w11 = []

ag_w1  = 0
ag_w2  = 0
ag_w3  = 0
ag_w4  = 0
ag_w5  = 0
ag_w6  = 0
ag_w7  = 0
ag_w8  = 0
ag_w9  = 0
ag_w10 = 0
ag_w11 = 0

save_file_results = True

stop_words = set([x.lower() for x in stopwords.words('english')])
stop_words.remove("won")

week_dates  = ['sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sun', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat']
month_dates = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december', 'jan', 'feb', 'mar', 'apr', 'may', 'june', 'july', 'aug', 'sept', 'oct', 'nov', 'dec']
other_dates = ['tomorrow', 'yesterday', 'today']

all_dates = week_dates + month_dates + other_dates