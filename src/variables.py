""" 
<Author> Airton Bordin Junior
<Email>  airtonbjunior@gmail.com

Federal University of Goias (UFG)
Computer Science Master's Degree


variables.py
	Variables used by the system

"""

from nltk.corpus import stopwords
from datetime import datetime

SYSTEM_VERSION = "0.3"

"""
	FILE PATHS
"""
PREFIX_PATH  = "../"

DATASET_PATH = {}
DIC_PATH     = {}

DATASET_PATH['train']    = PREFIX_PATH + 'datasets/train/train_messages.txt'
DATASET_PATH['test']     = PREFIX_PATH + 'datasets/test/test_messages.txt'
DIC_PATH["booster"]      = PREFIX_PATH + 'dictionaries/Special/Booster/'
DIC_PATH["negating"]     = PREFIX_PATH + 'dictionaries/Special/Negating/'
DIC_PATH["emoticon"]     = PREFIX_PATH + 'dictionaries/Special/Emoticon/'
DIC_PATH["hashtag"]      = PREFIX_PATH + 'dictionaries/Special/Hashtag/'
DIC_PATH["liu"]          = PREFIX_PATH + 'dictionaries/LIU/'
DIC_PATH["sentiwordnet"] = PREFIX_PATH + 'dictionaries/Sentiwordnet/'
DIC_PATH["afinn"]        = PREFIX_PATH + 'dictionaries/AFINN/'
DIC_PATH["vader"]        = PREFIX_PATH + 'dictionaries/Vader/'
DIC_PATH["slang"]        = PREFIX_PATH + 'dictionaries/Slang/'
DIC_PATH["effect"]       = PREFIX_PATH + 'dictionaries/Effect/'
DIC_PATH["semeval2015"]  = PREFIX_PATH + 'dictionaries/Semeval2015/'
DIC_PATH["nrc"]          = PREFIX_PATH + 'dictionaries/NRC/'
DIC_PATH["gi"]           = PREFIX_PATH + 'dictionaries/GeneralInquirer/'
DIC_PATH["s140"]         = PREFIX_PATH + 'dictionaries/Sentiment140/'
DIC_PATH["mpqa"]         = PREFIX_PATH + 'dictionaries/MPQA/'
TRAIN_WORDS 		     = PREFIX_PATH + 'datasets/train/words_train/words_train.txt'
TRAIN_WORDS_SPELLCHECK   = PREFIX_PATH + 'datasets/train/words_train/words_train_spell.txt'
TRAIN_WORDS_POS_TAGGED   = PREFIX_PATH + 'datasets/train/words_train/words_train_spell_pos-tagged.txt'
TRAIN_WORDS_POS_TAGGED_W = PREFIX_PATH + 'datasets/train/words_train/words_train_spell_pos-tagged_w.txt' # tagged words but without the tags, only words
TEST_WORDS               = PREFIX_PATH + 'datasets/test/words_test.txt'
TEST_WORDS_SPELLCHECK    = PREFIX_PATH + 'datasets/test/words_test_spell.txt'
TEST_WORDS_POS_TAGGED_W  = PREFIX_PATH + 'datasets/test/words_test_spell_pos-tagged_W.txt' # tagged words but without the tags, only words

BEST_INDIVIDUAL 		    = PREFIX_PATH + 'sandbox/partial_results/partial-best-'          + str(datetime.now())[11:13] + str(datetime.now())[14:16] + str(datetime.now())[17:19] + '.txt'
BEST_INDIVIDUAL_GP_ENSEMBLE = PREFIX_PATH + 'sandbox/partial_results/gp-ens-part-best-'      + str(datetime.now())[11:13] + str(datetime.now())[14:16] + str(datetime.now())[17:19] + '.txt'
BEST_INDIVIDUAL_2CLASSES    = PREFIX_PATH + 'sandbox/partial_results/partial-best-2classes-' + str(datetime.now())[11:13] + str(datetime.now())[14:16] + str(datetime.now())[17:19] + '.txt'
INCORRECT_EVALUATIONS       = PREFIX_PATH + 'sandbox/partial_results/incorrect-evaluations-' + str(datetime.now())[11:13] + str(datetime.now())[14:16] + str(datetime.now())[17:19] + '.txt'

"""
	CONTROL VARS (BOOLEAN)
"""
USE_SPELLCHECKED_WORDS = False # set True if want to use the spellchecked words
USE_ONLY_POS_WORDS     = False

USE_TRAIN_FOLDS = False

SAVE = {'incorrect_evaluations': False, 'only_best': False, 'dic_means': False, 'all_dic_values': True, 'file_results': True, 'save_n_individuals': 3}
LOG  = {'times': True, 'all_each_cicle': False, 'all_messages': False, 'partial_results': True, 'loads': True}

"""
	DICTIONARIES
"""
DICTIONARIES           = ["liu", "sentiwordnet", "afinn", "vader", "slang", "effect", "semeval2015", "nrc", "gi", "s140", "mpqa"]
SPECIAL_DICTIONARIES   = ["emoticon", "hashtag"] # always loaded
CLASSLESS_DICTIONARIES = ["booster", "negating"] # always loaded

DIC_WORDS  = {}
USE_DIC    = {'liu': True,  'sentiwordnet': True,  'afinn': True,  'vader': True,  'slang': True,  'effect': True,  'semeval2015': True,  'nrc': True,  'gi': True,  's140': True,  'mpqa': True}
DIC_LOADED = {'liu': False, 'sentiwordnet': False, 'afinn': False, 'vader': False, 'slang': False, 'effect': False, 'semeval2015': False, 'nrc': False, 'gi': False, 's140': False, 'mpqa': False, 'total': 0}

"""
	BEST AND HISTORY
"""
BEST    = {'fitness': 0, 'accuracy': 0, 'precision': {}, 'recall': {}, 'f1': {}}
HISTORY = {'precision': {}, 'recall': {}, 'f1': {}, 'fitness': {}}

BEST['precision']    = {'positive': 0, 'negative': 0, 'neutral': 0, 'avg': 0, 'avg_function': ""}
BEST['recall']       = {'positive': 0, 'negative': 0, 'neutral': 0, 'avg': 0, 'avg_function': ""}
BEST['f1']           = {'positive': 0, 'negative': 0, 'neutral': 0, 'avg': 0, 'avg_function': "", 'avg_pn': 0}
HISTORY['precision'] = {'positive': 0, 'negative': 0, 'neutral': 0, 'avg': 0}
HISTORY['recall']    = {'positive': 0, 'negative': 0, 'neutral': 0, 'avg': 0}
HISTORY['f1']        = {'positive': 0, 'negative': 0, 'neutral': 0, 'avg': 0, 'avg_pn': 0}
HISTORY['fitness']   = {'all': [], 'per_generation': [], 'best': []} 

"""
	SENTIMENT ANALYSIS MISC
"""
INVERT_SARCASM = False
BOOSTER_FACTOR = 2

# According Penn Treebank Project
USE_POS_CLASSES = ['VB', 'VBD', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'NN'] # use NN? I don't know

STOP_WORDS = set([x.lower() for x in stopwords.words('english')])
STOP_WORDS.remove("won")

WEEK_DATES  = ['sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sun', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat']
MONTH_DATES = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december', 'jan', 'feb', 'mar', 'apr', 'may', 'june', 'july', 'aug', 'sept', 'oct', 'nov', 'dec']
OTHER_DATES = ['tomorrow', 'yesterday', 'today']

ALL_DATES = WEEK_DATES + MONTH_DATES + OTHER_DATES

"""
	GP PARAMETERS
"""
CROSSOVER        = 0.9
MUTATION         = 0.1
MUTATE_EPHEMERAL = 0.85
GENERATIONS      = 15
POPULATION       = 15
TREE_MIN_HEIGHT  = 2
TREE_MAX_HEIGHT  = 6
TOTAL_MODELS     = 3
HOF              = 4 # Hall of fame

CICLES_UNCHANGED          = 0
GENERATIONS_UNCHANGED     = 0
MAX_UNCHANGED_GENERATIONS = 250
MAX_UNCHANGED_CICLES      = 9999999999

"""
	CONSTRAINTS
"""
CONSTRAINT = {'root': {}, 'massive': {}, 'neutral_range': {'active': False}, 'active': True}
CONSTRAINT['root']    = {'active': True,  'function': 'polSumAVGWeights', 'decrease_rate': 0.2}
CONSTRAINT['massive'] = {'active': True,  'function': 'polSumAVGWeights', 'decrease_rate': 0.2, 'max': 1}

GEN_UNCHANGED_REACH_MSG = False

"""
	RESULT FILES PATHS
"""
FILE_RESULTS           = PREFIX_PATH + 'sandbox/results/test_results-'         + str(datetime.now())[11:13] + str(datetime.now())[14:16] + str(datetime.now())[17:19] + '.txt'
FILE_RESULTS_2CLASSES  = PREFIX_PATH + 'sandbox/results/test_results-2classes' + str(datetime.now())[11:13] + str(datetime.now())[14:16] + str(datetime.now())[17:19] + '.txt'

TRAIN_RESULTS          = PREFIX_PATH + 'sandbox/results/train-' + str(TOTAL_MODELS) + 'mod'              + str(POPULATION) + 'p'+ str(GENERATIONS) +'g_' + str(datetime.now())[11:13] + str(datetime.now())[14:16] + str(datetime.now())[17:19] + '.txt'
TRAIN_RESULTS_GP_ENS   = PREFIX_PATH + 'sandbox/results/train-gp-ensemble-' + str(datetime.now())[11:13] + str(datetime.now())[14:16] + str(datetime.now())[17:19] + '.txt'
TRAIN_RESULTS_2CLASSES = PREFIX_PATH + 'sandbox/results/train-' + str(TOTAL_MODELS) + 'models_2classes_' + str(POPULATION) + 'p'+ str(GENERATIONS) +'g_' + str(datetime.now())[11:13] + str(datetime.now())[14:16] + str(datetime.now())[17:19] + '.txt'

TRAIN_RESULTS_IMG      = PREFIX_PATH + 'sandbox/results/train-' + str(TOTAL_MODELS) + 'mod' + str(POPULATION) + 'p'+ str(GENERATIONS) +'g_' + str(datetime.now())[11:13] + str(datetime.now())[14:16] + str(datetime.now())[17:19] + '-eachFitness'
BEST_RESULTS_IMG       = PREFIX_PATH + 'sandbox/results/train-' + str(TOTAL_MODELS) + 'mod' + str(POPULATION) + 'p'+ str(GENERATIONS) +'g_' + str(datetime.now())[11:13] + str(datetime.now())[14:16] + str(datetime.now())[17:19] + 'onlyBests'

"""
	DATASETS (TRAIN AND TEST)
"""
MESSAGES = {'train': [], 'test': []}
MODEL    = {'bests': [], 'others': []}

# Balance
MAX = {'train':{}, 'test': {}}
MAX['train'] = {'positive': 1500, 'negative': 1500, 'neutral': 1500, 'all': 0}
MAX['test']  = {'positive': 5000, 'negative': 5000, 'neutral': 5000, 'all': 0}

NEUTRAL = {'range': {'inferior': 0, 'superior': 0}}



############## REVIEW below ##############

all_train_words = []
all_test_words  = []

t2k13_outputs = []
t2k14_outputs = []
sms_outputs   = []
lj_outputs    = []
sar_outputs   = []
all_model_outputs = []

train_using_bagging = True
train_file_size     = 9684

use_svm_neutral 	  		= False
use_url_to_neutral	  		= False
use_date_to_neutral         = False
use_url_and_date_to_neutral = False
use_emoticon_analysis 		= False
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
use_only_SGD_classifier     = False
use_all_classifiers         = False

use_all_classifiers_nopg_sum = True

using_gp_default = False
use_gp_ensemble  = False # if False, all the models will be executed separated. If True, an ensemble of the models will be executed

inferior_range_gp_ensemble = 0
superior_range_gp_ensemble = 0

neutral_values = []

calling_by_test_file = False
calling_by_ag_file   = False

use_original_dic_values = True

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

# Counters
POSITIVE_MESSAGES = 0
NEGATIVE_MESSAGES = 0
NEUTRAL_MESSAGES  = 0

fitness_positive = 0
fitness_negative = 0
fitness_neutral  = 0

# Save the best values 
best_AG_weights_combination = []

false_neutral_log  = 0
false_negative_log = 0
false_positive_log = 0

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

limit_dictionary_weight = 3


#SEMEVAL_TRAIN_FILE             = PREFIX_PATH + 'datasets/train/train_messages.txt'
#SEMEVAL_TRAIN_FILE_ORIGINAL    = PREFIX_PATH + 'datasets/train/twitter-train-cleansed-B.txt'
#SEMEVAL_TRAIN_FILE_SPELLCHECK  = PREFIX_PATH + 'datasets/train/twitter-train-cleansed-B_spell.txt'
#SEMEVAL_TEST_FILE              = PREFIX_PATH + 'datasets/test/SemEval2014_SVM_Naive_MS_Lreg_S140.txt'
#SEMEVAL_TEST_FILE_SPELLCHECK   = PREFIX_PATH + 'datasets/test/SemEval2014-task9-test-B-all-tweets_withSVMValues_spell.txt'
#SEMEVAL_TEST_FILE_ONLY_POS     = PREFIX_PATH + 'datasets/test/SemEval2014-task9-test-B-tweets-only-some-POS-classes.txt'
#SEMEVAL_2017_TEST_FILE 	       = PREFIX_PATH + 'datasets/test/SemEval2017-task4-test.subtask-A.english.txt'
#STS_TRAIN_FILE                 = PREFIX_PATH + 'datasets/train/STS/STStrain.1600000.processed.noemoticon.EDITED.csv'
#STS_TEST_FILE                  = PREFIX_PATH + 'datasets/test/STS/STS_Gold_All.txt'
#DICTIONARY_POSITIVE_WORDS      = PREFIX_PATH + 'dictionaries/positive-words.txt'
#DICTIONARY_NEGATIVE_WORDS      = PREFIX_PATH + 'dictionaries/negative-words.txt'
#DICTIONARY_POSITIVE_HASHTAGS   = PREFIX_PATH + 'dictionaries/positive-hashtags.txt'
#DICTIONARY_NEGATIVE_HASHTAGS   = PREFIX_PATH + 'dictionaries/negative-hashtags.txt'
#DICTIONARY_POSITIVE_EMOTICONS  = PREFIX_PATH + 'dictionaries/positive-emoticons.txt'
#DICTIONARY_NEGATIVE_EMOTICONS  = PREFIX_PATH + 'dictionaries/negative-emoticons.txt'
#DICTIONARY_NEGATING_WORDS      = PREFIX_PATH + 'dictionaries/negating-word-list.txt'
#DICTIONARY_BOOSTER_WORDS       = PREFIX_PATH + 'dictionaries/boosterWords.txt'
#DICTIONARY_SENTIWORDNET        = PREFIX_PATH + 'dictionaries/SentiWordNet_3.0.0_20130122.txt'
#DICTIONARY_SENTIWORDNET_FOLDER = PREFIX_PATH + 'dictionaries/Sentiwordnet/'
#DICTIONARY_AFFIN               = PREFIX_PATH + 'dictionaries/affin.txt'
#DICTIONARY_SLANG               = PREFIX_PATH + 'dictionaries/slangSD.txt'
#DICTIONARY_VADER               = PREFIX_PATH + 'dictionaries/vaderLexicon.txt'
#DICTIONARY_SEMEVAL2015         = PREFIX_PATH + 'dictionaries/SemEval2015-English-Twitter-Lexicon.txt'
#DICTIONARY_EFFECT              = PREFIX_PATH + 'dictionaries/EffectWordNet.tff'
#DICTIONARY_EFFECT_FOLDER       = PREFIX_PATH + 'dictionaries/Effect/'
#DICTIONARY_NRC                 = PREFIX_PATH + 'dictionaries/nrc_words.txt'
#DICTIONARY_GENERAL_INQUIRER    = PREFIX_PATH + 'dictionaries/general-inquirer.txt'
#DICTIONARY_S140                = PREFIX_PATH + 'dictionaries/sentiment140_unigram.txt'
#DICTIONARY_MPQA                = PREFIX_PATH + 'dictionaries/mpqa.txt'

#best_accuracy = 0

#best_precision_positive = 0
#best_precision_negative = 0
#best_precision_neutral  = 0
#best_precision_avg      = 0

#best_recall_positive = 0
#best_recall_negative = 0
#best_recall_neutral  = 0
#best_recall_avg      = 0

#best_f1_positive = 0
#best_f1_negative = 0
#best_f1_neutral  = 0
#best_f1_avg      = 0
#best_f1_positive_negative_avg = 0

#best_precision_avg_function = ""
#best_recall_avg_function    = ""
#best_f1_avg_function        = ""

#precision_positive_history = []
#precision_negative_history = []
#precision_neutral_history  = []
#recall_positive_history    = []
#recall_negative_history    = []
#recall_neutral_history     = []
#f1_positive_history        = []
#f1_negative_history        = []
#f1_neutral_history         = []

#tweets_2013             = []
#tweets_2013_score       = []
#tweets_2013_score_svm   = []
#tweets_2013_score_naive = []
#tweets_2013_score_MS    = []
#tweets_2013_score_LReg  = []
#tweets_2013_score_S140  = []
#tweets_2013_score_RFor  = []
#tweets_2013_score_SGD   = []
#tweets_2013_score_ESumNoPG = []
#tweets_2013_positive    = 0
#tweets_2013_negative    = 0
#tweets_2013_neutral     = 0

#tweets_2014             = []
#tweets_2014_score       = []
#tweets_2014_score_svm   = []
#tweets_2014_score_naive = []
#tweets_2014_score_MS    = []
#tweets_2014_score_LReg  = []
#tweets_2014_score_S140  = []
#tweets_2014_score_RFor  = []
#tweets_2014_score_SGD   = []
#tweets_2014_score_ESumNoPG = []
#tweets_2014_positive    = 0
#tweets_2014_negative    = 0
#tweets_2014_neutral     = 0

#tweets_liveJournal2014             = []
#tweets_liveJournal2014_score       = []
#tweets_liveJournal2014_score_svm   = []
#tweets_liveJournal2014_score_naive = []
#tweets_liveJournal2014_score_MS    = []
#tweets_liveJournal2014_score_LReg  = []
#tweets_liveJournal2014_score_S140  = []
#tweets_liveJournal2014_score_RFor  = []
#tweets_liveJournal2014_score_SGD   = []
#tweets_liveJournal2014_score_ESumNoPG = []
#tweets_liveJournal2014_positive    = 0
#tweets_liveJournal2014_negative    = 0
#tweets_liveJournal2014_neutral     = 0

#tweets_2014_sarcasm             = []
#tweets_2014_sarcasm_score       = []
#tweets_2014_sarcasm_score_svm   = []
#tweets_2014_sarcasm_score_naive = []
#tweets_2014_sarcasm_score_MS    = []
#tweets_2014_sarcasm_score_LReg  = []
#tweets_2014_sarcasm_score_S140  = []
#tweets_2014_sarcasm_score_RFor  = []
#tweets_2014_sarcasm_score_SGD   = []
#tweets_2014_sarcasm_score_ESumNoPG = []
#tweets_2014_sarcasm_positive    = 0
#tweets_2014_sarcasm_negative    = 0
#tweets_2014_sarcasm_neutral     = 0

#sms_2013             = []
#sms_2013_score       = []
#sms_2013_score_svm   = []
#sms_2013_score_naive = []
#sms_2013_score_MS    = []
#sms_2013_score_LReg  = []
#sms_2013_score_S140  = []
#sms_2013_score_RFor  = []
#sms_2013_score_SGD   = []
#sms_2013_score_ESumNoPG = []
#sms_2013_positive    = 0
#sms_2013_negative    = 0
#sms_2013_neutral     = 0

#all_messages_in_file_order         = []
#all_polarities_in_file_order       = []
#all_polarities_in_file_order_svm   = []
#all_polarities_in_file_order_naive = []
#all_polarities_in_file_order_MS    = []
#all_polarities_in_file_order_LReg  = []
#all_polarities_in_file_order_S140  = []
#all_polarities_in_file_order_RFor  = []
#all_polarities_in_file_order_SGD   = []
#all_polarities_in_file_order_ESumNoPG = []


# Using python dictionary to improve the search performance

#dic_positive_semeval2015  = {}
#dic_negative_semeval2015  = {}
#dic_positive_slang        = {}
#dic_negative_slang        = {}
#dic_positive_affin        = {}
#dic_negative_affin        = {}
#dic_positive_sentiwordnet = {}
#dic_negative_sentiwordnet = {}
#dic_positive_effect       = {}
#dic_negative_effect       = {}
#dic_positive_vader        = {}
#dic_negative_vader        = {}
#dic_positive_nrc          = {}
#dic_negative_nrc          = {}
#dic_positive_gi           = {}
#dic_negative_gi           = {}
#dic_positive_s140         = {}
#dic_negative_s140         = {}
#dic_positive_mpqa         = {}
#dic_negative_mpqa         = {}

#root_constraint = False
#root_function = "polSumAVGWeights"
#root_function = "polaritySumAVG"
#root_functions = ["polaritySumAVGUsingWeights", "if_then_else"]
#root_decreased_value = 0.2

#massive_functions_constraint = True
#massive_function = "polSumAVGWeights"
#massive_functions_max = 1

#neutral_range_constraint = False

#liu_weight       	 = 1
#sentiwordnet_weight = 1
#affin_weight        = 1
#vader_weight        = 1
#slang_weight        = 1
#effect_weight       = 1
#semeval2015_weight  = 1
#nrc_weight          = 1
#gi_weight           = 1
#s140_weight         = 1
#mpqa_weight         = 1

#train_using_folds_gp_ensemble = False # This needs to be False for while