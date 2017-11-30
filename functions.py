# Airton Bordin Junior
# airtonbjunior@gmail.com
# Federal University of Goias (UFG)
# Computer Science Master's Degree

import math
import re 
import string
import time
import codecs

from stemming.porter2 import stem
from nltk.stem import WordNetLemmatizer

import matplotlib.pyplot as plt
import numpy as np

import variables

def getDictionary():
    start = time.time()
    print("[loading dictionary]")

# BING LIU POSITIVE/NEGATIVE WORDS
    with open(variables.DICTIONARY_POSITIVE_WORDS, 'r') as inF:
        variables.use_dic_liu = True
        for line in inF:
            variables.dic_positive_words.append(line.lower().strip())

    with codecs.open(variables.DICTIONARY_NEGATIVE_WORDS, "r", "latin-1") as inF2:
    #with open(variables.DICTIONARY_NEGATIVE_WORDS, 'r') as inF2:
        for line2 in inF2:
            variables.dic_negative_words.append(line2.lower().strip())

    with open(variables.DICTIONARY_POSITIVE_HASHTAGS, 'r') as inF3:
        for line3 in inF3:
            variables.dic_positive_hashtags.append(line3.lower().strip())

    with open(variables.DICTIONARY_NEGATIVE_HASHTAGS, 'r') as inF4:
        for line4 in inF4:
            variables.dic_negative_hashtags.append(line4.lower().strip())            

    with open(variables.DICTIONARY_POSITIVE_EMOTICONS, 'r') as inF5:
        for line5 in inF5:
            variables.dic_positive_emoticons.append(line5.strip()) 

    with open(variables.DICTIONARY_NEGATIVE_EMOTICONS, 'r') as inF6:
        for line6 in inF6:
            variables.dic_negative_emoticons.append(line6.strip())             

    with open(variables.DICTIONARY_NEGATING_WORDS, 'r') as inF7:
        for line7 in inF7:
            variables.dic_negation_words.append(line7.strip()) 

    with open(variables.DICTIONARY_BOOSTER_WORDS) as inF8:
        for line8 in inF8:
            variables.dic_booster_words.append(line8.strip())

# SENTIWORDNET            
    with open(variables.DICTIONARY_SENTIWORDNET, 'r') as inF9:
        variables.use_dic_sentiwordnet = True
        for line9 in inF9:
            if float(line9.split("\t")[2]) > float(line9.split("\t")[3]): #positive greater than negative
                words = line9.split("\t")[4].lower().strip().split()
                for word in words:
                    if not "_" in word and not word in variables.dic_positive_words:
                        variables.dic_positive_words.append(word[:word.find("#")])
                        #print("POSITIVE: " + word[:word.find("#")])
            
            elif float(line9.split("\t")[2]) < float(line9.split("\t")[3]):
                words = line9.split("\t")[4].lower().strip().split()
                for word in words:
                    if not "_" in word and not word in variables.dic_negative_words:
                        variables.dic_negative_words.append(word[:word.find("#")])
                        #print("NEGATIVE: " + word[:word.find("#")])

# EFFECT LEXICON
#    with open('dictionaries/goldStandard.tff', 'r') as inF8:
#        for line8 in inF8:
#            if (line8.split()[1] == "+Effect"):
#                for word in line8.split()[2].split(","):
#                    if word not in variables.dic_negative_words and word not in variables.dic_positive_words:
#                        variables.dic_positive_words.append(word)
#                        #print("[positive word]: " + word)
#
#            elif (line8.split()[1] == "-Effect"):
#                for word in line8.split()[2].split(","):
#                    if word not in variables.dic_negative_words and word not in variables.dic_positive_words:
#                        variables.dic_negative_words.append(word) 
#                        #print("[negative word]: " + word)

# ENGLISH TWITTER LEXICON SEMEVAL 2015

    with open('dictionaries/SemEval2015-English-Twitter-Lexicon.txt', 'r') as inF7:
        for line7 in inF7:
            #removing composite words for while 
            if float(line7.split("\t")[0]) > 0 and not ' ' in line7.split("\t")[1].strip():
                if "#" in line7.split("\t")[1].strip():
                    variables.dic_positive_hashtags.append(line7.split("\t")[1].strip()[1:])
                else:
                    variables.dic_positive_words.append(line7.split("\t")[1].strip())
            elif float(line7.split("\t")[0]) < 0 and not ' ' in line7.split("\t")[1].strip():
                if "#" in line7.split("\t")[1].strip():
                    variables.dic_negative_hashtags.append(line7.split("\t")[1].strip()[1:])
                else:
                    variables.dic_negative_words.append(line7.split("\t")[1].strip())

# AFFIN 
    with open(variables.DICTIONARY_AFFIN, 'r') as inF:
        variables.use_dic_affin = True
        for line in inF:
            if float(line.split("\t")[1].strip()) > 0:
                #if not line.split("\t")[0].strip() in variables.dic_positive_words and not line.split("\t")[0].strip() in variables.dic_negative_words:
                variables.dic_positive_words_affin.append(line.split("\t")[0].strip())
                variables.dic_positive_value_affin.append(float(line.split("\t")[1].strip()))
                #print("POSITIVE AFFIN " + line.split("\t")[0] + " " + line.split("\t")[1].strip())
            else:
                #if not line.split("\t")[0].strip() in variables.dic_positive_words and not line.split("\t")[0].strip() in variables.dic_negative_words:
                variables.dic_negative_words_affin.append(line.split("\t")[0].strip())
                variables.dic_negative_value_affin.append(float(line.split("\t")[1].strip()))
                #print("NEGATIVE AFFIN " + line.split("\t")[0] + " " + line.split("\t")[1].strip())


# SLANG
    with codecs.open(variables.DICTIONARY_SLANG, "r", "latin-1") as inF:
    #with open(variables.DICTIONARY_SLANG, 'r') as inF:
        variables.use_dic_slang = True
        for line in inF:    
            if float(line.split("\t")[1].strip()) > 0:
                if not line.split("\t")[0].strip() in variables.dic_positive_words and not line.split("\t")[0].strip() in variables.dic_negative_words:
                    variables.dic_positive_words.append(line.split("\t")[0].strip())

            elif float(line.split("\t")[1].strip()) < 0:
                if not line.split("\t")[0].strip() in variables.dic_positive_words and not line.split("\t")[0].strip() in variables.dic_negative_words:
                    variables.dic_negative_words.append(line.split("\t")[0].strip())


# Vader Lexicon
    with open(variables.DICTIONARY_VADER, 'r') as inF:
        variables.use_dic_vader = True
        for line in inF:
            if float(line.split("\t")[1].strip()) > 0:
                if not line.split("\t")[0].strip() in variables.dic_positive_words and not line.split("\t")[0].strip() in variables.dic_negative_words:
                    variables.dic_positive_words_vader.append(line.split("\t")[0].strip())
                    variables.dic_positive_value_vader.append(float(line.split("\t")[1].strip()))
                    #variables.dic_positive_words.append(line.split("\t")[0].strip())
                    #print("POSITIVE " + line.split("\t")[0])
            else:
                if not line.split("\t")[0].strip() in variables.dic_positive_words and not line.split("\t")[0].strip() in variables.dic_negative_words:
                    variables.dic_negative_words_vader.append(line.split("\t")[0].strip())
                    variables.dic_negative_value_vader.append(float(line.split("\t")[1].strip()))
                    #variables.dic_negative_words.append(line.split("\t")[0].strip())
                    #print("NEGATIVE " + line.split("\t")[0])
    
# Sentiment140 Lexicon
    with open(variables.DICTIONARY_SENTIMENT140, 'r') as inF:
        variables.use_dic_sentiment140 = True
        for line in inF:
            if float(line.split("\t")[1].strip()) > 0:
                if not line.split("\t")[0].strip() in variables.dic_positive_words and not line.split("\t")[0].strip() in variables.dic_negative_words:
                    variables.dic_positive_words_s140.append(line.split("\t")[0].strip())
                    variables.dic_positive_value_s140.append(float(line.split("\t")[1].strip()))
                    #variables.dic_positive_words.append(line.split("\t")[0].strip())
                    #print("POSITIVE " + line.split("\t")[0])
            else:
                if not line.split("\t")[0].strip() in variables.dic_positive_words and not line.split("\t")[0].strip() in variables.dic_negative_words:
                    variables.dic_negative_words_s140.append(line.split("\t")[0].strip())
                    variables.dic_negative_value_s140.append(float(line.split("\t")[1].strip()))
                    #variables.dic_negative_words.append(line.split("\t")[0].strip())
                    #print("NEGATIVE " + line.split("\t")[0])



    # Performance improvement test
    variables.dic_positive_words = set(variables.dic_positive_words)
    variables.dic_negative_words = set(variables.dic_negative_words)
    variables.dic_positive_hashtags = set(variables.dic_positive_hashtags)
    variables.dic_negative_hashtags = set(variables.dic_negative_hashtags)
    variables.dic_positive_emoticons = set(variables.dic_positive_emoticons)
    variables.dic_negative_emoticons = set(variables.dic_negative_emoticons)

    end = time.time()
    print("[dictionary loaded - words, hashtags and emoticons][" + str(format(end - start, '.3g')) + " seconds]\n")


# get tweets from id (SEMEVAL 2014 database)
def loadTrainTweets():
    start = time.time()
    print("[loading tweets from train file Semeval 2014]")

    tweets_loaded = 0

    with open(variables.SEMEVAL_TRAIN_FILE, 'r') as inF:
        for line in inF:
            if tweets_loaded < variables.MAX_ANALYSIS_TWEETS:
                tweet_parsed = line.split("\t")
                try:
                    # i'm ignoring the neutral tweets
                    if(tweet_parsed[2] != "neutral"):
                        if(tweet_parsed[2] == "positive"):
                            if(variables.positive_tweets < variables.MAX_POSITIVES_TWEETS):
                                variables.positive_tweets += 1
                                variables.tweets_semeval.append(tweet_parsed[3])
                                variables.tweets_semeval_score.append(1)
                                tweets_loaded += 1
                        else:
                            if(variables.negative_tweets < variables.MAX_NEGATIVES_TWEETS):
                                variables.negative_tweets += 1
                                variables.tweets_semeval.append(tweet_parsed[3])
                                variables.tweets_semeval_score.append(-1)
                                tweets_loaded += 1
                    else:
                        if(variables.neutral_tweets < variables.MAX_NEUTRAL_TWEETS):
                            variables.tweets_semeval.append(tweet_parsed[3])
                            variables.tweets_semeval_score.append(0)
                            variables.neutral_tweets += 1
                            tweets_loaded += 1
                except:
                    print("exception")
                    continue
    
    end = time.time()
    print("[train tweets loaded][" + str(format(end - start, '.3g')) + " seconds]\n")


# get the test tweets from Semeval 2014 task 9
def loadTestTweets():
    start = time.time()
    print("[loading tweets from test file Semeval 2014]")

    tweets_loaded = 0

    with open(variables.SEMEVAL_TEST_FILE, 'r') as inF:
        for line in inF:
            if tweets_loaded < variables.MAX_ANALYSIS_TWEETS:
                tweet_parsed = line.split("\t")
                try:
                    if tweet_parsed[1] == "Twitter2013":
                        variables.tweets_2013.append(tweet_parsed[2])
                        
                        if tweet_parsed[0] == "positive":
                            variables.tweets_2013_score.append(1)
                            variables.tweets_2013_positive += 1
                        
                        elif tweet_parsed[0] == "negative":
                            variables.tweets_2013_score.append(-1)
                            variables.tweets_2013_negative += 1
                        
                        elif tweet_parsed[0] == "neutral":
                            variables.tweets_2013_score.append(0)
                            variables.tweets_2013_neutral += 1

                    elif tweet_parsed[1] == "Twitter2014":
                        variables.tweets_2014.append(tweet_parsed[2])
                        
                        if tweet_parsed[0] == "positive":
                            variables.tweets_2014_score.append(1)
                            variables.tweets_2014_positive += 1
                        
                        elif tweet_parsed[0] == "negative":
                            variables.tweets_2014_score.append(-1)
                            variables.tweets_2014_negative += 1
                        
                        elif tweet_parsed[0] == "neutral":
                            variables.tweets_2014_score.append(0)
                            variables.tweets_2014_neutral += 1

                    elif tweet_parsed[1] == "SMS2013":
                        variables.sms_2013.append(tweet_parsed[2])
                        
                        if tweet_parsed[0] == "positive":
                            variables.sms_2013_score.append(1)
                            variables.sms_2013_positive += 1
                        
                        elif tweet_parsed[0] == "negative":
                            variables.sms_2013_score.append(-1)
                            variables.sms_2013_negative += 1
                        
                        elif tweet_parsed[0] == "neutral":
                            variables.sms_2013_score.append(0)
                            variables.sms_2013_neutral += 1

                    elif tweet_parsed[1] == "LiveJournal2014":
                        variables.tweets_liveJournal2014.append(tweet_parsed[2])
                        
                        if tweet_parsed[0] == "positive":
                            variables.tweets_liveJournal2014_score.append(1)
                            variables.tweets_liveJournal2014_positive += 1
                        
                        elif tweet_parsed[0] == "negative":
                            variables.tweets_liveJournal2014_score.append(-1)
                            variables.tweets_liveJournal2014_negative += 1
                        
                        elif tweet_parsed[0] == "neutral":
                            variables.tweets_liveJournal2014_score.append(0)
                            variables.tweets_liveJournal2014_neutral += 1

                    elif tweet_parsed[1] == "Twitter2014Sarcasm":
                        variables.tweets_2014_sarcasm.append(tweet_parsed[2])
                        
                        if tweet_parsed[0] == "positive":
                            variables.tweets_2014_sarcasm_score.append(1)
                            variables.tweets_2014_sarcasm_positive += 1
                        
                        elif tweet_parsed[0] == "negative":
                            variables.tweets_2014_sarcasm_score.append(-1)
                            variables.tweets_2014_sarcasm_negative += 1
                        
                        elif tweet_parsed[0] == "neutral":
                            variables.tweets_2014_sarcasm_score.append(0)                                                           
                            variables.tweets_2014_sarcasm_neutral += 1

                    tweets_loaded += 1

                except:
                    #print("exception")
                    continue
    
    end = time.time()
    print("[test tweets loaded][" + str(format(end - start, '.3g')) + " seconds]\n")


def add(left, right):
    return left + right


def sub(left, right):
    return left - right


def mul(left, right):
    return left * right


def exp(par):
    return math.exp(par)


def cos(par):
    return math.cos(par)


def sin(par):
    return math.sin(par)

# Protected Div (check division by zero)
def protectedDiv(left, right):
    try:
        return left / right
    except:
        return 1


# Log
def protectedLog(value):
    try:
        return math.log10(value)
    except:
        return 1    


# Sqrt
def protectedSqrt(value):
    try:
        return math.sqrt(value)
    except:
        return 1  


def invertSignal(val):
    return -val


def negativeWordsQuantity(phrase):
    #negative_words = variables.negative_words_quantity_cache

    #if negative_words > -1:
    negative_words = 0
    words = phrase.split()
    
    for word in words:
        if word in variables.dic_negative_words:
            negative_words += 1

    #variables.negative_words_quantity_cache = negative_words

    return negative_words


def positiveWordsQuantity(phrase):
    #positive_words = variables.positive_words_quantity_cache
    
    #if positive_words > -1:
    positive_words = 0
    words = phrase.split()
    
    for word in words:
        if word in variables.dic_positive_words:
            positive_words += 1
    
    #variables.positive_words_quantity_cache = positive_words
    
    return positive_words    


def polaritySum2(phrase):
    total_sum = 0
    dic_quantity = 0
    index = 0
    invert = False
    booster = False
    boosterAndInverter = False

    phrase = phrase.strip()
    words = phrase.split()

    for word in words:
        # Check booster and inverter words
        if index > 0 and words[index-1] == "insidenoteboosterword" and words[index-2] == "insidenoteinverterword":
            boosterAndInverter = True
        elif (index > 0 and words[index-1] == "insidenoteboosterword") or (index < len(words) - 1 and words[index+1] == "insidenoteboosterword" and (words[index-1] != "insidenoteboosterword" or index == 0)):
            booster = True
        elif index > 0 and words[index-1] == "insidenoteinverterword":
            invert = True

        # LIU pos/neg words
        if variables.use_dic_liu:
            if word in variables.dic_positive_words:
                if invert:
                    total_sum -= 1
                elif booster:
                    total_sum += 2
                elif boosterAndInverter:
                    total_sum -= 2
                else: 
                    total_sum += 1

                dic_quantity += 1

            elif word in variables.dic_negative_words:
                if invert:
                    total_sum += 1
                elif booster:
                    total_sum -= 2
                elif boosterAndInverter:
                    total_sum += 2
                else: 
                    total_sum -= 1

                dic_quantity += 1

        # AFFIN
        if variables.use_dic_affin:
            if word in variables.dic_positive_words_affin:
                if invert:
                    total_sum -= variables.dic_positive_value_affin[variables.dic_positive_words_affin.index(word)]
                elif booster:
                    total_sum += 2 * variables.dic_positive_value_affin[variables.dic_positive_words_affin.index(word)]
                else:
                    total_sum += variables.dic_positive_value_affin[variables.dic_positive_words_affin.index(word)]
                
                dic_quantity += 1
            elif word in variables.dic_negative_words_affin:
                if invert:
                    total_sum -= variables.dic_negative_value_affin[variables.dic_negative_words_affin.index(word)]
                elif booster:
                    total_sum += 2 * variables.dic_negative_value_affin[variables.dic_negative_words_affin.index(word)]
                else:
                    total_sum += variables.dic_negative_value_affin[variables.dic_negative_words_affin.index(word)]
                
                dic_quantity += 1

        # VADER
        if variables.use_dic_vader:
            if word in variables.dic_positive_words_vader:
                if invert:
                    total_sum -= variables.dic_positive_value_vader[variables.dic_positive_words_vader.index(word)]
                elif booster:
                    total_sum += 2 * variables.dic_positive_value_vader[variables.dic_positive_words_vader.index(word)]
                else:
                    total_sum += variables.dic_positive_value_vader[variables.dic_positive_words_vader.index(word)]
                
                dic_quantity += 1
            elif word in variables.dic_negative_words_vader:
                if invert:
                    total_sum -= variables.dic_negative_value_vader[variables.dic_negative_words_vader.index(word)]
                elif booster:
                    total_sum += 2 * variables.dic_negative_value_vader[variables.dic_negative_words_vader.index(word)]
                else:
                    total_sum += variables.dic_negative_value_vader[variables.dic_negative_words_vader.index(word)]
                
                dic_quantity += 1

        # SENTIMENT140
        if variables.use_dic_sentiment140:
            if word in variables.dic_positive_words_s140:
                if invert:
                    total_sum -= variables.dic_positive_value_s140[variables.dic_positive_words_s140.index(word)]
                elif booster:
                    total_sum += 2 * variables.dic_positive_value_s140[variables.dic_positive_words_s140.index(word)]
                else:
                    total_sum += variables.dic_positive_value_s140[variables.dic_positive_words_s140.index(word)]
                
                dic_quantity += 1
            elif word in variables.dic_negative_words_s140:
                if invert:
                    total_sum -= variables.dic_negative_value_s140[variables.dic_negative_words_s140.index(word)]
                elif booster:
                    total_sum += 2 * variables.dic_negative_value_s140[variables.dic_negative_words_s140.index(word)]
                else:
                    total_sum += variables.dic_negative_value_s140[variables.dic_negative_words_s140.index(word)]
                
                dic_quantity += 1
        

        index += 1 # word of phrase
        invert = False
        booster = False
        boosterAndInverter = False
    

    return total_sum # change this when i have more dictionaries

        #if dic_quantity > 1:
            #return round(total_sum/dic_quantity, 4)
        #else:
            #return total_sum


def polaritySumAVG(phrase):
    total_sum = 0
    total_sum_return = 0
    dic_quantity = 0
    index = 0
    invert = False
    booster = False
    boosterAndInverter = False

    phrase = phrase.strip()
    words = phrase.split()

    for word in words:
        # Check booster and inverter words
        if index > 0 and words[index-1] == "insidenoteboosterword" and words[index-2] == "insidenoteinverterword":
            boosterAndInverter = True
        elif (index > 0 and words[index-1] == "insidenoteboosterword") or (index < len(words) - 1 and words[index+1] == "insidenoteboosterword" and (words[index-1] != "insidenoteboosterword" or index == 0)):
            booster = True
        elif index > 0 and words[index-1] == "insidenoteinverterword":
            invert = True

        # LIU pos/neg words
        if variables.use_dic_liu:
            if word in variables.dic_positive_words:
                if invert:
                    total_sum -= 1
                elif booster:
                    total_sum += 2
                elif boosterAndInverter:
                    total_sum -= 2
                else: 
                    total_sum += 1

                dic_quantity += 1

            elif word in variables.dic_negative_words:
                if invert:
                    total_sum += 1
                elif booster:
                    total_sum -= 2
                elif boosterAndInverter:
                    total_sum += 2
                else: 
                    total_sum -= 1

                dic_quantity += 1

        # AFFIN
        if variables.use_dic_affin:
            if word in variables.dic_positive_words_affin:
                if invert:
                    total_sum -= variables.dic_positive_value_affin[variables.dic_positive_words_affin.index(word)]
                elif booster:
                    total_sum += 2 * variables.dic_positive_value_affin[variables.dic_positive_words_affin.index(word)]
                else:
                    total_sum += variables.dic_positive_value_affin[variables.dic_positive_words_affin.index(word)]
                
                dic_quantity += 1
            elif word in variables.dic_negative_words_affin:
                if invert:
                    total_sum -= variables.dic_negative_value_affin[variables.dic_negative_words_affin.index(word)]
                elif booster:
                    total_sum += 2 * variables.dic_negative_value_affin[variables.dic_negative_words_affin.index(word)]
                else:
                    total_sum += variables.dic_negative_value_affin[variables.dic_negative_words_affin.index(word)]
                
                dic_quantity += 1

        # VADER
        if variables.use_dic_vader:
            if word in variables.dic_positive_words_vader:
                if invert:
                    total_sum -= variables.dic_positive_value_vader[variables.dic_positive_words_vader.index(word)]
                elif booster:
                    total_sum += 2 * variables.dic_positive_value_vader[variables.dic_positive_words_vader.index(word)]
                else:
                    total_sum += variables.dic_positive_value_vader[variables.dic_positive_words_vader.index(word)]
                
                dic_quantity += 1
            elif word in variables.dic_negative_words_vader:
                if invert:
                    total_sum -= variables.dic_negative_value_vader[variables.dic_negative_words_vader.index(word)]
                elif booster:
                    total_sum += 2 * variables.dic_negative_value_vader[variables.dic_negative_words_vader.index(word)]
                else:
                    total_sum += variables.dic_negative_value_vader[variables.dic_negative_words_vader.index(word)]
                
                dic_quantity += 1

        # SENTIMENT140
        if variables.use_dic_sentiment140:
            if word in variables.dic_positive_words_s140:
                if invert:
                    total_sum -= variables.dic_positive_value_s140[variables.dic_positive_words_s140.index(word)]
                elif booster:
                    total_sum += 2 * variables.dic_positive_value_s140[variables.dic_positive_words_s140.index(word)]
                else:
                    total_sum += variables.dic_positive_value_s140[variables.dic_positive_words_s140.index(word)]
                
                dic_quantity += 1
            elif word in variables.dic_negative_words_s140:
                if invert:
                    total_sum -= variables.dic_negative_value_s140[variables.dic_negative_words_s140.index(word)]
                elif booster:
                    total_sum += 2 * variables.dic_negative_value_s140[variables.dic_negative_words_s140.index(word)]
                else:
                    total_sum += variables.dic_negative_value_s140[variables.dic_negative_words_s140.index(word)]
                
                dic_quantity += 1
        

        if(dic_quantity > 1):
            #print("More than one dictionary " + str(dic_quantity) + " - word: " + word)
            total_sum_return += round(total_sum/dic_quantity, 4)
        elif(dic_quantity == 1):
            total_sum_return += total_sum

        dic_quantity = 0
        total_sum = 0

        index += 1 # word of phrase
        invert = False
        booster = False
        boosterAndInverter = False
    

    return total_sum_return


def replaceNegatingWords(phrase):
    phrase = phrase.lower()

    splitted_phrase = phrase.split()

    if(splitted_phrase[-1] == "alreadynegatedbefore"):
        return phrase  

    if len(splitted_phrase) > 0 and splitted_phrase[0] in variables.dic_negation_words:
        phrase_list = splitted_phrase
        phrase_list[0] = "insidenoteinverterword"
        phrase = ' '.join(phrase_list)

    for negation_word in variables.dic_negation_words:
        negation_word = " " + negation_word + " "
        if negation_word in phrase:
        #if phrase.find(negation_word) > -1:
            phrase = phrase.replace(negation_word, " insidenoteinverterword ")

    return phrase + " alreadynegatedbefore"


def replaceBoosterWords(phrase):
    phrase = phrase + " "
    phrase = phrase.lower()

    if(phrase.split()[-1] == "alreadyboosteredbefore"):
        return phrase
    
    if len(phrase.split()) > 0 and phrase.split()[0] in variables.dic_booster_words:
        phrase_list = phrase.split()
        phrase_list[0] = "insidenoteboosterword"
        phrase = ' '.join(phrase_list)

    for booster_word in variables.dic_booster_words:
        booster_word = " " + booster_word + " "
        if booster_word in phrase: 
        #if phrase.find(booster_word) > -1:
            #print("has booster " + booster_word)
            phrase = phrase.replace(booster_word, " insidenoteboosterword ")

        elif booster_word[-1] in phrase:
        #elif phrase.find(booster_word[-1]) > -1:
            phrase = phrase.replace(booster_word, " insidenoteboosterword ")

    return phrase + " alreadyboosteredbefore" 


def boostUpperCase(phrase):
    words = phrase.split()
    phrase_return = ""

    for word in words:
        if word.isupper():
            phrase_return += "insidenoteboosteruppercase " + word + " "
        else:
            phrase_return += word + " "

    return phrase_return 


# sum of the hashtag polarities only
def hashtagPolaritySum(phrase):
    return positiveHashtags(phrase) - negativeHashtags(phrase)


# sum of the emoticons polarities only
def emoticonsPolaritySum(phrase):
    return positiveEmoticons(phrase) - negativeEmoticons(phrase)


def positiveEmoticons(phrase):
    words = phrase.split()

    total_sum = 0

    for word in words:
        if word.strip() in variables.dic_positive_emoticons:
            total_sum += 1               

    return total_sum


def negativeEmoticons(phrase):
    words = phrase.split()

    total_sum = 0

    for word in words:
        if word.strip() in variables.dic_negative_emoticons:
            total_sum += 1               

    return total_sum


# Positive Hashtags
def positiveHashtags(phrase):
    total = 0
    if "#" in phrase:
        #print("has hashtag")
        hashtags = re.findall(r"#(\w+)", phrase)

        for hashtag in hashtags:
            if hashtag.lower().strip() in variables.dic_positive_hashtags:
                total += 1 
            else:
                if hashtag.lower().strip() in variables.dic_positive_words:
                    total += 1 

    return total


# Negative Hashtags
def negativeHashtags(phrase):
    total = 0
    if "#" in phrase:
        #print("has hashtag")
        hashtags = re.findall(r"#(\w+)", phrase)

        for hashtag in hashtags:
            if hashtag.lower().strip() in variables.dic_negative_hashtags:
                total += 1 
            else:
                if hashtag.lower().strip() in variables.dic_negative_words:
                    total += 1 

    return total


# Check if has hashtags on phrase
def hasHashtag(phrase):
    return True if "#" in phrase else False


# Check if has emoticons on phrase
def hasEmoticons(phrase):
    words = phrase.split()

    for word in words:
        if (word in variables.dic_negative_emoticons) or (word in variables.dic_positive_emoticons):
            return True

    return False


# logic operators
# Define a new if-then-else function
def if_then_else(input, output1, output2):
    if input: return output1
    else: return output2


def removeStopWords(phrase):
    words = phrase.split()
    return_phrase = ""

    for word in words:
        if word not in variables.stop_words:
            return_phrase += word + " "               

    return return_phrase


def stemmingText(phrase):
    words = phrase.split()

    if len(words) > 0 and words[len(words)-1] == "insidenotestemmedphrase":
        return phrase

    stemmed_phrase = ""

    for word in words:
        stemmed_phrase += stem(word) + " "               

    stemmed_phrase += "insidenotestemmedphrase"

    return stemmed_phrase.strip()


def lemmingText(phrase):
    lemmatizer = WordNetLemmatizer()
    words = phrase.split()

    if len(words) > 0 and words[len(words)-1] == "insidenotelemmatizedphrase":
        return phrase

    lemmed_phrase = ""

    for word in words:
        # I'm always considering that the word is a verb
        lemmed_phrase += lemmatizer.lemmatize(word, 'v') + " "               

    lemmed_phrase += "insidenotelemmatizedphrase"

    return lemmed_phrase.strip()


## NOTE: I'm copying the phrase[str] variable for didact reasons in this functions
##       Same for variable named phrase_return

def removeLinks(phrase):
    phrase_copy = phrase
    phrase_return = re.sub(r'http\S+', '', phrase_copy, flags=re.MULTILINE)
    return phrase_return


def removeEllipsis(phrase):
    phrase_copy = phrase
    phrase_return = re.sub('\.{3}', ' ', phrase_copy)
    return phrase_return


def removeDots(phrase):
    phrase_copy = phrase
    return re.sub('\.', ' ', phrase_copy)


def removeAllPonctuation(phrase):
    phrase_copy = phrase
    return phrase_copy.translate(str.maketrans('','',string.punctuation.replace("-", ""))) # keep hyphens


# Evaluate the test messages using the model
def evaluateMessages(base, model):
    # parameters to calc the metrics
    true_positive  = 0
    true_negative  = 0
    false_positive = 0
    false_negative = 0
    true_neutral   = 0
    false_neutral  = 0

    accuracy = 0

    precision_positive = 0
    precision_negative = 0
    precision_neutral = 0
    precision_avg = 0

    recall_positive = 0
    recall_negative = 0
    recall_neutral = 0
    recall_avg = 0

    f1_positive = 0
    f1_negative = 0
    f1_neutral  = 0
    f1_avg = 0
    f1_positive_negative_avg = 0

    false_neutral_log  = 0
    false_negative_log = 0

    message = ""
    model_analysis = ""
    result = 0

    messages = []
    messages_score = []
    messages_positive = 0
    messages_negative = 0
    messages_neutral  = 0
    
    if len(variables.tweets_2013) == 0:
        loadTestTweets()

    if base == "tweets2013":
        messages = variables.tweets_2013
        messages_score = variables.tweets_2013_score
        messages_positive = variables.tweets_2013_positive
        messages_negative = variables.tweets_2013_negative
        messages_neutral  = variables.tweets_2013_neutral
    elif base == "tweets2014":
        messages = variables.tweets_2014
        messages_score = variables.tweets_2014_score
        messages_positive = variables.tweets_2014_positive
        messages_negative = variables.tweets_2014_negative
        messages_neutral  = variables.tweets_2014_neutral
    elif base == "livejournal":
        messages = variables.tweets_liveJournal2014
        messages_score = variables.tweets_liveJournal2014_score
        messages_positive = variables.tweets_liveJournal2014_positive
        messages_negative = variables.tweets_liveJournal2014_negative
        messages_neutral  = variables.tweets_liveJournal2014_neutral
    elif base == "sarcasm":
        messages = variables.tweets_2014_sarcasm
        messages_score = variables.tweets_2014_sarcasm_score
        messages_positive = variables.tweets_2014_sarcasm_positive
        messages_negative = variables.tweets_2014_sarcasm_negative
        messages_neutral  = variables.tweets_2014_sarcasm_neutral
    elif base == "sms":
        messages = variables.sms_2013
        messages_score = variables.sms_2013_score
        messages_positive = variables.sms_2013_positive
        messages_negative = variables.sms_2013_negative
        messages_neutral  = variables.sms_2013_neutral        
    elif base == "all":
        messages = variables.tweets_2013 + variables.tweets_2014 + variables.tweets_liveJournal2014 + variables.sms_2013 + variables.tweets_2014_sarcasm
        messages_score = variables.tweets_2013_score + variables.tweets_2014_score + variables.tweets_liveJournal2014_score + variables.sms_2013_score + variables.tweets_2014_sarcasm_score
        messages_positive = variables.tweets_2013_positive + variables.tweets_2014_positive + variables.tweets_liveJournal2014_positive + variables.tweets_2014_sarcasm_positive + variables.sms_2013_positive
        messages_negative = variables.tweets_2013_negative + variables.tweets_2014_negative + variables.tweets_liveJournal2014_negative + variables.tweets_2014_sarcasm_negative + variables.sms_2013_negative
        messages_neutral  = variables.tweets_2013_neutral  + variables.tweets_2014_neutral  + variables.tweets_liveJournal2014_neutral  + variables.tweets_2014_sarcasm_neutral  + variables.sms_2013_neutral

    for index, item in enumerate(messages): 
        message = str(messages[index]).strip().replace("'", "")
        message = message.replace("\\u2018", "").replace("\\u2019", "").replace("\\u002c", "")        
        message = "'" + message + "'"

        model_analysis = model.replace("(x)", "(" + message + ")")
        
        if not len(message) > 0:
            continue

        try:
            result = float(eval(model_analysis))
        except:
            print("\n\n[WARNING] eval(model_analysis) exception for the message: " + message + "\n\n")
            continue

        if messages_score[index] > 0:
            if result > 0:
                true_positive += 1
            else:
                if result == 0:
                    false_neutral += 1
                else:
                    false_negative += 1
                
        elif messages_score[index] < 0:
            if result < 0:
                true_negative += 1
            else:
                false_negative_log += 1
                if result == 0:
                    false_neutral += 1
                else:
                    false_positive += 1

                if false_negative_log <= 20:
                    if false_negative_log == 1:  
                        print("\n##### [FALSE NEGATIVES][" + base + "] #####\n")
                    print("[Negative phrase]: " + message)
                    print("[Polarity calculated]: " + str(result))

        elif messages_score[index] == 0:
            if result == 0:
                true_neutral += 1
            else:
                if result < 0:
                    false_negative += 1
                else:
                    false_positive += 1


    if true_positive + false_positive + true_negative + false_negative > 0:
        accuracy = (true_positive + true_negative + true_neutral) / (true_positive + false_positive + true_negative + false_negative + true_neutral + false_neutral)

    # Begin PRECISION
    if true_positive + false_positive > 0:
        precision_positive = true_positive / (true_positive + false_positive)


    if true_negative + false_negative > 0:
        precision_negative = true_negative / (true_negative + false_negative)
    

    if true_neutral + false_neutral > 0:
        precision_neutral = true_neutral / (true_neutral + false_neutral)
    # End PRECISION

    # Begin RECALL
    if messages_positive > 0:
        recall_positive = true_positive / messages_positive


    if messages_negative > 0:
        recall_negative = true_negative / messages_negative

    if messages_neutral > 0:
        recall_neutral = true_neutral / messages_neutral
    # End RECALL

    # Begin F1
    if precision_positive + recall_positive > 0:
        f1_positive = 2 * (precision_positive * recall_positive) / (precision_positive + recall_positive)

    if precision_negative + recall_negative > 0:
        f1_negative = 2 * (precision_negative * recall_negative) / (precision_negative + recall_negative)

    if precision_neutral + recall_neutral > 0:
        f1_neutral = 2 * (precision_neutral * recall_neutral) / (precision_neutral + recall_neutral)                  
    # End F1        

    # Precision, Recall and f1 means
    precision_avg = (precision_positive + precision_negative + precision_neutral) / 3
    
    recall_avg = (recall_positive + recall_negative + recall_neutral) / 3

    f1_avg = (f1_positive + f1_negative + f1_neutral) / 3

    f1_positive_negative_avg = (f1_positive + f1_negative) / 2         

    print("\n")
    print("[" + base + " messages]")
    print("[messages evaluated]: " + str(len(messages)))
    print("[correct evaluations]: " + str(true_positive + true_negative + true_neutral) + " (" + str(true_positive) + " positives, " + str(true_negative) + " negatives and " + str(true_neutral) + " neutrals)")
    print("[model]: " + str(model))
    print("[accuracy]: " + str(round(accuracy, 4)))
    print("[precision_positive]: " + str(round(precision_positive, 4)))
    print("[precision_negative]: " + str(round(precision_negative, 4)))
    print("[precision_neutral]: " + str(round(precision_neutral, 4)))    
    print("[precision_avg]: " + str(round(precision_avg, 4)))
    print("[recall_positive]: " + str(round(recall_positive, 4)))
    print("[recall_negative]: " + str(round(recall_negative, 4)))
    print("[recall_neutral]: " + str(round(recall_neutral, 4)))
    print("[recall avg]: " + str(round(recall_avg, 4)))
    print("[f1_positive]: " + str(round(f1_positive, 4)))
    print("[f1_negative]: " + str(round(f1_negative, 4)))
    print("[f1_neutral]: " + str(round(f1_neutral, 4)))
    print("[f1 avg]: " + str(round(f1_avg, 4)))
    print("[f1 avg SemEval (positive and negative)]: " + str(round(f1_positive_negative_avg, 4)))    
    print("[true_positive]: " + str(true_positive))
    print("[false_positive]: " + str(false_positive))
    print("[true_negative]: " + str(true_negative))
    print("[false_negative]: " + str(false_negative))
    print("[true_neutral]: " + str(true_neutral))
    print("[false_neutral]: " + str(false_neutral))
    print("\n")

    if variables.save_file_results:
        with open(variables.FILE_RESULTS, 'a') as f:
            if base == "tweets2013":
                f.write("[Model]\t" + model + "\n")
            f.write(base + "\t" + str(round(f1_positive_negative_avg, 4)) + "\n")
            if base == "all":
                f.write("\n")


def resultsAnalysis():
    models = 0

    t2k13_list   = []
    t2k14_list   = []
    sms_list     = []
    liveJ_list   = []
    sarcasm_list = []
    allB_list    = []

    with open(variables.FILE_RESULTS, 'r') as f:
        for line in f:
            if line.startswith("["):
                models += 1
            elif len(line) > 1:
                base = line.split("\t")[0]
                value = float(line.split("\t")[1])
                if base == "tweets2013":
                    t2k13_list.append(value)
                elif base == "tweets2014":
                    t2k14_list.append(value)
                elif base == "sms":
                    sms_list.append(value)                
                elif base == "livejournal":
                    liveJ_list.append(value)                     
                elif base == "sarcasm":
                    sarcasm_list.append(value)                     
                elif base == "all":
                    allB_list.append(value)

    with open(variables.FILE_RESULTS, 'a') as f:
        f.write("\n\n##Statistics##\n\n")
        f.write(str(models) + " models evaluated\n\n")
        f.write("AVGs")
        f.write("\nAVG Tweets2013 F1 SemEval\t" + str(round((sum(t2k13_list) / models), 4)))
        f.write("\nAVG Tweets2014 F1 SemEval\t" + str(round((sum(t2k14_list) / models), 4)))
        f.write("\nAVG SMS F1 SemEval\t" + str(round((sum(sms_list) / models), 4)))
        f.write("\nAVG LiveJournal F1 SemEval\t" + str(round((sum(liveJ_list) / models), 4)))
        f.write("\nAVG Sarcasm F1 SemEval\t" + str(round((sum(sarcasm_list) / models), 4)))
        f.write("\nAVG All F1 SemEval\t" + str(round((sum(allB_list) / models), 4)))
        f.write("\n\nBest Values")
        f.write("\nBest Tweets2013 F1 value\t" + str(round(max(t2k13_list), 4)))
        f.write("\nBest Tweets2014 F1 value\t" + str(round(max(t2k14_list), 4)))
        f.write("\nBest SMS F1 value\t" + str(round(max(sms_list), 4)))
        f.write("\nBest LiveJournal F1 value\t" + str(round(max(liveJ_list), 4)))
        f.write("\nBest Sarcasm F1 value\t" + str(round(max(sarcasm_list), 4)))
        f.write("\nBest All F1 value\t" + str(round(max(allB_list), 4)))
        f.write("\n\nValues by database")
        f.write("\nTweets2013 " + str(t2k13_list))
        f.write("\nTweets2014 " + str(t2k14_list))
        f.write("\nSMS " + str(sms_list))
        f.write("\nLiveJournal " + str(liveJ_list))
        f.write("\nSarcasm " + str(sarcasm_list))
        f.write("\nAll " + str(allB_list))
        f.write("\n\nStandard deviation")
        f.write("\nStandard Deviation Tweets2013\t" + str(calcStdDeviation(calcVariance(t2k13_list, models))))
        f.write("\nStandard Deviation Tweets2014\t" + str(calcStdDeviation(calcVariance(t2k14_list, models))))
        f.write("\nStandard Deviation SMS\t" + str(calcStdDeviation(calcVariance(sms_list, models))))
        f.write("\nStandard Deviation Live Journal\t" + str(calcStdDeviation(calcVariance(liveJ_list, models))))
        f.write("\nStandard Deviation Sarcasm\t" + str(calcStdDeviation(calcVariance(sarcasm_list, models))))
        f.write("\nStandard Deviation All\t" + str(calcStdDeviation(calcVariance(allB_list, models))))

    databases = ["tweets2013","tweets2014","sms","livejournal","sarcasm"]

    N = models
    ind = np.arange(N)  # the x locations for the groups
    width = 0.10      # the width of the bars

    fig, ax = plt.subplots()
    
    rects1 = ax.bar(ind, t2k13_list, width, color='r')
    rects2 = ax.bar(ind + width, t2k14_list, width, color='y')
    rects3 = ax.bar(ind + width * 2, sms_list, width, color='b')
    rects4 = ax.bar(ind + width * 3, liveJ_list, width, color='g')
    rects5 = ax.bar(ind + width * 4, sarcasm_list, width, color='k')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('F1')
    ax.set_xlabel('Models')
    ax.set_title('F1 by database')
    #ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(np.arange(models))
    #ax.set_xticklabels(('Tweets2013', 'Tweets2014', 'SMS', 'SARCASM', 'LiveJournal'))

    ax.legend((rects1[0], rects2[0], rects3[0], rects4[0], rects5[0]), ('Twitter2013', 'Twitter2014', 'SMS', 'LiveJournal', 'Sarcasm'))

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '%.2f' % float(height),
                    ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    autolabel(rects5)

    #plt.show()


def calcVariance(base, total_models):
    diffs = []
    
    avg = sum(base) / total_models
    
    for base_value in base:
        diffs.append(math.pow(base_value - avg, 2))

    variance = sum(diffs) / total_models
    return variance


def calcStdDeviation(variance):
    return math.sqrt(variance)