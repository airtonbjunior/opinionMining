# Airton Bordin Junior
# airtonbjunior@gmail.com
# Federal University of Goias (UFG)
# Computer Science Master's Degree

import math
import re 
import string
import time
import codecs
import dateparser

from stemming.porter2 import stem
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import numpy as np

from collections import Counter
from datetime import datetime
from validate_email import validate_email

import variables

# Load the dictionaries
def getDictionary(module):
    start = time.time()
    startDic = 0
    print("\n[loading dictionaries]")

    if module == "train":
        loadTrainWords()
    elif module == "test":
        loadTestWords()
    else:
        print("[error! Unknow module on getDictionary function]")
    
    #I'm always loading the hashtags, emoticons, negating and booster words
    startDic = time.time()
    print("  [loading hashtags, emoticons, negating and booster words]")
    with open(variables.DICTIONARY_POSITIVE_HASHTAGS, 'r') as inF:
        for line in inF:
            variables.dic_positive_hashtags.append(line.lower().strip())

    with open(variables.DICTIONARY_NEGATIVE_HASHTAGS, 'r') as inF:
        for line in inF:
            variables.dic_negative_hashtags.append(line.lower().strip())            

    with open(variables.DICTIONARY_POSITIVE_EMOTICONS, 'r') as inF:
        for line in inF:
            variables.dic_positive_emoticons.append(line.strip()) 

    with open(variables.DICTIONARY_NEGATIVE_EMOTICONS, 'r') as inF:
        for line in inF:
            variables.dic_negative_emoticons.append(line.strip())             

    with open(variables.DICTIONARY_NEGATING_WORDS, 'r') as inF:
        for line in inF:
            variables.dic_negation_words.append(line.strip()) 

    with open(variables.DICTIONARY_BOOSTER_WORDS) as inF:
        for line in inF:
            variables.dic_booster_words.append(line.strip())
    
    if variables.log_loads:
        print("    [" + str(len(variables.dic_positive_hashtags) + len(variables.dic_negative_hashtags) + len(variables.dic_positive_emoticons) + len(variables.dic_negative_emoticons) + len(variables.dic_negation_words) + len(variables.dic_booster_words)) + " words loaded]")
        print("      [hashtag, emoticon, negating and booster dictionary loaded][" + str(format(time.time() - startDic, '.3g')) + " seconds]\n")
    
    #BING LIU POSITIVE/NEGATIVE WORDS
    if(variables.use_dic_liu):
        startDic = time.time()
        print("  [loading liu]")
        with open(variables.DICTIONARY_POSITIVE_WORDS, 'r') as inF:
            variables.dic_liu_loaded = True
            variables.dic_loaded_total += 1
            for line in inF:
                line = line.lower().strip()
                if module == "train" and line in variables.all_train_words:
                    variables.dic_positive_words.append(line)
                elif module == "test" and line in variables.all_test_words:
                    variables.dic_positive_words.append(line)

        with codecs.open(variables.DICTIONARY_NEGATIVE_WORDS, "r", "latin-1") as inF:
            for line in inF:
                line = line.lower().strip()
                if module == "train" and line in variables.all_train_words:
                    variables.dic_negative_words.append(line)
                elif module == "test" and line in variables.all_test_words:
                    variables.dic_negative_words.append(line)
        
        if variables.log_loads:
            print("    [" + str(len(variables.dic_positive_words) + len(variables.dic_negative_words)) + " words loaded]")
            print("    [" + str(len(variables.dic_positive_words)) + " positive and " + str(len(variables.dic_negative_words)) + " negative]")
            print("      [liu dictionary loaded][" + str(format(time.time() - startDic, '.3g')) + " seconds]\n")

    #SENTIWORDNET
    if(variables.use_dic_sentiwordnet):
        startDic = time.time()
        print("  [loading sentiwordnet]")
        with open(variables.DICTIONARY_SENTIWORDNET, 'r') as inF:
            variables.dic_sentiwordnet_loaded = True
            variables.dic_loaded_total += 1
            for line in inF:
                splited = line.split("\t")
                if float(splited[2]) > float(splited[3]): #positive greater than negative
                    words = splited[4].lower().strip().split() 
                    for word in words:
                        if not "_" in word:
                            w = word[:word.find("#")]
                            if(len(w) > 2):
                                if (module == "train" and w in variables.all_train_words) or (module == "test" and w in variables.all_test_words):
                                    #variables.dic_positive_sentiwordnet[w] = 1
                                    variables.dic_positive_sentiwordnet[w] = float(splited[2])
                elif float(splited[2]) < float(splited[3]):
                    words = splited[4].lower().strip().split()
                    for word in words:
                        if not "_" in word:
                            w = word[:word.find("#")]
                            if(len(w) > 2):
                                if (module == "train" and w in variables.all_train_words) or (module == "test" and w in variables.all_test_words):
                                    #variables.dic_negative_sentiwordnet[w] = -1
                                    variables.dic_negative_sentiwordnet[w] = float(splited[3]) * -1
                            
        if variables.log_loads:
            print("    [" + str(len(variables.dic_positive_sentiwordnet) + len(variables.dic_negative_sentiwordnet)) + " words loaded]")
            print("    [" + str(len(variables.dic_positive_sentiwordnet)) + " positive and " + str(len(variables.dic_negative_sentiwordnet)) + " negative]")
            print("      [sentiwordnet dictionary loaded][" + str(format(time.time() - startDic, '.3g')) + " seconds]\n")

    #EFFECT
    if(variables.use_dic_effect):
        startDic = time.time()
        print("  [loading effect]")
        with open(variables.DICTIONARY_EFFECT, 'r') as infF:
            variables.dic_effect_loaded = True
            variables.dic_loaded_total += 1
            for line in infF:
                splited = line.split() 
                if (splited[1] == "+Effect"):
                    for word in splited[2].split(","):
                        word = word.lower().strip()
                        if (module == "train" and word in variables.all_train_words) or (module == "test" and word in variables.all_test_words):
                            variables.dic_positive_effect[word] = 1

                elif (splited[1].lower() == "-effect"):
                    for word in splited[2].split(","):
                        word = word.lower().strip()
                        if (module == "train" and word in variables.all_train_words) or (module == "test" and word in variables.all_test_words):
                            variables.dic_negative_effect[word] = -1

        
        if variables.log_loads:
            print("    [" + str(len(variables.dic_positive_effect) + len(variables.dic_negative_effect)) + " words loaded]")
            print("    [" + str(len(variables.dic_positive_effect)) + " positive and " + str(len(variables.dic_negative_effect)) + " negative]")
            print("      [effect dictionary loaded][" + str(format(time.time() - startDic, '.3g')) + " seconds]\n")

    #SEMEVAL2015 
    if(variables.use_dic_semeval2015):
        startDic = time.time()
        print("  [loading semeval2015]")
        with open(variables.DICTIONARY_SEMEVAL2015, 'r') as inF:
            variables.dic_semeval2015_loaded = True
            variables.dic_loaded_total += 1
            for line in inF:
                #removing composite words for while 
                splited = line.split("\t")
                if float(splited[0]) > 0 and not ' ' in splited[1].strip():
                    if "#" in splited[1].strip():
                        variables.dic_positive_hashtags.append(splited[1].strip()[1:])
                    else:
                        if (module == "train" and splited[1].strip() in variables.all_train_words) or (module == "test" and splited[1].strip() in variables.all_test_words):
                            variables.dic_positive_semeval2015[splited[1].strip()] = float(splited[0])

                elif float(splited[0]) < 0 and not ' ' in splited[1].strip():
                    if "#" in splited[1].strip():
                        variables.dic_negative_hashtags.append(splited[1].strip()[1:])
                    else:
                        if (module == "train" and splited[1].strip() in variables.all_train_words) or (module == "test" and splited[1].strip() in variables.all_test_words):
                            variables.dic_negative_semeval2015[splited[1].strip()] = float(splited[0])

        if variables.log_loads:
            print("    [" + str(len(variables.dic_positive_semeval2015) + len(variables.dic_negative_semeval2015)) + " words loaded]")
            print("    [" + str(len(variables.dic_positive_semeval2015)) + " positive and " + str(len(variables.dic_negative_semeval2015)) + " negative]")
            print("      [semeval2015 dictionary loaded][" + str(format(time.time() - startDic, '.3g')) + " seconds]\n")

    #AFFIN
    if(variables.use_dic_affin):
        startDic = time.time()
        print("  [loading affin]")
        with open(variables.DICTIONARY_AFFIN, 'r') as inF:
            variables.dic_affin_loaded = True
            variables.dic_loaded_total += 1
            for line in inF:
                splited = line.split("\t")
                if float(splited[1].strip()) > 0:
                    if (module == "train" and splited[0].strip() in variables.all_train_words) or (module == "test" and splited[0].strip() in variables.all_test_words):
                        variables.dic_positive_affin[splited[0].strip()] = float(splited[1].strip())
                else:
                    if (module == "train" and splited[0].strip() in variables.all_train_words) or (module == "test" and splited[0].strip() in variables.all_test_words):
                        variables.dic_negative_affin[splited[0].strip()] = float(splited[1].strip())

        if variables.log_loads:
            print("    [" + str(len(variables.dic_positive_affin) + len(variables.dic_negative_affin)) + " words loaded]")
            print("    [" + str(len(variables.dic_positive_affin)) + " positive and " + str(len(variables.dic_negative_affin)) + " negative]")
            print("      [affin dictionary loaded][" + str(format(time.time() - startDic, '.3g')) + " seconds]\n")

    #SLANG
    if(variables.use_dic_slang):
        startDic = time.time()
        print("  [loading slang]")
        with codecs.open(variables.DICTIONARY_SLANG, "r", "latin-1") as inF:
            variables.dic_slang_loaded = True
            variables.dic_loaded_total += 1
            for line in inF:
                splited = line.split("\t")
                if float(splited[1].strip()) > 0:
                    if (module == "train" and splited[0].strip() in variables.all_train_words) or (module == "test" and splited[0].strip() in variables.all_test_words):
                        variables.dic_positive_slang[splited[0].strip()] = float(splited[1].strip())

                elif float(line.split("\t")[1].strip()) < 0:
                    if (module == "train" and splited[0].strip() in variables.all_train_words) or (module == "test" and splited[0].strip() in variables.all_test_words):
                        variables.dic_negative_slang[splited[0].strip()] = float(splited[1].strip())
        
        if variables.log_loads:
            print("    [" + str(len(variables.dic_positive_slang) + len(variables.dic_negative_slang)) + " words loaded]")
            print("    [" + str(len(variables.dic_positive_slang)) + " positive and " + str(len(variables.dic_negative_slang)) + " negative]")
            print("      [slang dictionary loaded][" + str(format(time.time() - startDic, '.3g')) + " seconds]\n")

    #VADER
    if(variables.use_dic_vader):
        startDic = time.time()
        print("  [loading vader]")
        with open(variables.DICTIONARY_VADER, 'r') as inF:
            variables.dic_vader_loaded = True
            variables.dic_loaded_total += 1
            for line in inF:
                splited = line.split("\t")
                if float(splited[1].strip()) > 0:
                    if (module == "train" and splited[0].strip() in variables.all_train_words) or (module == "test" and splited[0].strip() in variables.all_test_words):
                        variables.dic_positive_vader[splited[0].strip()] = float(splited[1].strip())
                else:
                    if (module == "train" and splited[0].strip() in variables.all_train_words) or (module == "test" and splited[0].strip() in variables.all_test_words):
                        variables.dic_negative_vader[splited[0].strip()] = float(splited[1].strip())

        if variables.log_loads:
            print("    [" + str(len(variables.dic_positive_vader) + len(variables.dic_negative_vader)) + " words loaded]")
            print("    [" + str(len(variables.dic_positive_vader)) + " positive and " + str(len(variables.dic_negative_vader)) + " negative]")
            print("      [vader dictionary loaded][" + str(format(time.time() - startDic, '.3g')) + " seconds]\n")
    
    # Performance improvement test
    variables.dic_positive_words     = set(variables.dic_positive_words)
    variables.dic_negative_words     = set(variables.dic_negative_words)
    variables.dic_positive_hashtags  = set(variables.dic_positive_hashtags)
    variables.dic_negative_hashtags  = set(variables.dic_negative_hashtags)
    variables.dic_positive_emoticons = set(variables.dic_positive_emoticons)
    variables.dic_negative_emoticons = set(variables.dic_negative_emoticons)

    end = time.time()
    print("[all dictionaries (" + str(variables.dic_loaded_total) + ") loaded][" + str(format(end - start, '.3g')) + " seconds]\n")


def loadTrainWords():
    start = time.time()
    print("\n  [loading train words]")
    with open(variables.TRAIN_WORDS, 'r') as file:
        for line in file:
            variables.all_train_words.append(line.replace('\n', '').replace('\r', ''))

    print("    [train words loaded (" + str(len(variables.all_train_words)) + " words)][" + str(format(time.time() - start, '.3g')) + " seconds]\n")


def loadTestWords():
    start = time.time()
    print("\n  [loading test words]")    
    start = time.time()
    with open(variables.TEST_WORDS, 'r') as file:
        for line in file:
            variables.all_test_words.append(line.replace('\n', '').replace('\r', ''))

    print("    [test words loaded (" + str(len(variables.all_test_words)) + " words)][" + str(format(time.time() - start, '.3g')) + " seconds]\n")


# Load tweets from id (SEMEVAL 2014 database)
def loadTrainTweets():
    start = time.time()
    print("\n[loading train tweets]")

    #positive_words = []
    #negative_words = []
    #neutral_words  = []
    #train_words    = []

    tweets_loaded = 0

    with open(variables.SEMEVAL_TRAIN_FILE, 'r') as inF:
        for line in inF:
            if tweets_loaded < variables.MAX_ANALYSIS_TWEETS:
                tweet_parsed = line.split("\t")
                try:
                    if(tweet_parsed[2] != "neutral"):
                        if(tweet_parsed[2] == "positive"):
                            if(variables.positive_tweets < variables.MAX_POSITIVES_TWEETS):
                                variables.positive_tweets += 1
                                variables.tweets_semeval.append(tweet_parsed[3])
                                variables.tweets_semeval_score.append(1)
                                #for w in tweet_parsed[3].split():
                                #    if removeAllPonctuation(w).strip().lower() not in positive_words:
                                #        positive_words.append(removeAllPonctuation(w).strip().lower())
                                tweets_loaded += 1
                        else:
                            if(variables.negative_tweets < variables.MAX_NEGATIVES_TWEETS):
                                variables.negative_tweets += 1
                                variables.tweets_semeval.append(tweet_parsed[3])
                                variables.tweets_semeval_score.append(-1)
                                #for w in tweet_parsed[3].split():
                                #    if removeAllPonctuation(w).strip().lower() not in negative_words:
                                #        negative_words.append(removeAllPonctuation(w).strip().lower())
                                tweets_loaded += 1
                    else:
                        if(variables.neutral_tweets < variables.MAX_NEUTRAL_TWEETS):
                            variables.tweets_semeval.append(tweet_parsed[3])
                            variables.tweets_semeval_score.append(0)
                            variables.neutral_tweets += 1
                            #for w in tweet_parsed[3].split():
                            #    if removeAllPonctuation(w).strip().lower() not in neutral_words:
                            #        neutral_words.append(removeAllPonctuation(w).strip().lower())                            
                            tweets_loaded += 1
                except:
                    print("exception")
                    continue

    #train_words = positive_words + negative_words + neutral_words
    
    # save words on file
    #with open("words_train.txt", 'w') as f:
    #    for word in train_words:
    #        if len(word) > 0:
    #            f.write(str(word) + "\n")
    #with open("negative_words_train.txt", 'w') as f:
    #    for word in negative_words:
    #        f.write(str(word).lower().strip() + ",")
    #with open("neutral_words_train.txt", 'w') as f:
    #    for word in neutral_words:
    #        f.write(str(word).lower().strip() + ",")
    # save words on file        
    
    end = time.time()
    print("  [train tweets loaded (" + str(tweets_loaded) + " tweets)][" + str(format(end - start, '.3g')) + " seconds]\n")


# Load the test tweets from Semeval 2014 task 9
def loadTestTweets():
    start = time.time()
    print("\n[loading test tweets]")

    tweets_loaded = 0

    test_words = []

    with open(variables.SEMEVAL_TEST_FILE, 'r') as inF: #ORIGINAL FILE. ABOVE IS ONLY TEST
    #with open("datasets/STS_Gold_All.txt", 'r') as inF: #only for test STS GOLD   
        for line in inF:
            if tweets_loaded < variables.MAX_ANALYSIS_TWEETS:
                tweet_parsed = line.split("\t")
                try:
                    # TEST USING SVM - KEEP THE ORDER
                    variables.all_messages_in_file_order.append(tweet_parsed[2])
                    if tweet_parsed[0] == "positive":
                        variables.all_polarities_in_file_order.append(1)
                    elif tweet_parsed[0] == "negative":
                        variables.all_polarities_in_file_order.append(-1)
                    elif tweet_parsed[0] == "neutral":
                        variables.all_polarities_in_file_order.append(0)

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
                
                    #SVM Values
                    variables.svm_values_tweets.append(tweet_parsed[3])
                    svm_values = []
                    svm_values = tweet_parsed[3].strip()[1:-1].split()

                    if(float(svm_values[0]) >= -0.4):
                        variables.svm_is_neutral.append(False)
                        variables.svm_normalized_values.append(-1)

                    elif(float(svm_values[2]) > float(svm_values[1])):
                        variables.svm_is_neutral.append(False)
                        variables.svm_normalized_values.append(1)
                    else:
                        variables.svm_is_neutral.append(True)
                        variables.svm_normalized_values.append(0)
                    #SVM Values
                
                except Exception as e:
                    print("exception 1: " + e)
                    continue

    end = time.time()
    print("  [test tweets loaded (" + str(tweets_loaded) + " tweets)][" + str(format(end - start, '.3g')) + " seconds]\n")


# Load STS train tweets
def loadTrainTweets_STS():
    start = time.time()
    print("\n[loading STS train tweets]")

    tweets_loaded = 0
    
    with codecs.open("/home/airton/Desktop/training.1600000.processed.noemoticon.EDITED.csv", "r", "latin-1") as inF:
    #with open(variables.SEMEVAL_TRAIN_FILE, 'r') as inF:
        for line in inF:
            if tweets_loaded < variables.MAX_ANALYSIS_TWEETS:
                tweet_parsed = line.split("\t")
                try:
                    if(tweet_parsed[0] == "positive"):
                        if(variables.positive_tweets < variables.MAX_POSITIVES_TWEETS):
                            variables.positive_tweets += 1
                            variables.tweets_sts.append(tweet_parsed[2])
                            variables.tweets_sts_score.append(1)
                            #variables.tweets_semeval.append(tweet_parsed[3])
                            #variables.tweets_semeval_score.append(1)                            
                            tweets_loaded += 1
                    else:
                        if(variables.negative_tweets < variables.MAX_NEGATIVES_TWEETS):
                            variables.negative_tweets += 1
                            variables.tweets_sts.append(tweet_parsed[2])
                            variables.tweets_sts_score.append(-1)
                            #variables.tweets_semeval.append(tweet_parsed[3])
                            #variables.tweets_semeval_score.append(-1)
                            tweets_loaded += 1
                
                except:
                    print("exception")
                    continue

    end = time.time()
    print("  [train STS tweets loaded (" + str(tweets_loaded) + " tweets)][" + str(format(end - start, '.3g')) + " seconds]\n")


# TO-DO
# STS dataset
def loadTestTweets_STS():
    start = time.time()
    print("\n[loading test tweets (STS)]")    
    
    tweets_loaded = 0

    with open(variables.STS_TEST_FILE, 'r') as f:
        for line in f:
            if tweets_loaded < variables.MAX_ANALYSIS_TWEETS:
                tweet_parsed = line.split("\t")
                try:
                    if tweet_parsed[0] == "positive":
                        variables.tweets_sts_score.append(1)
                        variables.tweets_sts_positive += 1

                    elif tweet_parsed[0] == "negative":
                        variables.tweets_sts_score.append(-1)
                        variables.tweets_sts_negative += 1

                except Exception as e:
                    print("exception 3: " + e)
                    continue

    end = time.time()
    print("  [test tweets (STS) loaded (" + str(tweets_loaded) + " tweets)][" + str(format(end - start, '.3g')) + " seconds]\n")    
    return 0


# Mukherjee dataset
def loadTestTweets_smuk():
    start = time.time()
    print("\n[loading test tweets - Mukherjee]")

    tweets_loaded = 0
    
    with open('datasets/test/Dataset2.txt', 'r') as inF:
        for line in inF:
            if tweets_loaded < variables.MAX_ANALYSIS_TWEETS:
                tweet_parsed = line.split(" $ ")

                #print(tweet_parsed[0] + " - " + tweet_parsed[1])
                
                # I'm putting all the tweets on tweets2013 only for test
                variables.tweets_2013.append(tweet_parsed[2])
                if tweet_parsed[1] == "pos":
                    variables.tweets_2013_score.append(1)
                    variables.tweets_2013_positive += 1

                elif tweet_parsed[1] == "neg":
                    variables.tweets_2013_score.append(-1)
                    variables.tweets_2013_negative += 1

                #if tweet_parsed[0] == "pos":
                #    variables.tweets_mukh.append(1)
                #    variables.tweets_mukh_positive += 1
                #else:
                #    variables.tweets_mukh.append(-1)
                #    variables.tweets_mukh_negative += 1


#Aux functions
def add(left, right):
    return left + right

def sub(left, right):
    return left - right

def mul(left, right):
    return left * right

def addI(left, right):
    return left + right

def subI(left, right):
    return left - right

def mulI(left, right):
    return left * right

def exp(par):
    return math.exp(par)

def cos(par):
    return math.cos(par)

def sin(par):
    return math.sin(par)

def passInt(num):
    return num

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

def is_float_try(str):
    try:
        float(str)
        return True
    except ValueError:
        return False

#Phrase manipulation functions
def negativeWordsQuantity(phrase):
    negative_words = 0
    words = phrase.split()
    
    for word in words:
        if word in variables.dic_negative_words:
            negative_words += 1

    return negative_words


def positiveWordsQuantity(phrase):
    positive_words = 0
    words = phrase.split()
    
    for word in words:
        if word in variables.dic_positive_words:
            positive_words += 1
    
    return positive_words    


# Sequence:
# [w1]liu [w2]sentiwordnet [w3]affin [w4]vader [w5]slang [w6]effect [w7]semeval2015 
def dictionaryWeights(w1, w2, w3, w4, w5, w6, w7):
    variables.liu_weight          = w1
    variables.sentiwordnet_weight = w2
    variables.affin_weight        = w3
    variables.vader_weight        = w4
    variables.slang_weight        = w5
    variables.effect_weight       = w6
    variables.semeval2015_weight  = w7


def neutralRange(inferior, superior):
    #print("##NEUTRAL RANGE##")
    #print("Inferior -> " + str(inferior))
    #print("Superior -> " + str(superior))

    if float(inferior) > float(superior):
        #print("@@FAIL, inferior greater than superior@@")
        variables.neutral_inferior_range = 0
        variables.neutral_superior_range = 0
    else:
        variables.neutral_inferior_range = inferior
        variables.neutral_superior_range = superior
    
    return 0 # try not be used in other branches of the tree


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
        if(variables.use_dic_liu and variables.dic_liu_loaded):
            if word in variables.dic_positive_words:
                if invert:
                    total_sum -= 1
                elif booster:
                    total_sum += 2
                elif boosterAndInverter:
                    total_sum -= 2
                else: 
                    total_sum += 1

                #print("find word " + word + " on liu positive")
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

                #print("find word " + word + " on liu negative")
                dic_quantity += 1

        # SENTIWORDNET
        if(variables.use_dic_sentiwordnet and variables.dic_sentiwordnet_loaded):
            if word in variables.dic_positive_words_sentiwordnet:
                if invert:
                    total_sum -= variables.dic_positive_value_sentiwordnet[variables.dic_positive_words_sentiwordnet.index(word)]
                elif booster:
                    total_sum += 2 * variables.dic_positive_value_sentiwordnet[variables.dic_positive_words_sentiwordnet.index(word)]
                else:
                    total_sum += variables.dic_positive_value_sentiwordnet[variables.dic_positive_words_sentiwordnet.index(word)]
                
                #print("find word " + word + " on sentiwordnet positive")
                dic_quantity += 1
            elif word in variables.dic_negative_words_sentiwordnet:
                if invert:
                    total_sum -= variables.dic_negative_value_sentiwordnet[variables.dic_negative_words_sentiwordnet.index(word)]
                elif booster:
                    total_sum += 2 * variables.dic_negative_value_sentiwordnet[variables.dic_negative_words_sentiwordnet.index(word)]
                else:
                    total_sum += variables.dic_negative_value_sentiwordnet[variables.dic_negative_words_sentiwordnet.index(word)]
                
                #print("find word " + word + " on sentiwordnet negative")
                dic_quantity += 1

        # AFFIN
        if(variables.use_dic_affin and variables.dic_affin_loaded):
            if word in variables.dic_positive_words_affin:
                if invert:
                    total_sum -= variables.dic_positive_value_affin[variables.dic_positive_words_affin.index(word)]
                elif booster:
                    total_sum += 2 * variables.dic_positive_value_affin[variables.dic_positive_words_affin.index(word)]
                else:
                    total_sum += variables.dic_positive_value_affin[variables.dic_positive_words_affin.index(word)]
                
                #print("find word " + word + " on affin positive")
                dic_quantity += 1
            elif word in variables.dic_negative_words_affin:
                if invert:
                    total_sum -= variables.dic_negative_value_affin[variables.dic_negative_words_affin.index(word)]
                elif booster:
                    total_sum += 2 * variables.dic_negative_value_affin[variables.dic_negative_words_affin.index(word)]
                else:
                    total_sum += variables.dic_negative_value_affin[variables.dic_negative_words_affin.index(word)]
                
                #print("find word " + word + " on affin negative")
                dic_quantity += 1                

        # VADER
        if(variables.use_dic_vader and variables.dic_vader_loaded):
            if word in variables.dic_positive_words_vader:
                if invert:
                    total_sum -= variables.dic_positive_value_vader[variables.dic_positive_words_vader.index(word)]
                elif booster:
                    total_sum += 2 * variables.dic_positive_value_vader[variables.dic_positive_words_vader.index(word)]
                else:
                    total_sum += variables.dic_positive_value_vader[variables.dic_positive_words_vader.index(word)]
                
                #print("find word " + word + " on vader positive")
                dic_quantity += 1
            elif word in variables.dic_negative_words_vader:
                if invert:
                    total_sum -= variables.dic_negative_value_vader[variables.dic_negative_words_vader.index(word)]
                elif booster:
                    total_sum += 2 * variables.dic_negative_value_vader[variables.dic_negative_words_vader.index(word)]
                else:
                    total_sum += variables.dic_negative_value_vader[variables.dic_negative_words_vader.index(word)]
                
                #print("find word " + word + " on vader negative")
                dic_quantity += 1

        # SLANG
        if(variables.use_dic_slang and variables.dic_slang_loaded):
            if word in variables.dic_positive_words_slang:
                if invert:
                    total_sum -= variables.dic_positive_value_slang[variables.dic_positive_words_slang.index(word)]
                elif booster:
                    total_sum += 2 * variables.dic_positive_value_slang[variables.dic_positive_words_slang.index(word)]
                else:
                    total_sum += variables.dic_positive_value_slang[variables.dic_positive_words_slang.index(word)]
                
                #print("find word " + word + " on slang positive")
                dic_quantity += 1
            elif word in variables.dic_negative_words_slang:
                if invert:
                    total_sum -= variables.dic_negative_value_slang[variables.dic_negative_words_slang.index(word)]
                elif booster:
                    total_sum += 2 * variables.dic_negative_value_slang[variables.dic_negative_words_slang.index(word)]
                else:
                    total_sum += variables.dic_negative_value_slang[variables.dic_negative_words_slang.index(word)]
                
                #print("find word " + word + " on slang negative")                
                dic_quantity += 1    
        
        # EFFECT
        if(variables.use_dic_effect and variables.dic_effect_loaded):
            if word in variables.dic_positive_words_effect:
                if invert:
                    total_sum -= variables.dic_positive_value_effect[variables.dic_positive_words_effect.index(word)]
                elif booster:
                    total_sum += 2 * variables.dic_positive_value_effect[variables.dic_positive_words_effect.index(word)]
                else:
                    total_sum += variables.dic_positive_value_effect[variables.dic_positive_words_effect.index(word)]
                
                #print("find word " + word + " on effect positive")
                dic_quantity += 1
            elif word in variables.dic_negative_words_effect:
                if invert:
                    total_sum -= variables.dic_negative_value_effect[variables.dic_negative_words_effect.index(word)]
                elif booster:
                    total_sum += 2 * variables.dic_negative_value_effect[variables.dic_negative_words_effect.index(word)]
                else:
                    total_sum += variables.dic_negative_value_effect[variables.dic_negative_words_effect.index(word)]
                
                #print("find word " + word + " on effect negative")                
                dic_quantity += 1  

        # SEMEVAL2015
        if(variables.use_dic_semeval2015 and variables.dic_semeval2015_loaded):
            if word in variables.dic_positive_words_semeval2015:
                if invert:
                    total_sum -= variables.dic_positive_value_semeval2015[variables.dic_positive_words_semeval2015.index(word)]
                elif booster:
                    total_sum += 2 * variables.dic_positive_value_semeval2015[variables.dic_positive_words_semeval2015.index(word)]
                else:
                    total_sum += variables.dic_positive_value_semeval2015[variables.dic_positive_words_semeval2015.index(word)]
                
                #print("find word " + word + " on semeval2015 positive")
                dic_quantity += 1
            elif word in variables.dic_negative_words_semeval2015:
                if invert:
                    total_sum -= variables.dic_negative_value_semeval2015[variables.dic_negative_words_semeval2015.index(word)]
                elif booster:
                    total_sum += 2 * variables.dic_negative_value_semeval2015[variables.dic_negative_words_semeval2015.index(word)]
                else:
                    total_sum += variables.dic_negative_value_semeval2015[variables.dic_negative_words_semeval2015.index(word)]
                
                #print("find word " + word + " on semeval2015 negative")                
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
        if index > 0 and words[index-1] == "insidenoteboosterword" and (words[index-2] == "insidenoteinverterword" or words[index-3] == "insidenoteinverterword"):
            #print("boosterAndInverter")
            boosterAndInverter = True
        elif index > 0 and words[index-1] == "insidenoteinverterword":
            #print("inverter")
            invert = True
        elif (index > 0 and words[index-1] == "insidenoteboosterword") or (index < len(words) - 1 and words[index+1] == "insidenoteboosterword" and (words[index-1] != "insidenoteboosterword" or index == 0)):
            #print("booster")
            booster = True


        # LIU pos/neg words
        if(variables.use_dic_liu and variables.dic_liu_loaded):
            if word in variables.dic_positive_words:
                if invert:
                    total_sum -= 1
                elif booster:
                    total_sum += 2
                elif boosterAndInverter:
                    total_sum -= 2
                else: 
                    total_sum += 1

                #print("find word " + word + " on liu positive")
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

                #print("find word " + word + " on liu negative")
                dic_quantity += 1 

        # SENTIWORDNET
        if(variables.use_dic_sentiwordnet and variables.dic_sentiwordnet_loaded):
            if word in variables.dic_positive_sentiwordnet:
                
                #print("word " + word + " on sentiwordnet with the value " + str(variables.dic_positive_sentiwordnet[word]))

                if invert:
                    total_sum -= variables.dic_positive_sentiwordnet[word]
                    #total_sum -= 1 * w2
                elif booster:
                    total_sum += 2 * variables.dic_positive_sentiwordnet[word]
                    #total_sum += 2 * w2
                elif boosterAndInverter:
                    total_sum -= 2 * variables.dic_positive_sentiwordnet[word]
                    #total_sum -= 2 * w2
                else:
                    total_sum += variables.dic_positive_sentiwordnet[word]
                    #total_sum += 1 * w2
                
                dic_quantity += 1 
            elif word in variables.dic_negative_sentiwordnet:
                
                #print("word " + word + " on sentiwordnet with the value " + str(variables.dic_negative_sentiwordnet[word]))

                if invert:
                    total_sum -= variables.dic_negative_sentiwordnet[word]
                    #total_sum += 1 * w2
                elif booster:
                    total_sum += 2 * variables.dic_negative_sentiwordnet[word]
                    #total_sum -= 2 * w2
                elif boosterAndInverter:
                    total_sum -= 2 * variables.dic_negative_sentiwordnet[word]
                    #total_sum += 2 * w2
                else:
                    total_sum += variables.dic_negative_sentiwordnet[word]
                    #total_sum -= 1 * w2
                
                dic_quantity += 1

        # AFFIN
        if(variables.use_dic_affin and variables.dic_affin_loaded):
            if word in variables.dic_positive_affin:
                
                #print("word " + word + " on affin with the value " + str(variables.dic_positive_affin[word]))
                
                if invert:
                    total_sum -= variables.dic_positive_affin[word]
                    #total_sum -= 1 * w3
                elif booster:
                    total_sum += 2 * variables.dic_positive_affin[word]
                    #total_sum += 2 * w3
                elif boosterAndInverter:
                    total_sum -= 2 * variables.dic_positive_affin[word]
                    #total_sum -= 2 * w3                    
                else:
                    total_sum += variables.dic_positive_affin[word]
                    #total_sum += 1 * w3
                
                dic_quantity += 1
            elif word in variables.dic_negative_affin:
                
                #print("word " + word + " on affin with the value " + str(variables.dic_negative_affin[word]))

                if invert:
                    total_sum -= variables.dic_negative_affin[word]
                    #total_sum += 1 * w3
                elif booster:
                    total_sum += 2 * variables.dic_negative_affin[word]
                    #total_sum -= 2 * w3
                elif boosterAndInverter:
                    total_sum -= 2 * variables.dic_negative_affin[word]
                    #total_sum += 2 * w3                
                else:
                    total_sum += variables.dic_negative_affin[word]
                    #total_sum -= 1 * w3
                
                dic_quantity += 1           

        # VADER
        if(variables.use_dic_vader and variables.dic_vader_loaded):
            if word in variables.dic_positive_vader:

                #print("word " + word + " on vader with the value " + str(variables.dic_positive_vader[word]))

                if invert:
                    total_sum -= variables.dic_positive_vader[word]
                    #total_sum -= 1 * w4
                elif booster:
                    total_sum += 2 * variables.dic_positive_vader[word]
                    #total_sum += 2 * w4
                elif boosterAndInverter:
                    total_sum -= 2 * variables.dic_positive_vader[word]
                    #total_sum -= 2 * w4                    
                else:
                    total_sum += variables.dic_positive_vader[word]
                    #total_sum += 1 * w4
                
                dic_quantity += 1
            elif word in variables.dic_negative_vader:
                
                #print("word " + word + " on vader with the value " + str(variables.dic_negative_vader[word]))

                if invert:
                    total_sum -= variables.dic_negative_vader[word]
                    #total_sum += 1 * w4
                elif booster:
                    total_sum += 2 * variables.dic_negative_vader[word]
                    #total_sum -= 2 * w4
                elif boosterAndInverter:
                    total_sum -= 2 * variables.dic_negative_vader[word]
                    #total_sum += 2 * w4                
                else:
                    total_sum += variables.dic_negative_vader[word]
                    #total_sum -= 1 * w4
                
                dic_quantity += 1

        # SLANG
        if(variables.use_dic_slang and variables.dic_slang_loaded):
            if word in variables.dic_positive_slang:
                
                #print("word " + word + " on slang with the value " + str(variables.dic_positive_slang[word]))
                
                if invert:
                    total_sum -= variables.dic_positive_slang[word]
                    #total_sum -= 1 * w5
                elif booster:
                    total_sum += 2 * variables.dic_positive_slang[word]
                    #total_sum += 2 * w5
                elif boosterAndInverter:
                    total_sum -= 2 * variables.dic_positive_slang[word]
                    #total_sum -= 2 * w5                        
                else:
                    total_sum += variables.dic_positive_slang[word]
                    #total_sum += 1 * w5
                
                dic_quantity += 1
            elif word in variables.dic_negative_slang:

                #print("word " + word + " on slang with the value " + str(variables.dic_negative_slang[word]))
                
                if invert:
                    total_sum -= variables.dic_negative_slang[word]
                    #total_sum += 1 * w5
                elif booster:
                    total_sum += 2 * variables.dic_negative_slang[word]
                    #total_sum -= 2 * w5
                elif boosterAndInverter:
                    total_sum -= 2 * variables.dic_negative_slang[word]                     
                else:
                    total_sum += variables.dic_negative_slang[word]
                    #total_sum -= 1 * w5
                             
                dic_quantity += 1
        
        # EFFECT
        if(variables.use_dic_effect and variables.dic_effect_loaded):
            if word in variables.dic_positive_effect:
                
                #print("word " + word + " on effect with the value " + str(variables.dic_positive_effect[word]))

                if invert:
                    total_sum -= variables.dic_positive_effect[word]
                    #total_sum -= 1 * w6
                elif booster:
                    total_sum += 2 * variables.dic_positive_effect[word]
                    #total_sum += 2 * w6
                elif boosterAndInverter:
                    total_sum -= 2 * variables.dic_positive_effect[word]
                    #total_sum -= 2 * w6
                else:
                    total_sum += variables.dic_positive_effect[word]
                    #total_sum += 1 * w6
                
                dic_quantity += 1
            elif word in variables.dic_negative_effect:
                
                #print("word " + word + " on effect with the value " + str(variables.dic_negative_effect[word]))

                if invert:
                    total_sum -= variables.dic_negative_effect[word]
                    #total_sum += 1 * w6
                elif booster:
                    total_sum += 2 * variables.dic_negative_effect[word]
                    #total_sum -= 2 * w6
                elif boosterAndInverter:
                    total_sum -= 2 * variables.dic_negative_effect[word]
                    #total_sum += 2 * w6
                else:
                    total_sum += variables.dic_negative_effect[word]
                    #total_sum -= 1 * w6
                
                dic_quantity += 1 

        # SEMEVAL2015
        if(variables.use_dic_semeval2015 and variables.dic_semeval2015_loaded):
            if word in variables.dic_positive_semeval2015:
                
                #print("word " + word + " on semeval2015 with the value " + str(variables.dic_positive_semeval2015[word]))

                if invert:
                    total_sum -= variables.dic_positive_semeval2015[word]
                    #total_sum -= 1 * w7
                elif booster:
                    total_sum += 2 * variables.dic_positive_semeval2015[word]
                    #total_sum += 2 * w7
                elif boosterAndInverter:
                    #total_sum -= 2 * w7 
                    total_sum -= 2 * variables.dic_positive_semeval2015[word]
                else:
                    #total_sum += variables.dic_positive_semeval2015[word]
                    total_sum += variables.dic_positive_semeval2015[word]
            
                dic_quantity += 1

            elif word in variables.dic_negative_semeval2015:

                #print("word " + word + " on semeval2015 with the value " + str(variables.dic_negative_semeval2015[word]))

                if invert:
                    total_sum -= variables.dic_negative_semeval2015[word]
                    #total_sum += 1 * w7
                elif booster:
                    total_sum += 2* variables.dic_negative_semeval2015[word]
                    #total_sum -= 2 * w7
                elif boosterAndInverter:
                    total_sum -= 2 * variables.dic_negative_semeval2015[word]
                    #total_sum += 2 * w7                      
                else:
                    total_sum += variables.dic_negative_semeval2015[word]
                    #total_sum -= 1 * w7
                        
                dic_quantity += 1
        
        if(dic_quantity > 1):
            #print("i'll divide " + str(total_sum) + " by " + str(dic_quantity))
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


# ONLY TEST ONLY TEST ONLY TEST
def polaritySumAVGUsingWeights(phrase, w1, w2, w3, w4, w5, w6, w7):
    total_sum = 0
    total_sum_return = 0
    dic_quantity = 0
    index = 0
    invert = False
    booster = False
    boosterAndInverter = False

    total_weight = 0

    phrase = phrase.strip()
    words = phrase.split()

    for word in words:
        # Check booster and inverter words
        if index > 0 and words[index-1] == "insidenoteboosterword" and (words[index-2] == "insidenoteinverterword" or words[index-3] == "insidenoteinverterword"):
            #print("boosterAndInverter")
            boosterAndInverter = True
        elif index > 0 and words[index-1] == "insidenoteinverterword":
            #print("inverter")
            invert = True
        elif (index > 0 and words[index-1] == "insidenoteboosterword") or (index < len(words) - 1 and words[index+1] == "insidenoteboosterword" and (words[index-1] != "insidenoteboosterword" or index == 0)):
            #print("booster")
            booster = True


        # LIU pos/neg words
        if(variables.use_dic_liu and variables.dic_liu_loaded and w1 != 0):
            if word in variables.dic_positive_words:
                if invert:
                    total_sum -= 1 * w1
                elif booster:
                    total_sum += 2 * w1
                elif boosterAndInverter:
                    total_sum -= 2 * w1
                else: 
                    total_sum += 1 * w1

                #print("find word " + word + " on liu positive")
                dic_quantity += 1
                total_weight += w1

            elif word in variables.dic_negative_words:
                if invert:
                    total_sum += 1 * w1
                elif booster:
                    total_sum -= 2 * w1
                elif boosterAndInverter:
                    total_sum += 2 * w1
                else:
                    total_sum -= 1 * w1

                #print("find word " + word + " on liu negative")
                dic_quantity += 1
                total_weight += w1

        # SENTIWORDNET
        if(variables.use_dic_sentiwordnet and variables.dic_sentiwordnet_loaded and w2 != 0):
            if word in variables.dic_positive_sentiwordnet:
                
                #print("word " + word + " on sentiwordnet with the value " + str(variables.dic_positive_sentiwordnet[word]))

                if invert:
                    total_sum -= variables.dic_positive_sentiwordnet[word] * w2
                    #total_sum -= 1 * w2
                elif booster:
                    total_sum += 2 * variables.dic_positive_sentiwordnet[word] * w2
                    #total_sum += 2 * w2
                elif boosterAndInverter:
                    total_sum -= 2 * variables.dic_positive_sentiwordnet[word] * w2
                    #total_sum -= 2 * w2
                else:
                    total_sum += variables.dic_positive_sentiwordnet[word] * w2
                    #total_sum += 1 * w2
                
                dic_quantity += 1 
                total_weight += w2
            elif word in variables.dic_negative_sentiwordnet:
                
                #print("word " + word + " on sentiwordnet with the value " + str(variables.dic_negative_sentiwordnet[word]))

                if invert:
                    total_sum -= variables.dic_negative_sentiwordnet[word] * w2
                    #total_sum += 1 * w2
                elif booster:
                    total_sum += 2 * variables.dic_negative_sentiwordnet[word] * w2
                    #total_sum -= 2 * w2
                elif boosterAndInverter:
                    total_sum -= 2 * variables.dic_negative_sentiwordnet[word] * w2
                    #total_sum += 2 * w2
                else:
                    total_sum += variables.dic_negative_sentiwordnet[word] * w2
                    #total_sum -= 1 * w2
                
                dic_quantity += 1
                total_weight += w2

        # AFFIN
        if(variables.use_dic_affin and variables.dic_affin_loaded and w3 != 0):
            if word in variables.dic_positive_affin:
                
                #print("word " + word + " on affin with the value " + str(variables.dic_positive_affin[word]))
                
                if invert:
                    total_sum -= variables.dic_positive_affin[word] * w3
                    #total_sum -= 1 * w3
                elif booster:
                    total_sum += 2 * variables.dic_positive_affin[word] * w3
                    #total_sum += 2 * w3
                elif boosterAndInverter:
                    total_sum -= 2 * variables.dic_positive_affin[word] * w3
                    #total_sum -= 2 * w3                    
                else:
                    total_sum += variables.dic_positive_affin[word] * w3
                    #total_sum += 1 * w3
                
                dic_quantity += 1
                total_weight += w3
            elif word in variables.dic_negative_affin:
                
                #print("word " + word + " on affin with the value " + str(variables.dic_negative_affin[word]))

                if invert:
                    total_sum -= variables.dic_negative_affin[word] * w3
                    #total_sum += 1 * w3
                elif booster:
                    total_sum += 2 * variables.dic_negative_affin[word] * w3
                    #total_sum -= 2 * w3
                elif boosterAndInverter:
                    total_sum -= 2 * variables.dic_negative_affin[word] * w3
                    #total_sum += 2 * w3                
                else:
                    total_sum += variables.dic_negative_affin[word] * w3
                    #total_sum -= 1 * w3
                
                dic_quantity += 1
                total_weight += w3              

        # VADER
        if(variables.use_dic_vader and variables.dic_vader_loaded and w4 != 0):
            if word in variables.dic_positive_vader:

                #print("word " + word + " on vader with the value " + str(variables.dic_positive_vader[word]))

                if invert:
                    total_sum -= variables.dic_positive_vader[word] * w4
                    #total_sum -= 1 * w4
                elif booster:
                    total_sum += 2 * variables.dic_positive_vader[word] * w4
                    #total_sum += 2 * w4
                elif boosterAndInverter:
                    total_sum -= 2 * variables.dic_positive_vader[word] * w4
                    #total_sum -= 2 * w4                    
                else:
                    total_sum += variables.dic_positive_vader[word] * w4
                    #total_sum += 1 * w4
                
                dic_quantity += 1
                total_weight += w4
            elif word in variables.dic_negative_vader:
                
                #print("word " + word + " on vader with the value " + str(variables.dic_negative_vader[word]))

                if invert:
                    total_sum -= variables.dic_negative_vader[word] * w4
                    #total_sum += 1 * w4
                elif booster:
                    total_sum += 2 * variables.dic_negative_vader[word] * w4
                    #total_sum -= 2 * w4
                elif boosterAndInverter:
                    total_sum -= 2 * variables.dic_negative_vader[word] * w4
                    #total_sum += 2 * w4                
                else:
                    total_sum += variables.dic_negative_vader[word] * w4
                    #total_sum -= 1 * w4
                
                dic_quantity += 1
                total_weight += w4

        # SLANG
        if(variables.use_dic_slang and variables.dic_slang_loaded and w5 != 0):
            if word in variables.dic_positive_slang:
                
                #print("word " + word + " on slang with the value " + str(variables.dic_positive_slang[word]))
                
                if invert:
                    total_sum -= variables.dic_positive_slang[word] * w5
                    #total_sum -= 1 * w5
                elif booster:
                    total_sum += 2 * variables.dic_positive_slang[word] * w5
                    #total_sum += 2 * w5
                elif boosterAndInverter:
                    total_sum -= 2 * variables.dic_positive_slang[word] * w5
                    #total_sum -= 2 * w5                        
                else:
                    total_sum += variables.dic_positive_slang[word] * w5
                    #total_sum += 1 * w5
                
                dic_quantity += 1
                total_weight += w5
            elif word in variables.dic_negative_slang:

                #print("word " + word + " on slang with the value " + str(variables.dic_negative_slang[word]))
                
                if invert:
                    total_sum -= variables.dic_negative_slang[word] * w5
                    #total_sum += 1 * w5
                elif booster:
                    total_sum += 2 * variables.dic_negative_slang[word] * w5
                    #total_sum -= 2 * w5
                elif boosterAndInverter:
                    total_sum -= 2 * variables.dic_negative_slang[word] * w5                     
                else:
                    total_sum += variables.dic_negative_slang[word] * w5
                    #total_sum -= 1 * w5
                             
                dic_quantity += 1
                total_weight += w5  
        
        # EFFECT
        if(variables.use_dic_effect and variables.dic_effect_loaded and w6 != 0):
            if word in variables.dic_positive_effect:
                
                #print("word " + word + " on effect with the value " + str(variables.dic_positive_effect[word]))

                if invert:
                    total_sum -= variables.dic_positive_effect[word] * w6
                    #total_sum -= 1 * w6
                elif booster:
                    total_sum += 2 * variables.dic_positive_effect[word] * w6
                    #total_sum += 2 * w6
                elif boosterAndInverter:
                    total_sum -= 2 * variables.dic_positive_effect[word] * w6
                    #total_sum -= 2 * w6
                else:
                    total_sum += variables.dic_positive_effect[word] * w6
                    #total_sum += 1 * w6
                
                dic_quantity += 1 
                total_weight += w6
            elif word in variables.dic_negative_effect:
                
                #print("word " + word + " on effect with the value " + str(variables.dic_negative_effect[word]))

                if invert:
                    total_sum -= variables.dic_negative_effect[word] * w6
                    #total_sum += 1 * w6
                elif booster:
                    total_sum += 2 * variables.dic_negative_effect[word] * w6
                    #total_sum -= 2 * w6
                elif boosterAndInverter:
                    total_sum -= 2 * variables.dic_negative_effect[word] * w6
                    #total_sum += 2 * w6
                else:
                    total_sum += variables.dic_negative_effect[word] * w6
                    #total_sum -= 1 * w6
                
                dic_quantity += 1 
                total_weight += w6

        # SEMEVAL2015
        if(variables.use_dic_semeval2015 and variables.dic_semeval2015_loaded and w7 != 0):
            if word in variables.dic_positive_semeval2015:
                
                #print("word " + word + " on semeval2015 with the value " + str(variables.dic_positive_semeval2015[word]))

                if invert:
                    total_sum -= variables.dic_positive_semeval2015[word] * w7
                    #total_sum -= 1 * w7
                elif booster:
                    total_sum += 2 * variables.dic_positive_semeval2015[word] * w7
                    #total_sum += 2 * w7
                elif boosterAndInverter:
                    #total_sum -= 2 * w7 
                    total_sum -= 2 * variables.dic_positive_semeval2015[word] * w7                    
                else:
                    #total_sum += variables.dic_positive_semeval2015[word]
                    total_sum += variables.dic_positive_semeval2015[word] * w7
            
                dic_quantity += 1
                total_weight += w7

            elif word in variables.dic_negative_semeval2015:

                #print("word " + word + " on semeval2015 with the value " + str(variables.dic_negative_semeval2015[word]))

                if invert:
                    total_sum -= variables.dic_negative_semeval2015[word] * w7
                    #total_sum += 1 * w7
                elif booster:
                    total_sum += 2* variables.dic_negative_semeval2015[word] * w7
                    #total_sum -= 2 * w7
                elif boosterAndInverter:
                    total_sum -= 2 * variables.dic_negative_semeval2015[word] * w7
                    #total_sum += 2 * w7                      
                else:
                    total_sum += variables.dic_negative_semeval2015[word] * w7
                    #total_sum -= 1 * w7
                        
                dic_quantity += 1
                total_weight += w7                                            
        
        if (dic_quantity > 1) and (total_weight != 0):
            total_sum_return += round(total_sum/total_weight, 4)
        elif (dic_quantity == 1):
            total_sum_return += total_sum

        dic_quantity = 0
        total_sum    = 0
        #total_weight = 0

        index += 1 # word of phrase
        invert = False
        booster = False
        boosterAndInverter = False
    

    return total_sum_return


def replaceNegatingWords(phrase):
    phrase = phrase.lower()
    replaced = False

    splitted_phrase = phrase.split()

    if(len(splitted_phrase) > 0 and splitted_phrase[-1] == "alreadynegatedbefore"):
        return phrase  

    if len(splitted_phrase) > 0 and splitted_phrase[0] in variables.dic_negation_words:
        phrase_list = splitted_phrase
        phrase_list[0] = "insidenoteinverterword"
        phrase = ' '.join(phrase_list)

    for negation_word in variables.dic_negation_words:
        negation_word = " " + negation_word + " "
        if negation_word in phrase:
        #if phrase.find(negation_word) > -1:
            #print("A) [i'll negate the phrase '" + phrase + "' because it have the word " + negation_word + "]")
            phrase = phrase.replace(negation_word, " insidenoteinverterword ")
            replaced = True
            
    if (replaced):
        return phrase + " alreadynegatedbefore"
    else:
        return phrase


def replaceBoosterWords(phrase):
    phrase = phrase + " "
    phrase = phrase.lower()
    replaced = False

    splitted_phrase = phrase.split()

    if(len(splitted_phrase) > 0 and splitted_phrase[-1] == "alreadyboosteredbefore"):
        return phrase
    
    if len(phrase.split()) > 0 and phrase.split()[0] in variables.dic_booster_words:
        #print("1) [i'll boost the phrase '" + phrase + "' because it have the word " + phrase.split()[0] + "]")
        phrase_list = phrase.split()
        phrase_list[0] = "insidenoteboosterword"
        phrase = ' '.join(phrase_list)

    for booster_word in variables.dic_booster_words:
        booster_word = " " + booster_word + " "
        if booster_word in phrase: 
        #if phrase.find(booster_word) > -1:
            #print("has booster " + booster_word)
            #print("2) [i'll boost the phrase '" + phrase + "' because it have the word " + booster_word + "]")
            phrase = phrase.replace(booster_word, " insidenoteboosterword ")
            replaced = True
        
        #elif booster_word[-1] in phrase:
        #elif phrase.find(booster_word[-1]) > -1:
            #print("booster_word[-1]: " + booster_word[-1])
            #print("3 i'll boost the phrase '" + phrase + "' because it have the word " + booster_word)
            #phrase = phrase.replace(booster_word, " insidenoteboosterword ")

    if (replaced):
        return phrase + " alreadyboosteredbefore"
    else:
        return phrase


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
        hashtags = re.findall(r"#(\w+)", phrase)

        for hashtag in hashtags:
            if hashtag.lower().strip() in variables.dic_negative_hashtags:
                total += 1 
            else:
                if hashtag.lower().strip() in variables.dic_negative_words:
                    total += 1 

    return total


def hasDates(phrase):
    mm = phrase.split()
    for x in mm:
        x = x.replace(",", "").replace("(", "").replace(")", "").replace(";", "").replace(":", "").replace("?", "").replace("!", "")
        if x.lower() in variables.all_dates or len(re.findall('^[12][0-9]{3}$', x)) > 0 or len(re.findall('^([01]\d|2[0-3]):?([0-5]\d)$', x)) > 0:
            return True

    return False
    #print(x + " - " + str(x.lower() in variables.all_dates) + " " + str(len(re.findall('^[12][0-9]{3}$', x))) + " " + str(len(re.findall('^([01]\d|2[0-3]):?([0-5]\d)$', x))))

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
def getURLs(phrase):
    #improve this Regular Expression to get www.something.com and others
    return re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', phrase)


def hasEmails(phrase):
    words = phrase.split()
    for word in words:
        if validate_email(word):
            return True

    return False


def hasURLs(phrase):
    if len(getURLs(phrase)) > 0:
        return True
    return False


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
    return phrase_copy.translate(str.maketrans('','',string.punctuation.replace("-", "").replace("#", ""))) # keep hyphens

# Testing
neutral_url_qtty = 0
neutral_url_correct_pred = 0
# Testing

model_results_to_count_occurrences = []

def mutateW(individual):
    mutated_individual = ""
    #print("I'm in mutateW")
    #print("I received the individual above:\n")
    #print(str(individual))
    #print(type(individual))
    #print(dir(individual))
    #print("fitness -> " + str(individual.fitness))
    #print("----\n")
    #for x in str(individual).split(","):
    #    print("x part -> " + str(x))
    #print("----\n")

    return individual,


# Evaluate the test messages using the model
# http://text-analytics101.rxnlp.com/2014/10/computing-precision-and-recall-for.html
def evaluateMessages(base, model):
    global model_results_to_count_occurrences
    print("[starting evaluation of " + base + " messages]")
    
    # test
    global neutral_url_qtty
    global neutral_url_correct_pred
    count_has_date = 0
    # test

    # test
    variables.neutral_inferior_range = 0
    variables.neutral_superior_range = 0
    # test

    neutral_url_qtty = 0
    neutral_url_correct_pred = 0

    neutral_because_url = False
    # Testing

    # parameters to calc the metrics
    true_positive  = 0
    true_negative  = 0
    false_positive = 0
    false_negative = 0
    true_neutral   = 0
    false_neutral  = 0

    # confusion matrix 
    goldPos_classNeg = 0
    goldPos_classNeu = 0
    goldNeg_classPos = 0
    goldNeg_classNeu = 0
    goldNeu_classPos = 0
    goldNeu_classNeg = 0
    # confusion matrix 

    # calc of mode
    goldPos_classPos_value = []
    goldPos_classNeg_value = []
    goldPos_classNeu_value = []
    goldNeg_classPos_value = []
    goldNeg_classNeg_value = []
    goldNeg_classNeu_value = []
    goldNeu_classPos_value = []
    goldNeu_classNeg_value = []
    goldNeu_classNeu_value = []
    # calc of mode
    
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
    
    total_positive = variables.tweets_2013_positive + variables.tweets_2014_positive + variables.tweets_liveJournal2014_positive + variables.tweets_2014_sarcasm_positive + variables.sms_2013_positive
    total_negative = variables.tweets_2013_negative + variables.tweets_2014_negative + variables.tweets_liveJournal2014_negative + variables.tweets_2014_sarcasm_negative + variables.sms_2013_negative
    total_neutral  = variables.tweets_2013_neutral  + variables.tweets_2014_neutral  + variables.tweets_liveJournal2014_neutral  + variables.tweets_2014_sarcasm_neutral  + variables.sms_2013_neutral 

    if len(variables.tweets_2013) == 0:
        loadTestTweets()
        #loadTestTweets_smuk()

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
        messages = variables.all_messages_in_file_order
        messages_score = variables.all_polarities_in_file_order
        #messages = variables.tweets_2013 + variables.tweets_2014 + variables.tweets_liveJournal2014 + variables.sms_2013 + variables.tweets_2014_sarcasm
        #messages_score = variables.tweets_2013_score + variables.tweets_2014_score + variables.tweets_liveJournal2014_score + variables.sms_2013_score + variables.tweets_2014_sarcasm_score
        messages_positive = variables.tweets_2013_positive + variables.tweets_2014_positive + variables.tweets_liveJournal2014_positive + variables.tweets_2014_sarcasm_positive + variables.sms_2013_positive
        messages_negative = variables.tweets_2013_negative + variables.tweets_2014_negative + variables.tweets_liveJournal2014_negative + variables.tweets_2014_sarcasm_negative + variables.sms_2013_negative
        messages_neutral  = variables.tweets_2013_neutral  + variables.tweets_2014_neutral  + variables.tweets_liveJournal2014_neutral  + variables.tweets_2014_sarcasm_neutral  + variables.sms_2013_neutral

    for index, item in enumerate(messages): 
        message = str(messages[index]).strip().replace("'", "")
        message = message.replace("\\u2018", "").replace("\\u2019", "").replace("\\u002c", "")        
        message = "'" + message + "'"

        model_analysis = model.replace("(x", "(" + message)
        
        if not len(message) > 0:
            continue
        
        neutral_because_url = False
        
        try:
            if(variables.use_emoticon_analysis and hasEmoticons(message)):
                result = emoticonsPolaritySum(message)

            # If the tweet has url (and email now), set neutral (intuition to test)
            #elif(variables.use_url_to_neutral and len(getURLs(message)) > 0 and hasEmails(message)):
            elif(variables.use_url_to_neutral and (len(getURLs(message)) > 0 or hasEmails(message))):
                result = float(eval(model_analysis))
                if result < 0.5:                
                    result = 0
                    neutral_url_qtty += 1
                    neutral_because_url = True

            elif(variables.use_url_and_date_to_neutral and len(getURLs(message)) > 0 and hasDates(message)):
                result = float(eval(model_analysis))
                if result < 0.5:                
                    result = 0

            elif(variables.use_date_to_neutral and hasDates(message)):
                result = float(eval(model_analysis))
                if result < 0.5:
                    result = 0                

            # Check if SVM are saying that the message are neutral
            elif(variables.use_svm_neutral and variables.svm_normalized_values[index] == 0):
                result = 0

            # SVM only
            elif(variables.use_only_svm):
                result = variables.svm_normalized_values[index]
            
            # GP only
            else:
                result = float(eval(model_analysis))


        except Exception as e:
            print("exception 2: " + str(e))
            #print("\n\n[WARNING] eval(model_analysis) exception for the message: " + message + "\n\n")
            continue

        if(base == "all"):
            model_results_to_count_occurrences.append(result)


        #variables.neutral_superior_range = variables.neutral_superior_range
        
        if messages_score[index] > 0:
            if result > variables.neutral_superior_range:
                true_positive += 1
                if base == "all":
                    goldPos_classPos_value.append(result)
            else:
                if result >= variables.neutral_inferior_range and result <= variables.neutral_superior_range:
                    false_neutral += 1
                    goldPos_classNeu += 1
                    if base == "all":
                        goldPos_classNeu_value.append(result)
                elif result < variables.neutral_inferior_range:
                    false_negative += 1
                    goldPos_classNeg += 1
                    if base == "all":
                        goldPos_classNeg_value.append(result)
                
        elif messages_score[index] < 0:
            if result < variables.neutral_inferior_range:
                true_negative += 1
                if base == "all":
                    goldNeg_classNeg_value.append(result)
            else:
                false_negative_log += 1
                if result >= variables.neutral_inferior_range and result <= variables.neutral_superior_range:
                    false_neutral += 1
                    goldNeg_classNeu += 1
                    if base == "all":
                        goldNeg_classNeu_value.append(result)
                elif result > variables.neutral_superior_range:
                    false_positive += 1
                    goldNeg_classPos += 1
                    if base == "all":
                        goldNeg_classPos_value.append(result)                        

                #if false_negative_log <= 20:
                    #if false_negative_log == 1:  
                    #    print("\n##### [FALSE NEGATIVES][" + base + "] #####\n")
                    #print("[Negative phrase]: " + message)
                    #print("[Polarity calculated]: " + str(result))

        elif messages_score[index] == 0:
            if result >= variables.neutral_inferior_range and result <= variables.neutral_superior_range:
                true_neutral += 1
                if(neutral_because_url == True):
                    neutral_url_correct_pred += 1
                if base == "all":
                    goldNeu_classNeu_value.append(result)
            else:
                if result < variables.neutral_inferior_range:
                    false_negative += 1
                    goldNeu_classNeg += 1
                    if base == "all":
                        goldNeu_classNeg_value.append(result)

                        # LOG - Check the neutral messages that are defined as positive
                        print("[Neutral message] " + message)
                        print("[-][Negative Polarity calculated] " + str(result) + " | [Neutral inferior range] " + str(variables.neutral_inferior_range) + " | [Neutral superior range] " + str(variables.neutral_superior_range))
                        print("\n")

                elif result > variables.neutral_superior_range:
                    false_positive += 1
                    goldNeu_classPos += 1
                    if base == "all":
                        goldNeu_classPos_value.append(result)

                        # LOG - Check the neutral messages that are defined as positive
                        print("[Neutral message] " + message)
                        print("[+][Positive Polarity calculated] " + str(result) + " | [Neutral inferior range] " + str(variables.neutral_inferior_range) + " | [Neutral superior range] " + str(variables.neutral_superior_range))
                        print("\n")

                        #if (hasDates(message)):
                        #    print("HAS DATE: " + message)
                        #    count_has_date += 1
                        #else:
                        #    print("DOESN'T HAS DATE: " + message)


    print(str(count_has_date))

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
    if (base == "all"):
        print("[messages evaluated]: " + str(len(messages)) + " (" + str(total_positive) + " positives, " + str(total_negative) + " negatives, " + str(total_neutral) + " neutrals)")
    else:
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
    print("[dictionary quantity]: " + str(variables.dic_loaded_total))
    
    if(variables.use_url_to_neutral):
        print("\nNeutral choosed because of URL -> " + str(neutral_url_qtty))
        print("\nCorrect Neutral prediction because of URL -> " + str(neutral_url_correct_pred))
    
    print("\n")
    print("Confusion Matrix\n")
    print("          |  Gold_Pos  |  Gold_Neg  |  Gold_Neu  |")
    print("--------------------------------------------------")
    print("Pred_Pos  |  " + '{message: <{width}}'.format(message=str(true_positive), width=8) + "  |  " + '{message: <{width}}'.format(message=str(goldNeg_classPos), width=8) + "  |  " + '{message: <{width}}'.format(message=str(goldNeu_classPos), width=8) + "  |")
    print("Pred_Neg  |  " + '{message: <{width}}'.format(message=str(goldPos_classNeg), width=8) + "  |  " + '{message: <{width}}'.format(message=str(true_negative), width=8) + "  |  " + '{message: <{width}}'.format(message=str(goldNeu_classNeg), width=8) + "  |")
    print("Pred_Neu  |  " + '{message: <{width}}'.format(message=str(goldPos_classNeu), width=8) + "  |  " + '{message: <{width}}'.format(message=str(goldNeg_classNeu), width=8) + "  |  " + '{message: <{width}}'.format(message=str(true_neutral), width=8)  + "  |")

    print("\n")

    if variables.save_file_results:
        with open(variables.FILE_RESULTS, 'a') as f:
            if base == "tweets2013":
                f.write("[Model]\t" + model + "\n")
            f.write(base + "\t" + str(round(f1_positive_negative_avg, 4)) + "\n")
            if base == "all":
                f.write("\n")

'''
    if (base == "all"):
        with open("commom_values_goldPos_classPos " + str(datetime.now()) + ".txt", 'a') as f:
            f.write(str(Counter(goldPos_classPos_value)))
        with open("commom_values_goldPos_classNeg " + str(datetime.now()) + ".txt", 'a') as f:
            f.write(str(Counter(goldPos_classNeg_value)))            
        with open("commom_values_goldPos_classNeu " + str(datetime.now()) + ".txt", 'a') as f:
            f.write(str(Counter(goldPos_classNeu_value)))

        with open("commom_values_goldNeg_classPos " + str(datetime.now()) + ".txt", 'a') as f:
            f.write(str(Counter(goldNeg_classPos_value)))
        with open("commom_values_goldNeg_classNeg " + str(datetime.now()) + ".txt", 'a') as f:
            f.write(str(Counter(goldNeg_classNeg_value)))
        with open("commom_values_goldNeg_classNeu " + str(datetime.now()) + ".txt", 'a') as f:
            f.write(str(Counter(goldNeg_classNeu_value))) 

        with open("commom_values_goldNeu_classPos " + str(datetime.now()) + ".txt", 'a') as f:
            f.write(str(Counter(goldNeu_classPos_value)))
        with open("commom_values_goldNeu_classNeg " + str(datetime.now()) + ".txt", 'a') as f:
            f.write(str(Counter(goldNeu_classNeg_value)))
        with open("commom_values_goldNeu_classNeu " + str(datetime.now()) + ".txt", 'a') as f:
            f.write(str(Counter(goldNeu_classNeu_value)))                                        
'''

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
        if models > 0:
            f.write("\n\n##Statistics##\n\n")
            f.write(str(models) + " models evaluated\n")
            f.write(str(variables.dic_loaded_total) + " dictionaries\n\n")
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

    #N = models
    #ind = np.arange(N)  # the x locations for the groups
    #width = 0.10      # the width of the bars

    #fig, ax = plt.subplots()
    
    #rects1 = ax.bar(ind, t2k13_list, width, color='r')
    #rects2 = ax.bar(ind + width, t2k14_list, width, color='y')
    #rects3 = ax.bar(ind + width * 2, sms_list, width, color='b')
    #rects4 = ax.bar(ind + width * 3, liveJ_list, width, color='g')
    #rects5 = ax.bar(ind + width * 4, sarcasm_list, width, color='k')

    # add some text for labels, title and axes ticks
    #ax.set_ylabel('F1')
    #ax.set_xlabel('Models')
    #ax.set_title('F1 by database')
    #ax.set_xticks(ind + width / 2)
    #ax.set_xticklabels(np.arange(models))
    #ax.set_xticklabels(('Tweets2013', 'Tweets2014', 'SMS', 'SARCASM', 'LiveJournal'))

    #ax.legend((rects1[0], rects2[0], rects3[0], rects4[0], rects5[0]), ('Twitter2013', 'Twitter2014', 'SMS', 'LiveJournal', 'Sarcasm'))

    #def autolabel(rects):
        #for rect in rects:
        #    height = rect.get_height()
        #    ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
        #            '%.2f' % float(height),
        #            ha='center', va='bottom')

    #autolabel(rects1)
    #autolabel(rects2)
    #autolabel(rects3)
    #autolabel(rects4)
    #autolabel(rects5)

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