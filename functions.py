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

import variables

def getDictionary():
    start = time.time()
    print("[loading dictionary]")

    with open(variables.DICTIONARY_POSITIVE_WORDS, 'r') as inF:
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

#    with open('dictionaries/goldStandard.tff', 'r') as inF8:
#        for line8 in inF8:
#            if (line8.split()[1] == "+Effect"):
#                for word in line8.split()[2].split(","):
#                    if word not in dic_negative_words and word not in dic_positive_words:
#                        dic_positive_words.append(word)
#                        #print("[positive word]: " + word)
            
#            elif (line8.split()[1] == "-Effect"):
#                for word in line8.split()[2].split(","):
#                    if word not in dic_positive_words and word not in dic_positive_words:
#                        dic_negative_words.append(word) 
                        #print("[negative word]: " + word)

#    with open('dictionaries/SemEval2015-English-Twitter-Lexicon.txt', 'r') as inF7:
#        for line7 in inF7:
#            #removing composite words for while 
#            if float(line7.split("\t")[0]) > 0 and not ' ' in line7.split("\t")[1].strip():
#                if "#" in line7.split("\t")[1].strip():
#                    dic_positive_hashtags.append(line7.split("\t")[1].strip()[1:])
#                else:
#                    dic_positive_words.append(line7.split("\t")[1].strip())
#            elif float(line7.split("\t")[0]) < 0 and not ' ' in line7.split("\t")[1].strip():
#                if "#" in line7.split("\t")[1].strip():
#                    dic_negative_hashtags.append(line7.split("\t")[1].strip()[1:])
#                else:
#                    dic_negative_words.append(line7.split("\t")[1].strip())

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
    except ZeroDivisionError:
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


# Return the sum of the word polarities (positive[+1], negative[-1])
def polaritySum(phrase):
    total_sum = 0
    index = 0

    words = phrase.split()
    
    for word in words:
        if word.lower().strip() in variables.dic_positive_words:
            if index > 0 and words[index-1] == "insidenoteinverterword":
                total_sum -=1
            else:
                #print("[positive Word]: " + word)
                total_sum += 1 

        if word.lower().strip() in variables.dic_negative_words:
            if index > 0 and words[index-1] == "insidenoteinverterword":
                total_sum +=1
            else:
                #print("[negative Word]: " + word)
                total_sum -= 1

        index += 1    

    return total_sum  


def replaceNegatingWords(phrase):
    phrase = phrase.lower()
    
    if len(phrase.split()) > 0 and phrase.split()[0] in variables.dic_negation_words:
        phrase_list = phrase.split()
        phrase_list[0] = "insidenoteinverterword"
        phrase = ' '.join(phrase_list)

    for negation_word in variables.dic_negation_words:
        negation_word = " " + negation_word + " "
        if phrase.lower().find(negation_word.lower()) > -1:
            phrase = phrase.replace(negation_word, " insidenoteinverterword ")

    return phrase 


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

    #if not variables.stop_words_function_used:
    for word in words:
        if word not in variables.stop_words:
            return_phrase += word + " "               

    #variables.stop_words_function_used = True
    return return_phrase


def stemmingText(phrase):
    words = phrase.split()

    stemmed_phrase = ""

    #if not variables.stem_function_used:
    for word in words:
        stemmed_phrase += stem(word) + " "               

    #variables.stem_function_used = True
    return stemmed_phrase.strip()


def lemmingText(phrase):
    lemmatizer = WordNetLemmatizer()
    words = phrase.split()

    lemmed_phrase = ""

    for word in words:
        # I'm always considering that the word is a verb
        lemmed_phrase += lemmatizer.lemmatize(word, 'v') + " "               

    return lemmed_phrase.strip()


def removeLinks(phrase):
    phrase_copy = phrase
    #if not variables.remove_links_function_used:
    phrase_return = re.sub(r'http\S+', '', phrase_copy, flags=re.MULTILINE)
    variables.remove_links_function_used = True
    return phrase_return

    #return phrase


def removeEllipsis(phrase):
    phrase_copy = phrase
    #if not variables.remove_ellipsis_function_used:
    phrase_return = re.sub('\.{3}', ' ', phrase_copy)
    variables.remove_ellipsis_function_used = True
    return phrase_return
    
    #return phrase


def removeDots(phrase):
    phrase_copy = phrase
    #if not variables.remove_dots_function_used:
    variables.remove_dots_function_used = True
    return re.sub('\.', ' ', phrase_copy)
    
    #return phrase


def removeAllPonctuation(phrase):
    phrase_copy = phrase
    #if not variables.remove_all_ponctuaction_function_used:
    variables.remove_all_ponctuaction_function_used = True
    return phrase_copy.translate(str.maketrans('','',string.punctuation))

    #return phrase

### End functions (improve this - import the functions of the other file)


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
        result = eval(model_analysis)

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

                if false_negative_log <= 15:
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
