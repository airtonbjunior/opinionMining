# -*- coding: utf-8 -*-

#########################################################################
############## Semeval - Sentiment Analysis in Twitter  #################
#########################################################################

####
#### Authors: Pedro Paulo Balage Filho e Lucas Avanço
#### Version: 2.0
#### Date: 26/03/14
####

# Python 3 compatibility
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import with_statement
from __future__ import unicode_literals

# Import the libraries created for this task
from RulesClassifier import RulesClassifier
from LexiconClassifier import LexiconClassifier
from MachineLearningClassifier import MachineLearningClassifier
from PreProcess import pre_process

# Import other libraries used
import pickle
import codecs
import os
import sys

import var
#### Provides a hybrid Sentiment Analysis classifier.
#### This classifier was designed for Semeval 2014 Task 9 - Sentiment Analysis in Twitter
#### Information about Semeval format can be found at:
####    http://alt.qcri.org/semeval2014/task9/
####
#### The trainset must be in SemevalTwitter format. See SemevalTwitter.py for information.
class TwitterHybridClassifier(object):

    def __init__(self, tweets=[]):
        # initialize internal variables
        self.rules_classifier = RulesClassifier()
        self.lexicon_classifier = LexiconClassifier()
        self.ml_classifier = None

        # if the ML model has been generated, load the model from model.pkl
        if sys.version_info >= (3,0):
            if os.path.exists(str(var.model_classifier) + '-model_python3.pkl'):
                print ('Reading the ' + str(var.model_classifier) + ' model from model_python3.pkl')
                self.ml_classifier = pickle.load(open(str(var.model_classifier) + '-model_python3.pkl','rb'))
        else:
            if os.path.exists(str(var.model_classifier) + '-model_python2.pkl'):
                print ('Reading the ' + str(var.model_classifier) + ' model from model_python2.pkl')
                self.ml_classifier = pickle.load(open(str(var.model_classifier) + '-model_python2.pkl','rb'))

        if self.ml_classifier == None:
            # Preprocess the data and train a new model
            print ('Preprocessing the training data')
            tweet_messages = [tweet_message for tweet_message,label in tweets]
            tweet_labels = [label for tweet_message,label in tweets]

            # preproces all the tweet_messages (Tokenization, POS and normalization)
            tweet_tokens = pre_process(tweet_messages)

            # compile a trainset with tweek_tokens and labels (positive,
            # negative or neutral)

            trainset = [(tweet_tokens[i],tweet_labels[i]) for i in range(len(tweets))]

            # initialize the classifier and train it
            classifier = MachineLearningClassifier(trainset)

            # dump the model into de pickle
            python_version = sys.version_info[0]
            model_name = str(var.model_classifier) + '-model_python' + str(python_version) + '.pkl'
            print ('Saving the trained model at ' + model_name)
            pickle.dump(classifier, open(model_name, 'wb'))
            self.ml_classifier = classifier

    # Apply the classifier over a tweet message in String format
    def classify(self,tweet_text):

        # 0. Pre-process the teets (tokenization, tagger, normalizations)
        tweet_tokens_list = []

        print ('Preprocessing the string')
        # pre-process the tweets
        tweet_tokens_list = pre_process([tweet_text])

        predictions = []
        total_tweets = len(tweet_tokens_list)

        # iterate over the tweet_tokens
        for index, tweet_tokens in enumerate(tweet_tokens_list):

            # 1. Rule-based classifier. Look for emoticons basically
            positive_score,negative_score = self.rules_classifier.classify(tweet_tokens)

            # 1. Apply the rules, If any found, classify the tweet here. If none found, continue for the lexicon classifier.
            if positive_score >= 1 and negative_score == 0:
                sentiment = ('positive','RB')
                predictions.append(sentiment)
                continue
            elif positive_score == 0 and negative_score <= -1:
                sentiment = ('negative','RB')
                predictions.append(sentiment)
                continue

            # 2. Lexicon-based classifier
            positive_score, negative_score = self.lexicon_classifier.classify(tweet_tokens)
            lexicon_score = positive_score + negative_score

            # 2. Apply lexicon classifier,
            # If in the threshold classify the tweet here. If not, continue for the ML classifier
            if positive_score >= 1 and negative_score == 0:
                sentiment = ('positive','LB')
                predictions.append(sentiment)
                continue
            elif negative_score <= -2:
                sentiment = ('negative','LB')
                predictions.append(sentiment)
                continue

            # 3. Machine learning based classifier - used the Train+Dev set sto define the best features to classify new instances
            result = self.ml_classifier.classify(tweet_tokens)
            positive_conf = result['positive']
            negative_conf = result['negative']
            neutral_conf = result['neutral']

            if negative_conf >= -0.4:
                sentiment = ('negative','ML')
            elif positive_conf > neutral_conf:
                sentiment = ('positive','ML')
            else:
                sentiment = ('neutral','ML')

            predictions.append(sentiment)

        return predictions

    # Apply the classifier in batch over a list of tweet messages in String format
    def classify_batch(self,tweet_texts):

        # 0. Pre-process the teets (tokenization, tagger, normalizations)
        tweet_tokens_list = []

        if len(tweet_texts) == 0:
            return tweet_tokens_list

        print ('Preprocessing the test data')
        # pre-process the tweets
        tweet_tokens_list = pre_process(tweet_texts)

        predictions = []
        total_tweets = len(tweet_tokens_list)

        line_save = []

        my_index = 0

        # iterate over the tweet_tokens
        for index, tweet_tokens in enumerate(tweet_tokens_list):

            print('Testing for tweet n. {}/{}'.format(index+1,total_tweets))

            '''
            I comment this part to classify all the messages using only the ML method (airtonbjunior)

            # 1. Rule-based classifier. Look for emoticons basically
            positive_score,negative_score = self.rules_classifier.classify(tweet_tokens)

            # 1. Apply the rules, If any found, classify the tweet here. If none found, continue for the lexicon classifier.
            if positive_score >= 1 and negative_score == 0:
                sentiment = ('positive','RB')
                predictions.append(sentiment)
                continue
            elif positive_score == 0 and negative_score <= -1:
                sentiment = ('negative','RB')
                predictions.append(sentiment)
                continue

            # 2. Lexicon-based classifier
            positive_score, negative_score = self.lexicon_classifier.classify(tweet_tokens)
            lexicon_score = positive_score + negative_score

            # 2. Apply lexicon classifier,
            # If in the threshold classify the tweet here. If not, continue for the ML classifier
            if positive_score >= 1 and negative_score == 0:
                sentiment = ('positive','LB')
                predictions.append(sentiment)
                continue
            elif negative_score <= -2:
                sentiment = ('negative','LB')
                predictions.append(sentiment)
                continue
            '''
            
            # 3. Machine learning based classifier - used the Train+Dev set sto define the best features to classify new instances
            result = self.ml_classifier.classify(tweet_tokens)
            #print(str(result))
            #input("Press enter to continue...")
            positive_conf = result['positive']
            negative_conf = result['negative']
            neutral_conf  = result['neutral']

            line_save.append(str(positive_conf) + '\t' + str(negative_conf) + '\t' + str(neutral_conf))

            #print(str(positive_conf))
            #print(str(negative_conf))
            #print(str(neutral_conf))

            if var.model_classifier == "svm":
                if negative_conf >= -0.4:
                    sentiment = ('negative','ML')
                elif positive_conf > neutral_conf:
                    sentiment = ('positive','ML')
                else:
                    sentiment = ('neutral','ML')
            elif var.model_classifier == "randomForest":
                if positive_conf > negative_conf and positive_conf > neutral_conf:
                    sentiment = ('positive','ML')
                elif negative_conf > positive_conf and negative_conf > neutral_conf:
                    sentiment = ('negative','ML')
                elif neutral_conf > positive_conf and neutral_conf > negative_conf:
                    sentiment = ('neutral','ML')
                else:
                    if positive_conf == neutral_conf:
                        sentiment = ('positive','ML')
                    elif negative_conf == neutral_conf:
                        sentiment = ('negative','ML')
                    else:
                        sentiment = ('neutral','ML')
            elif var.model_classifier == "naive":
                #sentiment = var.naive_raw_predict[my_index]
                #print(str(sentiment))
                sentiment = ""

            elif var.model_classifier == "lreg":
                if positive_conf > negative_conf and positive_conf > neutral_conf:
                    sentiment = ('positive','ML')
                elif negative_conf > positive_conf and negative_conf > neutral_conf:
                    sentiment = ('negative','ML')
                elif neutral_conf > positive_conf and neutral_conf > negative_conf:
                    sentiment = ('neutral','ML')


            elif var.model_classifier == "sgd":
                if positive_conf > negative_conf and positive_conf > neutral_conf:
                    sentiment = ('positive','ML')
                elif negative_conf > positive_conf and negative_conf > neutral_conf:
                    sentiment = ('negative','ML')
                elif neutral_conf > positive_conf and neutral_conf > negative_conf:
                    sentiment = ('neutral','ML')

            predictions.append(sentiment)
            my_index += 1

        print('Saving the predictions values of ' + str(var.model_classifier) + ' on file ' + str(var.model_classifier) + '_test_results.txt')
        with open(str(var.model_classifier) + '_test_results.txt', 'a') as fr:
            ii = 0
            for pred in line_save:
                if (var.model_classifier) == "randomForest":
                    fr.write(pred + '\t' + str(var.rf_predicts[ii])[2:-2]  + '\n')
                elif (var.model_classifier) == "svm":
                    fr.write(pred + '\t' + str(var.svm_predicts[ii][2:-2]) + '\n')
                elif (var.model_classifier) == "naive":
                    fr.write(pred + '\t' + str(var.naive_predicts[ii][2:-2]) + '\n')
                elif (var.model_classifier) == "lreg":
                    fr.write(pred + '\t' + str(var.lreg_predicts[ii]) + '\n')
                elif (var.model_classifier) == "sgd":
                    fr.write(pred + '\t' + str(var.sgd_predicts[ii]) + '\n')                    
                ii += 1

        return predictions

    # Output Individual scores for each method
    def output_individual_scores(self,tweets):

        tweet_texts = [tweet_message for tweet_message,label in tweets]
        tweet_labels = [label for tweet_message,label in tweets]

        # write the log
        fp = codecs.open('individual_scores.tab','w',encoding='utf8')
        line = 'pos_score_rule\tneg_score_rule\tpos_score_lex\tneg_score_lex\tpos_conf\tneg_conf\tneutral_conf\tclass\tmessage\n'
        fp.write(line)

        # 0. Pre-process the text (emoticons, misspellings, tagger)
        tweet_tokens_list = None
        tweet_tokens_list = pre_process(tweet_texts)

        predictions = []
        for index,tweet_tokens in enumerate(tweet_tokens_list):
            line = ''

            # 1. Rule-based classifier. Look for emoticons basically
            positive_score,negative_score = self.rules_classifier.classify(tweet_tokens)
            line += str(positive_score) + '\t' + str(negative_score) + '\t'

            # 2. Lexicon-based classifier (using url_score obtained from RulesClassifier)
            positive_score, negative_score = self.lexicon_classifier.classify(tweet_tokens)
            lexicon_score = positive_score + negative_score
            line += str(positive_score) + '\t' + str(negative_score) + '\t'

            # 3. Machine learning based classifier - used the training set to define the best features to classify new instances
            result = self.ml_classifier.decision_function(tweet_tokens)
            line += str(result['positive']) + '\t' + str(result['negative']) + '\t' + str(result['neutral']) + '\t'

            line += tweet_labels[index] + '\t"' + tweet_texts[index].replace('"','') + '"\n'

            fp.write(line)
        print('Indivual score saved in the file: individual_scores.tab')