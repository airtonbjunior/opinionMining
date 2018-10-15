#!/usr/bin/env python
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

import codecs
import os

import var

#### Class to provide a data reader for Semeval 2014 Task 9 format.
#### The reader was designed for subtask A (sentiment for the twitter message) only.
#### Information about Semeval format can be found at:
####    http://alt.qcri.org/semeval2014/task9/
####
class SemevalTwitter(object):

    # Constructor.
    def __init__(self,train_path,dev_path,test_path):

        # class variables
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.trainset = list()
        self.devset = list()
        self.testset = list()
        self.reader()

    # read semeval dataset format
    # modifies the train, dev and testset class variables
    def reader(self):
        pos, neg, neu = 0, 0, 0
        # Read the trainset
        tweets = []
        if not os.path.exists(self.train_path):
            print('Trainset file not found. However, they are not necessary if provided the training model (file: model_pythonx.pkl)')
        else:
            fp = codecs.open(self.train_path, 'r', encoding='utf8')
            fp = codecs.open(self.train_path, 'r', encoding='utf8')
            lines = fp.readlines()
            for line in lines:
            #for line in fp:
                line = line.strip()
                tweet_line = line.split('\t')
                if len(tweet_line) != 4:
                    print('Error to read TrainSet. Must have 4 args. Line: ', line)
                
                if ((tweet_line[2] == "positive" and pos > 1500) or (tweet_line[2] == "negative" and neg > 1500) or (tweet_line[2] == "neutral" and neu > 1500)) and (var.balance_train_data):
                    #print("skipping " + tweet_line[2] + " tweet because we're using balanced train data")
                    continue
                else:
                    if tweet_line[2] == "positive":
                        pos += 1
                    elif tweet_line[2] == "negative":
                        neg += 1
                    elif tweet_line[2] == "neutral":
                        neu += 1

                    tweet = {}
                    tweet['SID'] = tweet_line[0]
                    tweet['UID'] = tweet_line[1]
                    #sentiment = tweet_line[2][1:-1]
                    sentiment = tweet_line[2]
                    # classes objective and neutral merged as proposed in the task
                    if sentiment in ['objective','objective-OR-neutral']:
                        sentiment = 'neutral'
                    tweet['SENTIMENT'] = sentiment
                    tweet['MESSAGE'] = tweet_line[3].strip()
                    if tweet['MESSAGE'] != 'Not Available':
                        tweets.append(tweet)

            print('training using ' + str(pos) + ' pos ' + str(neg) + ' neg and ' + str(neu) + ' neu messages')
            if (var.balance_train_data):
                print('using balanced train data')
        self.trainset = tweets

        # Read the devset
        tweets = []
        if not os.path.exists(self.dev_path):
            print('Devset file not found. However, they are not necessary if provided the training model (file: model_pythonx.pkl)')
        else:
            fp = codecs.open(self.dev_path, 'r', encoding='utf8')
            lines = fp.readlines()
            for line in lines:
            #for line in fp:
                # corrects the \u and scape caracters in the message (only applies
                # for the dev and testset provided by the organization
                line = line.strip()
                line = line.replace('\\\\', '').replace('\\"', '"').replace("\\'", "'").replace('\\u2019', '\'').replace('\\u002c', ',')
                tweet_line = line.split('\t')
                if len(tweet_line) != 4:
                    print('Error to read TrialSet. Must have 4 args. Line: ', line)
                tweet = {}
                tweet['SID'] = tweet_line[0]
                tweet['UID'] = tweet_line[1]
                sentiment = tweet_line[2]
                # classes objective and neutral merged as proposed in the task
                if sentiment in ['objective','objective-OR-neutral']:
                    sentiment = 'neutral'
                tweet['SENTIMENT'] = sentiment
                tweet['MESSAGE'] = tweet_line[3].strip()
                tweets.append(tweet)

        self.devset = tweets

        # Read the testset
        tweets = []
        if not os.path.exists(self.test_path):
            print('Testset file not found. You should provide this file if you want to replicate Semeval results.')
        else:
            fp = codecs.open(self.test_path, 'r', encoding='utf8')
            lines = fp.readlines()
            for line in lines:
            #for line in fp:
                # corrects the \u and scape caracters in the message (only applies
                # for the dev and testset provided by the organization
                line = line.strip()
                line = line.replace('\\\\', '').replace('\\"', '"').replace("\\'", "'").replace('\\u2019', '\'').replace('\\u002c', ',')
                tweet_line = line.split('\t')
                if len(tweet_line) != 4:
                    print('Error to read TestSet. Must have 4 args. Line: ', line)
                tweet = {}
                tweet['SID'] = tweet_line[0]
                tweet['UID'] = tweet_line[1]
                sentiment = tweet_line[2]
                tweet['SENTIMENT'] = sentiment
                tweet['MESSAGE'] = tweet_line[3]
                tweets.append(tweet)

        self.testset = tweets