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

#### This python script provides a tempalte to run the hybrid sentiment classifier for Semeval 2014 Task 9
#### Information about Semeval format can be found at:
####    http://alt.qcri.org/semeval2014/task9/
####

 # Python 3 compatibility
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import with_statement
from __future__ import unicode_literals

# Import the libraries created for this task
from SemevalTwitter import SemevalTwitter
from TwitterHybridClassifier import TwitterHybridClassifier
from nltk import ConfusionMatrix
from pprint import pprint
import codecs

import var

# Print the confusion matrix for classification
def confusion_matrix(gold,guess):
	correct = 0
	total = len(gold)
	for i in range(len(gold)):
		if guess[i] == gold[i]:
			correct += 1
	accuracy = float(correct) / float(total)
	print('Accuracy: {:.2%}'.format(accuracy))

	# Confusion Matrix
	#cm = ConfusionMatrix(gold, guess)
	#print (cm.pp())

# These are the paths I used in my experient. Due SemEval constraints
# I could not provide these data. However, if you e-mail me,
# I could send you these datasets privately.

#train_path = 'Data/Semeval/TrainSet/twitter-train-gold-B-2014.tsv'
train_path = '../../datasets/train/twitter-train-cleansed-B.txt'
#dev_path   = 'Data/Semeval/DevSet/twitter-dev-gold-B.tsv'
dev_path   = '../../datasets/train/twitter-train-cleansed-B.txt'
#test_path  = 'Data/Semeval/TestSet/2014/SemEval2014-task9-test-B-input.txt'

test_path  = '../../datasets/test/test-b-original-complete.txt'
#test_path  = '../../datasets/train/twitter-train-cleansed-B.txt'

# TestSet from 2013
#test_path  = 'Data/Semeval/TestSet/2013/twitter-test-GOLD-B.tsv'
print('Training/Testing using the ' + str(var.model_classifier) + ' classifier')
print('Reading Datasets and Pre-processing...')
# read the data in the format used by the library
semeval = SemevalTwitter(train_path,dev_path,test_path)
trainset = semeval.trainset
devset   = semeval.devset
testset  = semeval.testset

# Training the supervised model. You should send (tweet_message, label) for training
print('Training...')
tweets = [(tweet['MESSAGE'],tweet['SENTIMENT']) for tweet in trainset]
#tweets += [(tweet['MESSAGE'],tweet['SENTIMENT']) for tweet in devset]
classifier = TwitterHybridClassifier(tweets)


# Apply the classifier for all tweets in the testset
print('Testing...')

# count how many instances were classified by each method
# RB: Ruble-based, LB: Lexicon-base, ML: Machine Learning classifier
count = {'RB':0, 'LB':0, 'ML':0 }

# Evaluate if tested with the gold standard
guess = list()
gold = list()

# Keep the predictions string
output = ''

# Load test set tweets
#tweets = [tweet['MESSAGE'] for tweet in devset]
tweets = [tweet['MESSAGE'] for tweet in testset]

# Classify each instance in the testeset in the TwitterHybridClassifier loaded before
predictions = classifier.classify_batch(tweets)

print("features test size -> " + str(len(var.features_test)))

#import pickle

#pickle_out = open("../../featuresTest.pickle","ab")
#pickle.dump(var.features_test, pickle_out)
#pickle_out.close()

# Output the semeval prediction file and the evaluation variables
# if testset if provided
if len(testset) > 0:
    for index, tweet in enumerate(testset):
        prediction,method = predictions[index]
        count[method] += 1
        output += tweet['SID'] + '\t' + tweet['UID'] + '\t' + prediction + '\t' + tweet['MESSAGE'] + '\n'
        guess.append(prediction)
        gold.append(tweet['SENTIMENT'])
    confusion_matrix(gold,guess)

    # Write Semeval output file
    output_file =  str(var.model_classifier) + '-task9-NILC_USP-B-twitter-constrained.output'
    codecs.open(output_file,'w','utf8').write(output)

    # Print some statistics
    print ('Statistics -  Number of instances processed by each method')
    print ('Rule Based:       ',count['RB'])
    print ('Lexicon Based:    ',count['LB'])
    print ('Machine Learning: ',count['ML'])

    print ('\nSemeval output file in: ',output_file)

# Print individual scores
#tweets = [(tweet['MESSAGE'],tweet['SENTIMENT']) for tweet in testset]
#classifier.output_individual_scores(tweets)