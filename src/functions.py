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
import random

from stemming.porter2 import stem
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import numpy as np

from collections import Counter
from datetime import datetime
from validate_email import validate_email

import variables
from loadFunctions import *


# Load the dictionaries
def getDictionary(module):
	start = time.time()
	startDic = 0
	print("\n[loading dictionaries]")
	if (variables.use_original_dic_values):
		print(" [using dictionaries original polarity values]")
	else:
		print(" [using normalized polarity values: +1 pos -1 neg]")

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
			variables.all_positive_words.append(line.lower().strip())

	with open(variables.DICTIONARY_NEGATIVE_HASHTAGS, 'r') as inF:
		for line in inF:
			variables.dic_negative_hashtags.append(line.lower().strip())
			variables.all_negative_words.append(line.lower().strip())

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
				variables.dic_positive_words.append(line.lower().strip())
				variables.all_positive_words.append(line.lower().strip())
				#if module == "train":# and line in variables.all_train_words:
				#    variables.dic_positive_words.append(line)
				#elif module == "test" and line in variables.all_test_words:
				#elif module == "test":# and line in variables.all_test_words:
				#    variables.dic_positive_words.append(line)

		with codecs.open(variables.DICTIONARY_NEGATIVE_WORDS, "r", "latin-1") as inF:
			for line in inF:
				variables.dic_negative_words.append(line.lower().strip())
				variables.all_negative_words.append(line.lower().strip())
				#if module == "train" and line in variables.all_train_words:
				#    variables.dic_negative_words.append(line)
				#elif module == "test" and line in variables.all_test_words:
				#elif module == "test":# and line in variables.all_test_words:
				#    variables.dic_negative_words.append(line)
		
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
								if (variables.use_original_dic_values):
									variables.dic_positive_sentiwordnet[w] = float(splited[2])
								else:
									variables.dic_positive_sentiwordnet[w] = 1

				elif float(splited[2]) < float(splited[3]):
					words = splited[4].lower().strip().split()
					for word in words:
						if not "_" in word:
							w = word[:word.find("#")]
							if(len(w) > 2):
								if (variables.use_original_dic_values):
									variables.dic_negative_sentiwordnet[w] = float(splited[3]) * -1
								else:
									variables.dic_negative_sentiwordnet[w] = -1
							
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
				if (splited[1].lower() == "+effect"):
					for word in splited[2].split(","):
						word = word.lower().strip()
						variables.dic_positive_effect[word] = 1

				elif (splited[1].lower() == "-effect"):
					for word in splited[2].split(","):
						word = word.lower().strip()
						#if (module == "train" and word in variables.all_train_words) or (module == "test" and word in variables.all_test_words):
						#if (module == "train" and w in variables.all_train_words) or (module == "test"):# and w in variables.all_test_words):
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
				# POSITIVE
				if float(splited[0]) > 0 and not ' ' in splited[1].strip():
					if "#" in splited[1].strip():
						variables.dic_positive_hashtags.append(splited[1].strip()[1:])
						if (variables.use_original_dic_values):
							variables.dic_positive_semeval2015[splited[1].strip()[1:]] = float(splited[0])
						else:
							variables.dic_positive_semeval2015[splited[1].strip()[1:]] = 1
					
					# not hashtag
					else:
						if (variables.use_original_dic_values):
							variables.dic_positive_semeval2015[splited[1].strip()] = float(splited[0])
						else:
							variables.dic_positive_semeval2015[splited[1].strip()] = 1

				# NEGATIVE
				elif float(splited[0]) < 0 and not ' ' in splited[1].strip():
					if "#" in splited[1].strip():
						variables.dic_negative_hashtags.append(splited[1].strip()[1:])
						if (variables.use_original_dic_values):
							variables.dic_positive_semeval2015[splited[1].strip()[1:]] = float(splited[0])
						else:
							variables.dic_positive_semeval2015[splited[1].strip()[1:]] = -1
					
					# not hashtag
					else:
						if (variables.use_original_dic_values):
							variables.dic_negative_semeval2015[splited[1].strip()] = float(splited[0])
						else:
							variables.dic_negative_semeval2015[splited[1].strip()] = -1

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
					if (variables.use_original_dic_values):
						variables.dic_positive_affin[splited[0].strip()] = float(splited[1].strip())
					else:
						variables.dic_positive_affin[splited[0].strip()] = 1
				else:
					if (variables.use_original_dic_values):
						variables.dic_negative_affin[splited[0].strip()] = float(splited[1].strip())
					else:
						variables.dic_negative_affin[splited[0].strip()] = -1

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
					if (variables.use_original_dic_values):
						variables.dic_positive_slang[splited[0].strip()] = float(splited[1].strip())
					else:
						variables.dic_positive_slang[splited[0].strip()] = 1

				elif float(line.split("\t")[1].strip()) < 0:
					if (variables.use_original_dic_values):
						variables.dic_negative_slang[splited[0].strip()] = float(splited[1].strip())
					else:
						variables.dic_negative_slang[splited[0].strip()] = -1
		
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
					if (variables.use_original_dic_values):
						variables.dic_positive_vader[splited[0].strip()] = float(splited[1].strip())
					else:
						variables.dic_positive_vader[splited[0].strip()] = 1
				else:
					if (variables.use_original_dic_values):
						variables.dic_negative_vader[splited[0].strip()] = float(splited[1].strip())
					else:
						variables.dic_negative_vader[splited[0].strip()] = -1

		if variables.log_loads:
			print("    [" + str(len(variables.dic_positive_vader) + len(variables.dic_negative_vader)) + " words loaded]")
			print("    [" + str(len(variables.dic_positive_vader)) + " positive and " + str(len(variables.dic_negative_vader)) + " negative]")
			print("      [vader dictionary loaded][" + str(format(time.time() - startDic, '.3g')) + " seconds]\n")
	

	#NRC
	if(variables.use_dic_nrc):
		startDic = time.time()
		print("  [loading NRC]")
		with open(variables.DICTIONARY_NRC, 'r') as inF:
			variables.dic_nrc_loaded = True
			variables.dic_loaded_total += 1
			for line in inF:
				splited = line.split("\t")
				if float(splited[1].strip()) > 0:
					if (variables.use_original_dic_values):
						variables.dic_positive_nrc[splited[0].strip()] = float(splited[1].strip())
					else:
						variables.dic_positive_nrc[splited[0].strip()] = 1

				elif float(line.split("\t")[1].strip()) < 0:
					if (variables.use_original_dic_values):
						variables.dic_negative_nrc[splited[0].strip()] = float(splited[1].strip()) 
					else:
						variables.dic_negative_nrc[splited[0].strip()] = -1

		if variables.log_loads:
			print("    [" + str(len(variables.dic_positive_nrc) + len(variables.dic_negative_nrc)) + " words loaded]")
			print("    [" + str(len(variables.dic_positive_nrc)) + " positive and " + str(len(variables.dic_negative_nrc)) + " negative]")
			print("      [NRC dictionary loaded][" + str(format(time.time() - startDic, '.3g')) + " seconds]\n")

	#GENERAL INQUIRER
	if(variables.use_dic_nrc):
		startDic = time.time()
		print("  [loading general inquirer]")
		with open(variables.DICTIONARY_GENERAL_INQUIRER, 'r') as inF:
			variables.dic_gi_loaded = True
			variables.dic_loaded_total += 1
			for line in inF:
				splited = line.split("\t")
				if float(splited[1].strip()) > 0:
					if (variables.use_original_dic_values):
						variables.dic_positive_gi[splited[0].strip()] = float(splited[1].strip())
					else:
						variables.dic_positive_gi[splited[0].strip()] = 1

				elif float(line.split("\t")[1].strip()) < 0:
					if (variables.use_original_dic_values):
						variables.dic_negative_gi[splited[0].strip()] = float(splited[1].strip())
					else:
						variables.dic_negative_gi[splited[0].strip()] = -1

		if variables.log_loads:
			print("    [" + str(len(variables.dic_positive_gi) + len(variables.dic_negative_gi)) + " words loaded]")
			print("    [" + str(len(variables.dic_positive_gi)) + " positive and " + str(len(variables.dic_negative_gi)) + " negative]")
			print("      [general inquirer dictionary loaded][" + str(format(time.time() - startDic, '.3g')) + " seconds]\n")

	#SENTIMENT140
	if(variables.use_dic_s140):
		startDic = time.time()
		print("  [loading sentiment 140]")
		with open(variables.DICTIONARY_S140, 'r') as inF:
			variables.dic_s140_loaded = True
			variables.dic_loaded_total += 1
			for line in inF:
				splited = line.split("\t")
				if float(splited[1].strip()) > 0:
					if (variables.use_original_dic_values):
						variables.dic_positive_s140[splited[0].strip()] = float(splited[1].strip())
					else:
						variables.dic_positive_s140[splited[0].strip()] = 1

				elif float(line.split("\t")[1].strip()) < 0:
					if (variables.use_original_dic_values):
						variables.dic_negative_s140[splited[0].strip()] = float(splited[1].strip())
					else:
						variables.dic_negative_s140[splited[0].strip()] = -1

		if variables.log_loads:
			print("    [" + str(len(variables.dic_positive_s140) + len(variables.dic_negative_s140)) + " words loaded]")
			print("    [" + str(len(variables.dic_positive_s140)) + " positive and " + str(len(variables.dic_negative_s140)) + " negative]")
			print("      [sentiment 140 dictionary loaded][" + str(format(time.time() - startDic, '.3g')) + " seconds]\n")

	#MPQA
	if(variables.use_dic_mpqa):
		startDic = time.time()
		print("  [loading mpqa]")
		with open(variables.DICTIONARY_MPQA, 'r') as inF:
			variables.dic_mpqa_loaded = True
			variables.dic_loaded_total += 1
			for line in inF:
				splited = line.split(" ")
				#print(str(splited[0].strip()) + " | " + str(splited[1].strip()))
				if float(splited[1].strip()) > 0:
					#if (module == "train" and splited[0].strip() in variables.all_train_words) or (module == "test" and splited[0].strip() in variables.all_test_words):
					#if (module == "train" and splited[0].strip() in variables.all_train_words) or (module == "test"):# and splited[0].strip() in variables.all_test_words):
					variables.dic_positive_mpqa[splited[0].strip()] = float(splited[1].strip())

				elif float(line.split(" ")[1].strip()) < 0:
					#if (module == "train" and splited[0].strip() in variables.all_train_words) or (module == "test" and splited[0].strip() in variables.all_test_words):
					#if (module == "train" and splited[0].strip() in variables.all_train_words) or (module == "test"):# and splited[0].strip() in variables.all_test_words):
					variables.dic_negative_mpqa[splited[0].strip()] = float(splited[1].strip())                                        

		if variables.log_loads:
			print("    [" + str(len(variables.dic_positive_mpqa) + len(variables.dic_negative_mpqa)) + " words loaded]")
			print("    [" + str(len(variables.dic_positive_mpqa)) + " positive and " + str(len(variables.dic_negative_mpqa)) + " negative]")
			print("      [mpqa dictionary loaded][" + str(format(time.time() - startDic, '.3g')) + " seconds]\n")                   

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

	fileWords = variables.TRAIN_WORDS

	if variables.USE_SPELLCHECKED_WORDS:
		print("  [using spellchecked words]")
		fileWords = variables.TRAIN_WORDS_SPELLCHECK
	else:
		print("  [using original words]")

	with open(fileWords, 'r') as file:
		for line in file:
			variables.all_train_words.append(line.replace('\n', '').replace('\r', ''))

	print("    [train words loaded (" + str(len(variables.all_train_words)) + " words)][" + str(format(time.time() - start, '.3g')) + " seconds]\n")


def loadTestWords():
	start = time.time()
	print("\n  [loading test words]")  

	fileWords = variables.TEST_WORDS

	if variables.USE_SPELLCHECKED_WORDS:
		print("  [using spellchecked words]")
		fileWords = variables.TEST_WORDS_SPELLCHECK
	elif variables.USE_ONLY_POS_WORDS:
		print("  [using only pos words]")
		fileWords = variables.TEST_WORDS_POS_TAGGED_W
	else:
		print("  [using original words]")

	start = time.time()
	with open(fileWords, 'r') as file:
		for line in file:
			variables.all_test_words.append(line.replace('\n', '').replace('\r', ''))

	print("    [test words loaded (" + str(len(variables.all_test_words)) + " words)][" + str(format(time.time() - start, '.3g')) + " seconds]\n")


# Load tweets from id (SEMEVAL 2014 database)
def loadTrainTweets():
	start = time.time()
	print("\n[loading train tweets]")

	tweets_loaded    = 0
	all_train_tweets = [] 

	choosed_messages = []

	with open(variables.SEMEVAL_TRAIN_FILE, 'r') as inF:
		for line in inF:
			all_train_tweets.append(line)

	
	for ind in all_train_tweets:
		
		if tweets_loaded < variables.MAX_ANALYSIS_TWEETS:
			if variables.train_using_bagging:
				random.seed()
				i = random.randint(0, variables.train_file_size - 1)
				tweet_parsed = all_train_tweets[i].split("\t")
				
			else:
				tweet_parsed = ind.split("\t")
			
			try:
				if(tweet_parsed[2] != "neutral"):
					if(tweet_parsed[2] == "positive"):
						#if(variables.positive_tweets < variables.MAX_POSITIVES_TWEETS):
						variables.positive_tweets += 1
						variables.tweets_semeval.append(tweet_parsed[3])
						variables.tweets_semeval_score.append(1)
						tweets_loaded += 1
						if variables.train_using_bagging:
							choosed_messages.append(i)
					else:
						#if(variables.negative_tweets < variables.MAX_NEGATIVES_TWEETS):
						variables.negative_tweets += 1
						variables.tweets_semeval.append(tweet_parsed[3])
						variables.tweets_semeval_score.append(-1)
						tweets_loaded += 1
						if variables.train_using_bagging:
							choosed_messages.append(i)
				else:
					#if(variables.neutral_tweets < variables.MAX_NEUTRAL_TWEETS):
					variables.tweets_semeval.append(tweet_parsed[3])
					variables.tweets_semeval_score.append(0)
					variables.neutral_tweets += 1                         
					tweets_loaded += 1
					if variables.train_using_bagging:
						choosed_messages.append(i)
			except:
				print("exception")
				continue
	
	if variables.train_using_bagging:
		print("[using bagging to train][" + str(round((len(set(choosed_messages)) * 100) / tweets_loaded, 2)) + "% of messages loaded][" + str(tweets_loaded) + " total messages][" + str(len(set(choosed_messages))) + " choosed messages]")
		print("[" + str(variables.positive_tweets) + " positives][" + str(variables.negative_tweets) + " negatives][" + str(variables.neutral_tweets) + " neutrals]")
	end = time.time()
	print("   [train tweets loaded (" + str(tweets_loaded) + " tweets)][" + str(format(end - start, '.3g')) + " seconds]\n")
	#input("Press enter to continue...")


def createChunks(message_list, n_chunks):
	import random

	ys = list(message_list)
	random.shuffle(ys)
	
	ylen = len(ys)
	size = int(ylen / n_chunks)
	
	chunks = [ys[0+size*i : size*(i+1)] for i in range(n_chunks)]
	leftover = ylen - size*n_chunks
	
	edge = size * n_chunks
	
	for i in range(leftover):
		chunks[i % n_chunks].append(ys[edge+i])
	
	return chunks


def createIndexChunks(all_index, n_chunks):
	import random

	ys = list(all_index)
	random.shuffle(ys)
	
	ylen = len(ys)
	size = int(ylen / n_chunks)
	
	chunks = [ys[0+size*i : size*(i+1)] for i in range(n_chunks)]
	leftover = ylen - size*n_chunks
	
	edge = size * n_chunks
	
	for i in range(leftover):
		chunks[i % n_chunks].append(ys[edge+i])
	
	return chunks


def createRandomIndexChunks(all_index, n_chunks):
	ys = list(all_index)
	size = int(len(ys) / n_chunks)

	chunks = list()

	for i in range(n_chunks): 
		random.shuffle(ys)
		chunks.append(list(ys[:size]))

	return chunks

# Load the test tweets from Semeval 2017 task 4a
###############################
# TEST
###############################
def loadTestTweetsSemeval2017():
	start = time.time()
	print("\n[loading test tweets - semeval 2017]")
	tweets_loaded = 0
	test_words = []

	with open(variables.SEMEVAL_2017_TEST_FILE, 'r') as inF:
		for line in inF:
			if tweets_loaded < variables.MAX_ANALYSIS_TWEETS or 1==1:
				tweet_parsed = line.split("\t")

				try:
					# TEST USING SVM - KEEP THE ORDER
					variables.all_messages_in_file_order.append(tweet_parsed[2].replace('right now', 'rightnow'))
					if tweet_parsed[1] == "positive":
						variables.all_polarities_in_file_order.append(1)
					elif tweet_parsed[1] == "negative":
						variables.all_polarities_in_file_order.append(-1)
					elif tweet_parsed[1] == "neutral":
						variables.all_polarities_in_file_order.append(0)

					if tweet_parsed[1] == "Twitter2013" or 1==1:
						variables.tweets_2013.append(tweet_parsed[2].replace('right now', 'rightnow'))
						
						if tweet_parsed[1] == "positive":
							variables.tweets_2013_score.append(1)
							variables.tweets_2013_positive += 1

						elif tweet_parsed[1] == "negative":
							variables.tweets_2013_score.append(-1)
							variables.tweets_2013_negative += 1
						
						elif tweet_parsed[1] == "neutral":
							variables.tweets_2013_score.append(0)
							variables.tweets_2013_neutral += 1

					tweets_loaded += 1
				
				except Exception as e:
					print("exception 1 (2017): " + str(e))
					continue

	end = time.time()
	print("  [test tweets (semeval 2017) loaded (" + str(tweets_loaded) + " tweets)][" + str(format(end - start, '.3g')) + " seconds]\n")


def normalize_svm_polarity(polarity_string):  
	svm_values = []
	svm_values = polarity_string.strip().split()

	if(float(svm_values[0]) >= -0.4):
		return -1
	elif(float(svm_values[2]) > float(svm_values[1])):
		return 1
	else:
		return 0


def normalize_rforest_polarity(polarity_string):
	# Check how to do it properly
	# ['negative' 'neutral' 'positive']
	negative_conf, neutral_conf, positive_conf  = float(polarity_string.split()[0]), float(polarity_string.split()[1]), float(polarity_string.split()[2])

	#print(str(type(negative_conf)))
	#print(str(neutral_conf))
	#print(str(positive_conf))

	#input("s")

	if positive_conf > negative_conf and positive_conf > neutral_conf:
		#sentiment = ('positive','ML')
		return 1
	elif negative_conf > positive_conf and negative_conf > neutral_conf:
		#sentiment = ('negative','ML')
		return -1
	elif neutral_conf > positive_conf and neutral_conf > negative_conf:
		#sentiment = ('neutral','ML')
		return 0
	else:
		if positive_conf == neutral_conf:
			#sentiment = ('positive','ML')
			return 1
		elif negative_conf == neutral_conf:
			#sentiment = ('negative','ML')
			return -1
		else:
			#sentiment = ('neutral','ML')
			return 0

def normalize_naive_polarity(polarity_string):
	if polarity_string == "positive":
		return 1
	elif polarity_string == "negative":
		return -1
	elif polarity_string == "neutral":
		return 0


def normalize_MS_polarity(polarity_string):
	# from MS docs (https://westus.dev.cognitive.microsoft.com/docs/services/TextAnalytics.V2.0/operations/56f30ceeeda5650db055a3c9)
	# The API returns a numeric score between 0 and 1. Scores close to 1 indicate positive sentiment, while scores close to 0 indicate negative sentiment. A score of 0.5 indicates the lack of sentiment (e.g. a factoid statement).
	if float(polarity_string) > 0.5:
		return 1
	elif float(polarity_string) < 0.5:
		return -1
	elif float(polarity_string) == 0.5:
		return 0



def getResultsClassifier(file_name):
	results = []
	with open(file_name, 'r') as f:
		for line in f:
			if not line.startswith("#"): # ignore comments
				results.append(line.strip())

	return results


# Load the test tweets from Semeval 2014 task 9
def loadTestTweets():
	start = time.time()
	print("\n[loading test tweets]")

	tweets_loaded = 0

	test_words = []

	# Load results from each classifier
	LReg_results    = getResultsClassifier("datasets/test/lreg_test_results.txt")
	#Naive_results   = getResultsClassifier("datasets/test/Naive_test_results.txt") # class <tab> pos_prob <tab> neg_prob <tab> neu_prob
	Naive_results   = getResultsClassifier("datasets/test/Naive_test_results_probs.txt") # class <tab> pos_prob <tab> neg_prob <tab> neu_prob
	SVM_results     = getResultsClassifier("datasets/test/SVM_test_results.txt")   # confirm the values pattern
	RForest_results = getResultsClassifier("datasets/test/RandomForest_test_results.txt")   # confirm the values pattern
	SGD_results     = getResultsClassifier("datasets/test/sgd_test_results.txt")

	ESumNoPG_results = []
	with open("ensemble_no_pg_results_PRODUCT.txt", "r") as ff:
		for i in ff:
			ESumNoPG_results.append(i)


	with open(variables.SEMEVAL_TEST_FILE, 'r') as inF:
		index_results = 0
		for line in inF:
			if tweets_loaded < variables.MAX_ANALYSIS_TWEETS_TEST:
				tweet_parsed = line.split("\t")

				svm_class, rforest_class, naive_class, MS_class, LReg_class, S140_class, SGD_class, ESumNoPG_class = 0, 0, 0, 0, 0, 0, 0, 0

				try:
					#svm_class     = normalize_svm_polarity(tweet_parsed[3].strip())
					svm_class      = normalize_svm_polarity(SVM_results[index_results].strip())
					#naive_class   = normalize_naive_polarity(tweet_parsed[4].strip())
					naive_class    = normalize_naive_polarity(Naive_results[index_results].split("\t")[0].strip()) # index 0 contains the class - on 1, 2 and 3 we have the probs of each class
					MS_class       = normalize_MS_polarity(str(tweet_parsed[5].strip()))
					#LReg_class    = normalize_naive_polarity(tweet_parsed[6].strip())             # the same pattern of naive classifier ('positive', 'negative' and 'neutral')
					LReg_class     = normalize_naive_polarity(LReg_results[index_results].strip()) # the same pattern of naive classifier ('positive', 'negative' and 'neutral')
					S140_class     = normalize_naive_polarity(tweet_parsed[7].strip())             # the same pattern of naive classifier ('positive', 'negative' and 'neutral')
					rforest_class  = normalize_naive_polarity(RForest_results[index_results].split('\t')[3].strip())
					SGD_class      = normalize_naive_polarity(SGD_results[index_results].split('\t')[3].strip())
					ESumNoPG_class = normalize_naive_polarity(ESumNoPG_results[index_results].strip())
					#print(str(rforest_class))

					index_results += 1

					# TEST USING SVM - KEEP THE ORDER
					variables.all_messages_in_file_order.append(tweet_parsed[2].replace('right now', 'rightnow'))
					if tweet_parsed[0] == "positive":
						variables.all_polarities_in_file_order.append(1)
					elif tweet_parsed[0] == "negative":
						variables.all_polarities_in_file_order.append(-1)
					elif tweet_parsed[0] == "neutral":
						variables.all_polarities_in_file_order.append(0)

					variables.all_polarities_in_file_order_svm.append(svm_class)
					variables.all_polarities_in_file_order_naive.append(naive_class)
					variables.all_polarities_in_file_order_MS.append(MS_class)
					variables.all_polarities_in_file_order_LReg.append(LReg_class)
					variables.all_polarities_in_file_order_S140.append(S140_class)
					variables.all_polarities_in_file_order_RFor.append(rforest_class)
					variables.all_polarities_in_file_order_SGD.append(SGD_class)
					variables.all_polarities_in_file_order_ESumNoPG.append(ESumNoPG_class)


					if tweet_parsed[1] == "Twitter2013":

						variables.tweets_2013.append(tweet_parsed[2].replace('right now', 'rightnow'))
						
						if tweet_parsed[0] == "positive":
							variables.tweets_2013_score.append(1)
							variables.tweets_2013_positive += 1

						elif tweet_parsed[0] == "negative":
							variables.tweets_2013_score.append(-1)
							variables.tweets_2013_negative += 1
						
						elif tweet_parsed[0] == "neutral":
							variables.tweets_2013_score.append(0)
							variables.tweets_2013_neutral += 1

						variables.tweets_2013_score_svm.append(svm_class)
						variables.tweets_2013_score_naive.append(naive_class)
						variables.tweets_2013_score_MS.append(MS_class)
						variables.tweets_2013_score_LReg.append(LReg_class)
						variables.tweets_2013_score_S140.append(S140_class)
						variables.tweets_2013_score_RFor.append(rforest_class)
						variables.tweets_2013_score_SGD.append(SGD_class)
						variables.tweets_2013_score_ESumNoPG.append(ESumNoPG_class)

					elif tweet_parsed[1] == "Twitter2014":
						variables.tweets_2014.append(tweet_parsed[2].replace('right now', 'rightnow'))

						if tweet_parsed[0] == "positive":
							variables.tweets_2014_score.append(1)
							variables.tweets_2014_positive += 1
						
						elif tweet_parsed[0] == "negative":
							variables.tweets_2014_score.append(-1)
							variables.tweets_2014_negative += 1
						
						elif tweet_parsed[0] == "neutral":
							variables.tweets_2014_score.append(0)
							variables.tweets_2014_neutral += 1

						variables.tweets_2014_score_svm.append(svm_class)
						variables.tweets_2014_score_naive.append(naive_class)
						variables.tweets_2014_score_MS.append(MS_class)
						variables.tweets_2014_score_LReg.append(LReg_class)
						variables.tweets_2014_score_S140.append(S140_class)
						variables.tweets_2014_score_RFor.append(rforest_class)
						variables.tweets_2014_score_SGD.append(SGD_class)
						variables.tweets_2014_score_ESumNoPG.append(ESumNoPG_class)

					elif tweet_parsed[1] == "SMS2013":
						variables.sms_2013.append(tweet_parsed[2].replace('right now', 'rightnow'))

						if tweet_parsed[0] == "positive":
							variables.sms_2013_score.append(1)
							variables.sms_2013_positive += 1
						
						elif tweet_parsed[0] == "negative":
							variables.sms_2013_score.append(-1)
							variables.sms_2013_negative += 1
						
						elif tweet_parsed[0] == "neutral":
							variables.sms_2013_score.append(0)
							variables.sms_2013_neutral += 1

						variables.sms_2013_score_svm.append(svm_class)
						variables.sms_2013_score_naive.append(naive_class)
						variables.sms_2013_score_MS.append(MS_class)
						variables.sms_2013_score_LReg.append(LReg_class)
						variables.sms_2013_score_S140.append(S140_class)
						variables.sms_2013_score_RFor.append(rforest_class)
						variables.sms_2013_score_SGD.append(SGD_class)
						variables.sms_2013_score_ESumNoPG.append(ESumNoPG_class)
					
					elif tweet_parsed[1] == "LiveJournal2014":
						variables.tweets_liveJournal2014.append(tweet_parsed[2].replace('right now', 'rightnow'))

						if tweet_parsed[0] == "positive":
							variables.tweets_liveJournal2014_score.append(1)
							variables.tweets_liveJournal2014_positive += 1
						
						elif tweet_parsed[0] == "negative":
							variables.tweets_liveJournal2014_score.append(-1)
							variables.tweets_liveJournal2014_negative += 1
						
						elif tweet_parsed[0] == "neutral":
							variables.tweets_liveJournal2014_score.append(0)
							variables.tweets_liveJournal2014_neutral += 1

						variables.tweets_liveJournal2014_score_svm.append(svm_class)
						variables.tweets_liveJournal2014_score_naive.append(naive_class)
						variables.tweets_liveJournal2014_score_MS.append(MS_class)
						variables.tweets_liveJournal2014_score_LReg.append(LReg_class)
						variables.tweets_liveJournal2014_score_S140.append(S140_class)
						variables.tweets_liveJournal2014_score_RFor.append(rforest_class)
						variables.tweets_liveJournal2014_score_SGD.append(SGD_class)
						variables.tweets_liveJournal2014_score_ESumNoPG.append(ESumNoPG_class)

					elif tweet_parsed[1] == "Twitter2014Sarcasm":
						variables.tweets_2014_sarcasm.append(tweet_parsed[2].replace('right now', 'rightnow'))
						
						if tweet_parsed[0] == "positive":
							variables.tweets_2014_sarcasm_score.append(1)
							variables.tweets_2014_sarcasm_positive += 1
						
						elif tweet_parsed[0] == "negative":
							variables.tweets_2014_sarcasm_score.append(-1)
							variables.tweets_2014_sarcasm_negative += 1
						
						elif tweet_parsed[0] == "neutral":
							variables.tweets_2014_sarcasm_score.append(0)                                                           
							variables.tweets_2014_sarcasm_neutral += 1
						
						variables.tweets_2014_sarcasm_score_svm.append(svm_class)
						variables.tweets_2014_sarcasm_score_naive.append(naive_class)
						variables.tweets_2014_sarcasm_score_MS.append(MS_class)
						variables.tweets_2014_sarcasm_score_LReg.append(LReg_class)
						variables.tweets_2014_sarcasm_score_S140.append(S140_class)
						variables.tweets_2014_sarcasm_score_RFor.append(rforest_class)
						variables.tweets_2014_sarcasm_score_SGD.append(SGD_class)
						variables.tweets_2014_sarcasm_score_ESumNoPG.append(ESumNoPG_class)
					
					tweets_loaded += 1                          
				
				except Exception as e:
					print("exception 1: " + str(e))
					continue

	end = time.time()
	print("  [test tweets loaded (" + str(tweets_loaded) + " tweets)][" + str(format(end - start, '.3g')) + " seconds]\n")


def saveSVMValuesOnFile():
	all_svm_values = []
	with open(variables.SEMEVAL_TEST_FILE, 'r') as f:
		for line in f:
			all_svm_values.append(str(line.split("\t")[3]))

	with open('SVM_test_results.txt', 'a') as f:
		for value in all_svm_values:
			f.write(str(value[1:-1].split()[0]) + "\t" + str(value[1:-1].split()[1]) + "\t" + str(value[1:-1].split()[2]) + "\n")



# Load STS train tweets
def loadTrainTweets_STS():
	start = time.time()
	print("\n[loading STS train tweets]")

	tweets_loaded = 0

	with codecs.open(variables.STS_TRAIN_FILE, "r", "latin-1") as inF:
		for line in inF:
			if tweets_loaded < variables.MAX_ANALYSIS_TWEETS:
				tweet_parsed = line.split("\t")
				try:
					if(tweet_parsed[0] == "positive"):
						if(variables.positive_tweets < variables.MAX_POSITIVES_TWEETS):
							variables.positive_tweets += 1         
							variables.tweets_sts.append(tweet_parsed[2])                   
							variables.tweets_sts_score.append(1)
							tweets_loaded += 1
					else:
						if(variables.negative_tweets < variables.MAX_NEGATIVES_TWEETS):
							variables.negative_tweets += 1
							variables.tweets_sts.append(tweet_parsed[2])
							variables.tweets_sts_score.append(-1)
							tweets_loaded += 1
				
				except:
					print("exception")
					continue

	end = time.time()
	print("  [train STS tweets loaded (" + str(tweets_loaded) + " tweets)][" + str(format(end - start, '.3g')) + " seconds]\n")


# Train file to textblob classifier (Naive Bayes)
def createTextBlobTrainFile():
	with open(variables.SEMEVAL_TRAIN_FILE, 'r') as inF:
		
		with open('trainMessages3.json', 'a') as f_w:
			f_w.write("[\n")
			for line in inF:
				tweet_parsed = line.split("\t")
				try:
					if(tweet_parsed[2] != "neutral"):
						if(tweet_parsed[2] == "positive"):
							#if(variables.positive_tweets < variables.MAX_POSITIVES_TWEETS):
							if(variables.positive_tweets < 1000):
								f_w.write('{"text": "' + str(tweet_parsed[3].strip().replace('"', '')) + '", "label": "pos"},\n')
								variables.positive_tweets += 1
						else:
							#if(variables.negative_tweets < variables.MAX_NEGATIVES_TWEETS):
							if(variables.negative_tweets < 1000):
								f_w.write('{"text": "' + str(tweet_parsed[3].strip().replace('"', '')) + '", "label": "neg"},\n')
								variables.negative_tweets += 1
					else:
						#if(variables.neutral_tweets < variables.MAX_NEUTRAL_TWEETS):
						if(variables.neutral_tweets < 1000):
							f_w.write('{"text": "' + str(tweet_parsed[3].strip().replace('"', '')) + '", "label": "neu"},\n')
							variables.neutral_tweets += 1
				except:
					print("exception")
					continue    
			f_w.write("\n]")


def trainNaiveBayesClassifier(file_train):
	from textblob.classifiers import NaiveBayesClassifier
	with open(file_train, 'r') as fp:
		cl = NaiveBayesClassifier(fp, format="json")

		import pickle

		save_classifier = open("naivebayes.classifier","wb") # the pickle file isn't avaible on github because it's too big (>1GB)
		pickle.dump(cl, save_classifier)
		save_classifier.close()


def loadNaiveBayesClassifier():
	start = time.time()
	import pickle

	print("[loading classifier]")
	
	classifier_f = open("naivebayes.classifier", "rb")
	classifier = pickle.load(classifier_f)
	classifier_f.close()
	
	end = time.time()
	print("  [classifier loaded][" + str(format(end - start, '.3g')) + " seconds]\n")
	
	return classifier


def saveNaiveBayesValues():
	start = time.time()
	print("\n[saving naive bayes values]")

	classifier = loadNaiveBayesClassifier()

	pos_prob, neg_prob, neu_prob = 0, 0, 0

	with open('naive_train_values.txt', 'a') as f_w:
		with open(variables.SEMEVAL_TRAIN_FILE, 'r') as inF:
			for line in inF:
				pos_prob = classifier.prob_classify(line.split("\t")[3].strip()).prob("pos")
				neg_prob = classifier.prob_classify(line.split("\t")[3].strip()).prob("neg")
				neu_prob = classifier.prob_classify(line.split("\t")[3].strip()).prob("neu")

				f_w.write(str(pos_prob) + '\t' + str(neg_prob) + '\t' + str(neu_prob) + '\n')


	end = time.time()
	print("  [values included][" + str(format(end - start, '.3g')) + " seconds]\n")


def includeNaiveBayesValuesOnTestFile():
	start = time.time()
	print("\n[including NaiveBayes values on test file]")

	classifier = loadNaiveBayesClassifier()

	with open('testWithSVM_Naive.txt', 'a') as f_w:
		with open(variables.SEMEVAL_TEST_FILE, 'r') as inF:
			for line in inF:
				t_class = ""
				if (classifier.classify(line.split("\t")[2].strip()) == "pos"):
					t_class = "positive"
				elif (classifier.classify(line.split("\t")[2].strip()) == "neg"):
					t_class = "negative"
				elif (classifier.classify(line.split("\t")[2].strip()) == "neu"):
					t_class = "neutral"

				f_w.write(str(line.split("\t")[0]) + "\t" + line.split("\t")[1] + "\t" + line.split("\t")[2] + "\t" + str(line.split("\t")[3]).strip() + "\t" + t_class + "\n")


	end = time.time()
	print("  [values included][" + str(format(end - start, '.3g')) + " seconds]\n")


def saveLogisticRegressionPredictionsOnTestFile():
	start = time.time()
	print("\n[including LogisticRegression values on test file]")
	
	from sklearn.feature_extraction.text import CountVectorizer
	from sklearn.model_selection import train_test_split
	from sklearn.linear_model import LogisticRegression

	data, data_labels = [], []
	maximum = 0

	with open("datasets/train/positiveTweets.txt") as f:
		for i in f: 
			maximum += 1
			if maximum <= 1000:
				data.append(i) 
				data_labels.append('positive')

	maximum = 0
	with open("datasets/train/negativeTweets.txt") as f:
		for i in f: 
			maximum += 1
			if maximum <= 1000:
				data.append(i)
				data_labels.append('negative')

	maximum = 0
	with open("datasets/train/neutralTweets.txt") as f:
		for i in f: 
			maximum += 1
			if maximum <= 1000:
				data.append(i)
				data_labels.append('neutral')        


	vectorizer = CountVectorizer(
		analyzer = 'word',
		stop_words = 'english', 
		ngram_range = (1, 2), 
		lowercase = False,
	)

	#vectorizer = FeatureUnion([
	#    ('cv', CountVectorizer(analyzer = 'word', ngram_range = (1,2), lowercase = False)),
	#    ('av_len', AverageLenVectizer(...))
	#])

	features = vectorizer.fit_transform(
		data
	)

	features_nd = features.toarray()

	X_train, X_test, y_train, y_test  = train_test_split(
			features_nd, 
			data_labels,
			train_size=0.80, 
			random_state=1234)

	from sklearn.linear_model import LogisticRegression
	log_model = LogisticRegression()
	
	print(" [training model]")
	log_model = log_model.fit(X=X_train, y=y_train)
	print(" [training model completed]")
	
	test_tweets = []
	all_lines_test_file = []

	with open(variables.SEMEVAL_TEST_FILE, 'r') as t_f:
		for line in t_f:
			test_tweets.append(line.split("\t")[2].strip())
			all_lines_test_file.append(line)
	
	print("[begin tweets predictions]")
	results = []
	results = log_model.predict(vectorizer.transform(test_tweets))
	print("[end tweets predictions]")

	print("[saving results on file]")
	with open('test_SVM_MS_LReg.txt', 'a') as f_s:
		for index, prediction in enumerate(results):
			f_s.write(all_lines_test_file[index].strip() + "\t" + str(prediction) + "\n")
	print("[result saved]")

	end = time.time()
	print("  [values included][" + str(format(end - start, '.3g')) + " seconds]\n")
	
	#import pickle
	#save_classifier = open("logRegression.classifier","wb") # the pickle file isn't avaible on github because it's too big (>1GB)
	#pickle.dump(log_model, save_classifier)
	#save_classifier.close()


def includeMicrosoftClassifierValuesOnTestFile():
	start = time.time()
	print("\n[including Microsoft Classifier values on test file]")

	allMicrosoftValues = []

	with open('microsoftScoresAll.txt', 'r') as mF: # Get all microsoft values on file and save this on test file
		for line in mF:
			allMicrosoftValues.append(line.strip())
	
	index_ = 0
	with open('testWithSVM_Naive_MS.txt', 'a') as f_w:
		with open(variables.SEMEVAL_TEST_FILE, 'r') as inF:
			for line in inF:
				f_w.write(str(line.split("\t")[0]) + "\t" + line.split("\t")[1] + "\t" + line.split("\t")[2] + "\t" + str(line.split("\t")[3]).strip() + "\t" + str(line.split("\t")[4]).strip() + "\t" + str(allMicrosoftValues[index_]) + "\n")
				index_ += 1

	end = time.time()
	print("  [values included][" + str(format(end - start, '.3g')) + " seconds]\n")


def createFileForMicrosoftClassifier():
	id_tweet = 0
	with open(variables.SEMEVAL_TEST_FILE, 'r') as inF:
		with open('fileMicrosoftClassifier.json', 'a') as f_w:
			f_w.write('{"documents": [\n')
			for line in inF:
				id_tweet += 1
				tweet_parsed = line.split("\t")
				f_w.write('{"id": "' + str(id_tweet) + '", "language": "en", "text": "' + str(tweet_parsed[2].strip().replace('"', '').replace("\\u2018", "").replace("\\u2019", "").replace("\\u002c", "")) + '"},\n')  
			f_w.write("]}")


def classifyUsingMicrosoftClassifier():
	import variables2
	# chuck the test files because the api limit (the test file isn't avaiable on github because it's easy to create then using the createFileForMicrosoftClassifier function)
	testMicrosoftClassifier(variables2.microsoft_document_1000)
	testMicrosoftClassifier(variables2.microsoft_document_2000)
	testMicrosoftClassifier(variables2.microsoft_document_3000)
	testMicrosoftClassifier(variables2.microsoft_document_4000)
	testMicrosoftClassifier(variables2.microsoft_document_5000)
	testMicrosoftClassifier(variables2.microsoft_document_6000)
	testMicrosoftClassifier(variables2.microsoft_document_7000)
	testMicrosoftClassifier(variables2.microsoft_document_8000)
	testMicrosoftClassifier(variables2.microsoft_document_9000)


def testMicrosoftClassifier(documents):
	subscription_key = "ae40849f41e84c5b85bde44f5e54cbbd"
	assert subscription_key

	import requests
	from pprint import pprint

	text_analytics_base_url = "https://westcentralus.api.cognitive.microsoft.com/text/analytics/v2.0/"

	sentiment_api_url = text_analytics_base_url + "sentiment"

	# Parameters format
	#documents = {'documents' : [
	#    {'id': '1', 'language': 'en', 'text': 'it sounds really bad but it just couldnt have been worse timing .'},
	#    {'id': '2', 'language': 'en', 'text': 'we confirm when we meeting then we arrange time again.'},
	#]}

	headers   = {"Ocp-Apim-Subscription-Key": subscription_key}
	response  = requests.post(sentiment_api_url, headers=headers, json=documents)
	sentiments = response.json()
	#print(sentiments['documents'])
	with open('microsoftScoresAll.txt', 'a') as f_w:
		for lines in sentiments['documents']:
			f_w.write(str(lines['score']) + "\n")
	#pprint(sentiments)


def createFileForS140Classifier():
	with open(variables.SEMEVAL_TEST_FILE, 'r') as inF:
		with open('fileS140.txt', 'a') as f_w:
			f_w.write('{"data": [\n')
			for line in inF:
				tweet_parsed = line.split("\t")
				f_w.write('{"text": "' + str(tweet_parsed[2].strip().replace('"', '').replace("\\u2018", "").replace("\\u2019", "").replace("\\u002c", "")) + '"},\n')  
			f_w.write("]}")


def normalize_s140(polarity_int):
	# from Sentiment140 doc page (http://help.sentiment140.com/api#TOC-Developer-Documentation)
	# The polarity values are: 0: negative 2: neutral 4: positive
	if polarity_int == 0:
		return "negative"
	elif polarity_int == 2:
		return "neutral"
	elif polarity_int == 4:
		return "positive"


def saveSentiment140PredictionsOnTestFile():
	import urllib.request
	import json
	import var2

	url = 'http://www.sentiment140.com/api/bulkClassifyJson'
	# Example API parameter
	# values = {'data': [{'text': 'I love Titanic.'}, {'text': 'I hate Titanic.'}]} 

	data = json.dumps(var2.all_messages)
	response = urllib.request.urlopen(url, data=data.encode("utf-8"))
	page = response.read()
	page_loaded = json.loads(page.decode("utf-8"))

	results = [] 

	for result in page_loaded["data"]:
		results.append(normalize_s140(result['polarity']))
		#print(normalize_s140(result['polarity']))


	all_lines_test_file = []
	with open(variables.SEMEVAL_TEST_FILE, 'r') as t_f:
		for line in t_f:
			all_lines_test_file.append(line)

	with open('test_SVM_MS_LReg_s140.txt', 'a') as f_s:
		for index, polarities in enumerate(results):
			f_s.write(all_lines_test_file[index].strip() + "\t" + str(polarities) + "\n")



def saveAylienPredictionsOnTestFile():
	import var2
	from aylienapiclient import textapi
	client = textapi.Client("f60113d3", "c05ff1ac96609b50c4620564d6b99f61")

	all_lines_test_file = []
	texts = []
	#texts.append({'text': ';-) please screen or delete this comment .'})
	#texts.append({'text': 'I hate you'})
	#texts.append({'text': 'Today is sunny'})

	
	with open(variables.SEMEVAL_TEST_FILE, 'r') as t_f:
		for line in t_f:
			all_lines_test_file.append(line.strip())
			texts.append("{'text': '" + str(clean_tweet(line.split("\t")[2])) + "'}")
	
	i = 0
	for text in texts:
		i += 1
		sentiment = client.Sentiment(text)
		print("line " + str(i) + str(sentiment["polarity"]))
	
	with open('test_SVM_MS_LReg_s140_Ayl.txt', 'a') as f_s:
		for index, polarities in enumerate(results):
			f_s.write(all_lines_test_file[index].strip() + "\t" + str(polarities) + "\n")



def getNeutralRangesFromTestFile():
	import tkinter as tk
	from tkinter import filedialog

	inferior = []
	superior = []

	root = tk.Tk()
	root.withdraw()

	file_path = filedialog.askopenfilename()

	next_line = False

	with open(file_path, 'r') as f:
		for line in f:
			if line.startswith("# [neutral ranges]"):
				next_line = True
				continue
			if next_line:
				for i in line[4:-2].replace("'", "").split("],"):
					a = i.strip().replace("[", "").replace("]", "").split(", ")
					#print("len: " + str(len(a)))
					#print("neg " + a[0] + " pos " + a[1])
					inferior.append(a[0])
					superior.append(a[1])

					#print("\n")	
				#print("--\n")
				next_line = False

	import seaborn as sns
	import matplotlib.pyplot as plt

	sns.set(style="whitegrid")
	all_outs = []

	#labels = ['Twitter2013', 'Twitter2014', 'Sarcasm', 'SMS', 'LiveJournal', 'Todas']

	all_outs.append(inferior)
	all_outs.append(superior)

	ax = sns.boxplot(data=all_outs, orient="v", palette="pastel", showmeans=True)
	#ax = sns.boxplot(data=all_outs, orient="v", showmeans=True)
	
	#plt.ylim(-6, 6)


	plt.ion()
	plt.show()

	print("avg inferior: " + str(sum(list(map(float, inferior))) / len(inferior)))
	print("avg superior: " + str(sum(list(map(float, superior))) / len(superior)))
	
	print("min inferior: " + str(min(list(map(float, inferior)))))
	print("min superior: " + str(min(list(map(float, superior)))))

	print("max inferior: " + str(max(list(map(float, inferior)))))
	print("max superior: " + str(max(list(map(float, superior)))))

	input("press to continue")


def getWeightsMeanFromTestFile():
	import tkinter as tk
	from tkinter import filedialog

	means = []

	root = tk.Tk()
	root.withdraw()

	file_path = filedialog.askopenfilename()

	#next_line = False

	with open(file_path, 'r') as f:
		for line in f:
			if line.startswith("w") and line.split(" ")[1].startswith("values"):
				line_means = line.split("\t")[1][1:-2].split(", ")
				means.append(list(map(float, line_means)))
				#print(line.split("\t")[1][1:-2])
				#print(str(line.split("\t")[1][1:-2].split(", ")))


			#if next_line:
			#	for i in line[4:-2].replace("'", "").split("],"):
			#		a = i.strip().replace("[", "").replace("]", "").split(", ")
					#print("len: " + str(len(a)))
					#print("neg " + a[0] + " pos " + a[1])
			#		inferior.append(a[0])
			#		superior.append(a[1])

					#print("\n")	
				#print("--\n")
			#	next_line = False

	#geral_means = []
	#for i in means:
	#	m = sum(i) / len(i)
	#	geral_means.append(m)  

	import seaborn as sns
	import matplotlib.pyplot as plt

	sns.set(style="whitegrid")
	all_outs = []

	#labels = ['Twitter2013', 'Twitter2014', 'Sarcasm', 'SMS', 'LiveJournal', 'Todas']

	#all_outs.append(inferior)
	#all_outs.append(superior)

	#ax = sns.boxplot(data=geral_means, orient="v", showmeans=True, color=".90")
	#ax = sns.catplot(jitter=False, data=means);
	#ax = sns.catplot(palette="ch:.25", data=geral_means);

	ax = sns.swarmplot(data=means, orient="v", color=".25", jitter=False)

	#ax = sns.relplot(data=means, orient="v", palette="pastel")
	#ax = sns.boxplot(data=all_outs, orient="v", showmeans=True)
	
	plt.ylim(0, 2)


	plt.ion()
	plt.show()

	#print("avg inferior: " + str(sum(list(map(float, inferior))) / len(inferior)))
	#print("avg superior: " + str(sum(list(map(float, superior))) / len(superior)))
	
	#print("min inferior: " + str(min(list(map(float, inferior)))))
	#print("min superior: " + str(min(list(map(float, superior)))))

	#print("max inferior: " + str(max(list(map(float, inferior)))))
	#print("max superior: " + str(max(list(map(float, superior)))))

	input("press to continue")

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
					variables.tweets_sts_test.append(tweet_parsed[2])
					if tweet_parsed[0] == "positive":
						variables.tweets_sts_score_test.append(1)
						variables.tweets_sts_positive += 1

					elif tweet_parsed[0] == "negative":
						variables.tweets_sts_score_test.append(-1)
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


def getOnlyPOSClass(phrase, pos_class, return_type="array"):
	start = time.time()
	print("\n[start getPOSClass]")
	import nltk
	from nltk.tokenize import TweetTokenizer
	
	tknzr  = TweetTokenizer()
	tokens = tknzr.tokenize(phrase)
	tags   = nltk.pos_tag(tokens)
	words  =  [word for word, pos in tags if (pos == pos_class)]

	end = time.time()
	print("  [getPOSClasse finished][" + str(format(end - start, '.3g')) + " seconds]\n") 

	if return_type == "phrase":
		return ' '.join(words).lower()
	return words
	

def getPOSTag(message):
	import nltk
	from nltk.tokenize import TweetTokenizer
	
	tknzr = TweetTokenizer()
	tokens = tknzr.tokenize(message)

	#tokens = nltk.word_tokenize(message)
	tagged = nltk.pos_tag(tokens)

	return tagged


def saveWordsTokenized(module):
	if module == "train":
		file_words     = variables.TRAIN_WORDS_SPELLCHECK
		file_words_POS = variables.TRAIN_WORDS_POS_TAGGED_W
	elif module == "test":
		file_words     = variables.TEST_WORDS_SPELLCHECK
		file_words_POS = variables.TEST_WORDS_POS_TAGGED_W

	with open(file_words, 'r') as f_words:
		for line in f_words:
			with open(file_words_POS, 'a') as f_words_POS:
				w_class = str(getPOSTag(line)).split(",")[1].strip()[1:-3]
				if w_class in variables.USE_POS_CLASSES:
					f_words_POS.write(line.strip() + "\n")
					#f_words_POS.write(line.strip() + "\t" + w_class + "\n")


def saveTweetsbyClass(tweet_class, origin_file, destin_file):
	with open(destin_file, 'a') as d_f:
		with open(origin_file, 'r') as o_f:
			for line in o_f:
				if line.split("\t")[2].strip() == tweet_class:
					d_f.write(line.split("\t")[3].strip() + "\n")


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


def phraseLength(phrase):
	return len(phrase.strip())


def wordCount(phrase):
	return len(phrase.split())


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

	#if variables.calling_by_ag_file:
	#    return 0

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

		# NRC
		if(variables.use_dic_nrc and variables.dic_nrc_loaded):
			if word in variables.dic_positive_nrc:
				
				#print("word " + word + " on nrc with the value " + str(variables.dic_positive_nrc[word]))

				if invert:
					total_sum -= variables.dic_positive_nrc[word]
					#total_sum -= 1 * w7
				elif booster:
					total_sum += 2 * variables.dic_positive_nrc[word]
					#total_sum += 2 * w7
				elif boosterAndInverter:
					#total_sum -= 2 * w7 
					total_sum -= 2 * variables.dic_positive_nrc[word]
				else:
					#total_sum += variables.dic_positive_semeval2015[word]
					total_sum += variables.dic_positive_nrc[word]
			
				dic_quantity += 1

			elif word in variables.dic_negative_nrc:

				#print("word " + word + " on nrc with the value " + str(variables.dic_negative_nrc[word]))

				if invert:
					total_sum -= variables.dic_negative_nrc[word]
					#total_sum += 1 * w7
				elif booster:
					total_sum += 2* variables.dic_negative_nrc[word]
					#total_sum -= 2 * w7
				elif boosterAndInverter:
					total_sum -= 2 * variables.dic_negative_nrc[word]
					#total_sum += 2 * w7                      
				else:
					total_sum += variables.dic_negative_nrc[word]
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



def polaritySumAVGUsingWeights(phrase, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10=0, w11=0):
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

	if variables.calling_by_ag_file:
		w1  = variables.ag_w1
		w2  = variables.ag_w2
		w3  = variables.ag_w3
		w4  = variables.ag_w4
		w5  = variables.ag_w5
		w6  = variables.ag_w6
		w7  = variables.ag_w7
		w8  = variables.ag_w8
		w9  = variables.ag_w9
		w10 = variables.ag_w10
		w11 = variables.ag_w11

	if variables.calling_by_test_file:
		#TEST
		#w = [1.0, 0.0, 1.0, 1.1009543916315279, 0.0, 0.0, 0.0, -0.8059644858852133, 1.303579956878811, -1.6076203174025039, 0.0]
		#w1 = w[0]
		#w2 = w[1]
		#w3 = w[2]
		#w4 = w[3]
		#w5 = w[4]
		#w6 = w[5]
		#w7 = w[6]
		#w8 = w[7]
		#w9 = w[8]

		#variables.neutral_inferior_range = w[9]
		#variables.neutral_superior_range = w[10]
		
		variables.w1.append(w1)
		variables.w2.append(w2)
		variables.w3.append(w3)
		variables.w4.append(w4)
		variables.w5.append(w5)
		variables.w6.append(w6)
		variables.w7.append(w7)
		variables.w8.append(w8)
		variables.w9.append(w9)
		variables.w10.append(w10)
		variables.w11.append(w11)

		variables.neutral_values.append("[" + str(variables.neutral_inferior_range) + ", " + str(variables.neutral_superior_range) + "]")

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
		
		elif (index > 0 and words[index-1] == "insidenoteboosteruppercase") or (index < len(words) - 1 and words[index+1] == "insidenoteboosteruppercase" and (words[index-1] != "insidenoteboosteruppercase" or index == 0)):
			#print("booster")
			booster = True      

		hashtag = False

		# TEST - CHANGE THIS
		if word.startswith("#"):
			word = word[1:]
			booster = True
			if word in variables.dic_positive_hashtags:
				total_sum += 2
				hashtag = True
			elif word in variables.dic_negative_hashtags:
				total_sum -= 2
				hashtag = True
			
			if(hashtag):
				hashtag = False
				dic_quantity = 0
				total_sum    = 0
				total_weight = 0

				index += 1 # word of phrase
				invert = False
				booster = False
				boosterAndInverter = False
				continue
		# TEST - CHANGE THIS

		# LIU pos/neg words
		if(variables.use_dic_liu and variables.dic_liu_loaded and w1 != 0):
			if word in variables.dic_positive_words:# or word in variables.dic_positive_hashtags:
				if invert:
					total_sum -= 1 * w1
				elif booster:
					total_sum += 2 * w1
				elif boosterAndInverter:
					total_sum -= 2 * w1
				else: 
					total_sum += 1 * w1

				print("find word " + word + " on liu positive")
				dic_quantity += 1
				total_weight += w1

			elif word in variables.dic_negative_words:# or word in variables.dic_negative_hashtags:
				if invert:
					total_sum += 1 * w1
				elif booster:
					total_sum -= 2 * w1
				elif boosterAndInverter:
					total_sum += 2 * w1
				else:
					total_sum -= 1 * w1

				print("find word " + word + " on liu negative")
				dic_quantity += 1
				total_weight += w1

		# SENTIWORDNET
		if(variables.use_dic_sentiwordnet and variables.dic_sentiwordnet_loaded and w2 != 0):
			if word in variables.dic_positive_sentiwordnet:
				
				print("word " + word + " on sentiwordnet with the value " + str(variables.dic_positive_sentiwordnet[word]))

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
				
				print("word " + word + " on sentiwordnet with the value " + str(variables.dic_negative_sentiwordnet[word]))

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
				
				print("word " + word + " on affin with the value " + str(variables.dic_positive_affin[word]))
				
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
				
				print("word " + word + " on affin with the value " + str(variables.dic_negative_affin[word]))

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

				print("word " + word + " on vader with the value " + str(variables.dic_positive_vader[word]))

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
				
				print("word " + word + " on vader with the value " + str(variables.dic_negative_vader[word]))

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
				
				print("word " + word + " on slang with the value " + str(variables.dic_positive_slang[word]))
				
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

				print("word " + word + " on slang with the value " + str(variables.dic_negative_slang[word]))
				
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
				
				print("word " + word + " on effect with the value " + str(variables.dic_positive_effect[word]))

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
				
				print("word " + word + " on effect with the value " + str(variables.dic_negative_effect[word]))

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
				
				print("word " + word + " on semeval2015 with the value " + str(variables.dic_positive_semeval2015[word]))

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

				print("word " + word + " on semeval2015 with the value " + str(variables.dic_negative_semeval2015[word]))

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

		# NRC
		if(variables.use_dic_nrc and variables.dic_nrc_loaded and w8 != 0):
			if word in variables.dic_positive_nrc:
				
				print("word " + word + " on nrc with the value " + str(variables.dic_positive_nrc[word]))

				if invert:
					total_sum -= variables.dic_positive_nrc[word] * w8
					#total_sum -= 1 * w7
				elif booster:
					total_sum += 2 * variables.dic_positive_nrc[word] * w8
					#total_sum += 2 * w7
				elif boosterAndInverter:
					#total_sum -= 2 * w7 
					total_sum -= 2 * variables.dic_positive_nrc[word] * w8                    
				else:
					#total_sum += variables.dic_positive_nrc[word]
					total_sum += variables.dic_positive_nrc[word] * w8
			
				dic_quantity += 1
				total_weight += w8

			elif word in variables.dic_negative_nrc:

				print("word " + word + " on nrc with the value " + str(variables.dic_negative_nrc[word]))

				if invert:
					total_sum -= variables.dic_negative_nrc[word] * w8
				elif booster:
					total_sum += 2* variables.dic_negative_nrc[word] * w8
				elif boosterAndInverter:
					total_sum -= 2 * variables.dic_negative_nrc[word] * w8                      
				else:
					total_sum += variables.dic_negative_nrc[word] * w8
						
				dic_quantity += 1
				total_weight += w8 

		# GENERAL INQUIRER
		if(variables.use_dic_gi and variables.dic_gi_loaded and w9!= 0):
			if word in variables.dic_positive_gi:
				
				print("word " + word + " on gi with the value " + str(variables.dic_positive_gi[word]))

				if invert:
					total_sum -= variables.dic_positive_gi[word] * w9
				elif booster:
					total_sum += 2 * variables.dic_positive_gi[word] * w9
				elif boosterAndInverter: 
					total_sum -= 2 * variables.dic_positive_gi[word] * w9                    
				else:
					total_sum += variables.dic_positive_gi[word] * w9
			
				dic_quantity += 1
				total_weight += w9

			elif word in variables.dic_negative_gi:

				print("word " + word + " on gi with the value " + str(variables.dic_negative_gi[word]))

				if invert:
					total_sum -= variables.dic_negative_gi[word] * w9
				elif booster:
					total_sum += 2* variables.dic_negative_gi[word] * w9
				elif boosterAndInverter:
					total_sum -= 2 * variables.dic_negative_gi[word] * w9                      
				else:
					total_sum += variables.dic_negative_gi[word] * w9
						
				dic_quantity += 1
				total_weight += w9  

		# S140
		if(variables.use_dic_s140 and variables.dic_s140_loaded and w10!= 0):
			if word in variables.dic_positive_s140:
				
				print("word " + word + " on s140 with the value " + str(variables.dic_positive_s140[word]))

				if invert:
					total_sum -= variables.dic_positive_s140[word] * w10
				elif booster:
					total_sum += 2 * variables.dic_positive_s140[word] * w10
				elif boosterAndInverter: 
					total_sum -= 2 * variables.dic_positive_s140[word] * w10                    
				else:
					total_sum += variables.dic_positive_s140[word] * w10
			
				dic_quantity += 1
				total_weight += w10

			elif word in variables.dic_negative_s140:

				print("word " + word + " on s140 with the value " + str(variables.dic_negative_s140[word]))

				if invert:
					total_sum -= variables.dic_negative_s140[word] * w10
				elif booster:
					total_sum += 2* variables.dic_negative_s140[word] * w10
				elif boosterAndInverter:
					total_sum -= 2 * variables.dic_negative_s140[word] * w10                      
				else:
					total_sum += variables.dic_negative_s140[word] * w10
						
				dic_quantity += 1
				total_weight += w10

		# MPQA
		if(variables.use_dic_s140 and variables.dic_mpqa_loaded and w11!= 0):
			if word in variables.dic_positive_mpqa:
				
				print("word " + word + " on mpqa with the value " + str(variables.dic_positive_mpqa[word]))

				if invert:
					total_sum -= variables.dic_positive_mpqa[word] * w11
				elif booster:
					total_sum += 2 * variables.dic_positive_mpqa[word] * w11
				elif boosterAndInverter:
					total_sum -= 2 * variables.dic_positive_mpqa[word] * w11                    
				else:
					total_sum += variables.dic_positive_mpqa[word] * w11
			
				dic_quantity += 1
				total_weight += w11

			elif word in variables.dic_negative_mpqa:

				print("word " + word + " on mpqa with the value " + str(variables.dic_negative_mpqa[word]))

				if invert:
					total_sum -= variables.dic_negative_mpqa[word] * w11
				elif booster:
					total_sum += 2* variables.dic_negative_mpqa[word] * w11
				elif boosterAndInverter:
					total_sum -= 2 * variables.dic_negative_mpqa[word] * w11
				else:
					total_sum += variables.dic_negative_mpqa[word] * w11
						
				dic_quantity += 1
				total_weight += w11                                                                                                          


		if (dic_quantity > 1) and (total_weight != 0):
			total_sum_return += round(total_sum/total_weight, 4)
		elif (dic_quantity == 1):
			total_sum_return += total_sum        

		dic_quantity = 0
		total_sum    = 0
		total_weight = 0

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
		if word.isupper() and len(word) > 1:
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
		#print(word.strip().replace("'",""))
		if word.strip().replace("'","") in variables.dic_positive_emoticons:
			total_sum += 1               

	#print(str(total_sum))

	return total_sum


def negativeEmoticons(phrase):
	words = phrase.split()

	total_sum = 0

	for word in words:
		if word.strip().replace("'","") in variables.dic_negative_emoticons:
			total_sum += 1               

	return total_sum


# Positive Hashtags
def positiveHashtags(phrase):
	total = 0
	if "#" in phrase:
		hashtags = re.findall(r"#(\w+)", phrase)

		for hashtag in hashtags:
			#print(hashtag)
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
		if (word.replace("'","") in variables.dic_negative_emoticons) or (word.replace("'","") in variables.dic_positive_emoticons):
			return True

	return False


# logic operators
# Define a new if-then-else function
def if_then_else(input_, output1, output2):
	if input_: return output1
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


def clean_tweet(tweet):
	return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())


def floatToStr_polarity_value(numerical_polarity_value, neutral_inferior_range, neutral_superior_range):
	if float(numerical_polarity_value) > float(neutral_superior_range):
		return "positive"
	elif float(numerical_polarity_value) < float(neutral_inferior_range):
		return "negative"
	else:
		return "neutral"


draw = 0

def get_best_evaluation(classifiers_evaluations, type="majority"):
	negatives, neutrals, positives = 0, 0, 0

	for value in classifiers_evaluations:
		if value == "positive":
			positives += 1
		elif value == "negative":
			negatives += 1
		elif value == "neutral":
			neutrals += 1

	# test - remove this
	v = ""
	for value in classifiers_evaluations:
		v += value + ", "

	#print(str(v).strip() + " Positives: " + str(positives) + ", negatives: " + str(negatives) + ", neutrals: " + str(neutrals))
	# test - remove this

	if positives > negatives and positives > neutrals:
		return "positive"
	elif negatives > positives and negatives > neutrals:
		return "negative"
	elif neutrals > positives and neutrals > negatives:
		return "neutral"
	else: # there is no majority polarity
		#print("[DRAW] Positives: " + str(positives) + ", negatives: " + str(negatives) + ", neutrals: " + str(neutrals))
		#print(str(classifiers_evaluations))
		
		sufix = ""

		if positives == negatives and positives == neutrals:
			#print("   [SUPER DRAW BETWEEN " + str(len(classifiers_evaluations)) + " CLASSIFIERS]")
			sufix = "_all"
		elif negatives < positives and negatives < neutrals:
			#print("   [NEGATIVE CLASS MINORITY]")
			sufix = "_negative"
		elif positives < negatives and positives < neutrals:
			#print("   [POSITIVE CLASS MINORITY]")
			sufix = "_positive"
		elif neutrals < negatives and neutrals < positives:
			#print("   [NEUTRAL CLASS MINORITY]")
			sufix = "_neutral"
		return "DRAW" + sufix
		#if positives > neutrals or positives == neutrals:
			#print(str(v).strip() + " Positives: " + str(positives) + ", negatives: " + str(negatives) + ", neutrals: " + str(neutrals) + " I'll return positive")
			#print("Positives: " + str(positives) + ", negatives: " + str(negatives) + ", neutrals: " + str(neutrals) + " I'll return positive")
		#    return "DRAW"
		#elif negatives > neutrals or negatives == neutrals:
			#print(str(v).strip() + " Positives: " + str(positives) + ", negatives: " + str(negatives) + ", neutrals: " + str(neutrals) + " I'll return negative")
			#print("Positives: " + str(positives) + ", negatives: " + str(negatives) + ", neutrals: " + str(neutrals) + " I'll return negative")
		#    return "DRAW"


#from aylienapiclient import textapi
#client = textapi.Client("f60113d3", "c05ff1ac96609b50c4620564d6b99f61")

# Evaluate the test messages using the model
# http://text-analytics101.rxnlp.com/2014/10/computing-precision-and-recall-for.html
def evaluateMessages(base, model, model_ensemble=False):
	global model_results_to_count_occurrences
	print("[starting evaluation of " + base + " messages]")
	if model_ensemble == True:
		print("   [using ensemble of " + str(len(model)) + " models]")
	from textblob import TextBlob

	# test
	global neutral_url_qtty
	global neutral_url_correct_pred
	count_has_date = 0
	# test

	neutral_url_qtty, neutral_url_correct_pred = 0, 0

	neutral_because_url = False
	# Testing

	# parameters to calc the metrics
	true_positive, true_negative, false_positive, false_negative, true_neutral, false_neutral  = 0, 0, 0, 0, 0, 0

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
	precision_positive, precision_negative, precision_neutral, precision_avg = 0, 0, 0, 0
	recall_positive,    recall_negative,    recall_neutral,    recall_avg    = 0, 0, 0, 0
	f1_positive,        f1_negative,        f1_neutral,        f1_avg        = 0, 0, 0, 0
	f1_positive_negative_avg = 0


	false_neutral_log, false_negative_log  = 0, 0

	message, model_analysis = "", ""
	result = 0

	messages             = []
	messages_score       = []
	messages_score_svm   = []
	messages_score_naive = []
	messages_score_MS    = []
	messages_score_LReg  = []
	messages_score_S140  = []
	messages_score_RFor  = []
	messages_score_SGD   = []
	messages_positive, messages_negative, messages_neutral = 0, 0, 0
	
	total_positive = variables.tweets_2013_positive + variables.tweets_2014_positive + variables.tweets_liveJournal2014_positive + variables.tweets_2014_sarcasm_positive + variables.sms_2013_positive
	total_negative = variables.tweets_2013_negative + variables.tweets_2014_negative + variables.tweets_liveJournal2014_negative + variables.tweets_2014_sarcasm_negative + variables.sms_2013_negative
	total_neutral  = variables.tweets_2013_neutral  + variables.tweets_2014_neutral  + variables.tweets_liveJournal2014_neutral  + variables.tweets_2014_sarcasm_neutral  + variables.sms_2013_neutral 

	if len(variables.tweets_2013) == 0:
		loadTestTweets()
		#loadTestTweetsSemeval2017()
		#loadTestTweets_smuk()

	if base == "tweets2013":
		messages             = variables.tweets_2013
		messages_score       = variables.tweets_2013_score
		messages_score_svm   = variables.tweets_2013_score_svm
		messages_score_naive = variables.tweets_2013_score_naive
		messages_score_MS    = variables.tweets_2013_score_MS
		messages_score_LReg  = variables.tweets_2013_score_LReg
		messages_score_S140  = variables.tweets_2013_score_S140
		messages_score_RFor  = variables.tweets_2013_score_RFor
		messages_score_SGD   = variables.tweets_2013_score_SGD
		messages_score_ESumNoPG = variables.tweets_2013_score_ESumNoPG
		messages_positive    = variables.tweets_2013_positive
		messages_negative    = variables.tweets_2013_negative
		messages_neutral     = variables.tweets_2013_neutral

		variables.t2k13_outputs = []
	elif base == "tweets2014":
		messages             = variables.tweets_2014
		messages_score       = variables.tweets_2014_score
		messages_score_svm   = variables.tweets_2014_score_svm
		messages_score_naive = variables.tweets_2014_score_naive
		messages_score_MS    = variables.tweets_2014_score_MS
		messages_score_LReg  = variables.tweets_2014_score_LReg
		messages_score_S140  = variables.tweets_2014_score_S140
		messages_score_RFor  = variables.tweets_2014_score_RFor
		messages_score_SGD   = variables.tweets_2014_score_SGD
		messages_score_ESumNoPG = variables.tweets_2014_score_ESumNoPG
		messages_positive    = variables.tweets_2014_positive
		messages_negative    = variables.tweets_2014_negative
		messages_neutral     = variables.tweets_2014_neutral

		variables.t2k14_outputs = []
	elif base == "livejournal":
		messages             = variables.tweets_liveJournal2014
		messages_score       = variables.tweets_liveJournal2014_score
		messages_score_svm   = variables.tweets_liveJournal2014_score_svm
		messages_score_naive = variables.tweets_liveJournal2014_score_naive
		messages_score_MS    = variables.tweets_liveJournal2014_score_MS
		messages_score_LReg  = variables.tweets_liveJournal2014_score_LReg
		messages_score_S140  = variables.tweets_liveJournal2014_score_S140
		messages_score_RFor  = variables.tweets_liveJournal2014_score_RFor
		messages_score_SGD   = variables.tweets_liveJournal2014_score_SGD
		messages_score_ESumNoPG = variables.tweets_liveJournal2014_score_ESumNoPG
		messages_positive    = variables.tweets_liveJournal2014_positive
		messages_negative    = variables.tweets_liveJournal2014_negative
		messages_neutral     = variables.tweets_liveJournal2014_neutral

		variables.lj_outputs = []
	elif base == "sarcasm":
		messages             = variables.tweets_2014_sarcasm
		messages_score       = variables.tweets_2014_sarcasm_score
		messages_score_svm   = variables.tweets_2014_sarcasm_score_svm
		messages_score_naive = variables.tweets_2014_sarcasm_score_naive
		messages_score_MS    = variables.tweets_2014_sarcasm_score_MS
		messages_score_LReg  = variables.tweets_2014_sarcasm_score_LReg
		messages_score_S140  = variables.tweets_2014_sarcasm_score_S140
		messages_score_RFor  = variables.tweets_2014_sarcasm_score_RFor
		messages_score_SGD   = variables.tweets_2014_sarcasm_score_SGD
		messages_score_ESumNoPG = variables.tweets_2014_sarcasm_score_ESumNoPG
		messages_positive    = variables.tweets_2014_sarcasm_positive
		messages_negative    = variables.tweets_2014_sarcasm_negative
		messages_neutral     = variables.tweets_2014_sarcasm_neutral

		variables.sar_outputs = []
	elif base == "sms":
		messages             = variables.sms_2013
		messages_score       = variables.sms_2013_score
		messages_score_svm   = variables.sms_2013_score_svm
		messages_score_naive = variables.sms_2013_score_naive
		messages_score_MS    = variables.sms_2013_score_MS
		messages_score_LReg  = variables.sms_2013_score_LReg
		messages_score_S140  = variables.sms_2013_score_S140
		messages_score_RFor  = variables.sms_2013_score_RFor
		messages_score_SGD   = variables.sms_2013_score_SGD
		messages_score_ESumNoPG = variables.sms_2013_score_ESumNoPG
		messages_positive    = variables.sms_2013_positive
		messages_negative    = variables.sms_2013_negative
		messages_neutral     = variables.sms_2013_neutral        

		variables.sms_outputs = []
	elif base == "all":
		messages             = variables.all_messages_in_file_order
		messages_score       = variables.all_polarities_in_file_order
		messages_score_svm   = variables.all_polarities_in_file_order_svm
		messages_score_naive = variables.all_polarities_in_file_order_naive
		messages_score_MS    = variables.all_polarities_in_file_order_MS
		messages_score_LReg  = variables.all_polarities_in_file_order_LReg
		messages_score_S140  = variables.all_polarities_in_file_order_S140
		messages_score_RFor  = variables.all_polarities_in_file_order_RFor
		messages_score_SGD   = variables.all_polarities_in_file_order_SGD
		messages_score_ESumNoPG = variables.all_polarities_in_file_order_ESumNoPG
		#messages = variables.tweets_2013 + variables.tweets_2014 + variables.tweets_liveJournal2014 + variables.sms_2013 + variables.tweets_2014_sarcasm
		#messages_score = variables.tweets_2013_score + variables.tweets_2014_score + variables.tweets_liveJournal2014_score + variables.sms_2013_score + variables.tweets_2014_sarcasm_score
		messages_positive    = variables.tweets_2013_positive + variables.tweets_2014_positive + variables.tweets_liveJournal2014_positive + variables.tweets_2014_sarcasm_positive + variables.sms_2013_positive
		messages_negative    = variables.tweets_2013_negative + variables.tweets_2014_negative + variables.tweets_liveJournal2014_negative + variables.tweets_2014_sarcasm_negative + variables.sms_2013_negative
		messages_neutral     = variables.tweets_2013_neutral  + variables.tweets_2014_neutral  + variables.tweets_liveJournal2014_neutral  + variables.tweets_2014_sarcasm_neutral  + variables.sms_2013_neutral

	
	if variables.SAVE_INCORRECT_EVALUATIONS:
		with open(variables.INCORRECT_EVALUATIONS, 'a') as f_incorrect:
			if base == "tweets2013": # only one header on file
				f_incorrect.write("[gold]\t[prediction]\t[neutral range]\t[base]\t[message]\n\n")

	variables.all_model_outputs = []

	for index, item in enumerate(messages): 
		message = str(messages[index]).strip().replace("'", "")
		message = message.replace("\\u2018", "").replace("\\u2019", "").replace("\\u002c", "")        
		message = "'" + message + "'"

		# check if the analysis will use an ensemble of all models
		if model_ensemble:
			models_analysis = []
			for mod in model:
				models_analysis.append(mod.replace("(x", "(" + message))
		else:
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

			elif(variables.use_only_emoticons):
				result = emoticonsPolaritySum(message)

			# SVM only
			elif(variables.use_only_svm):
				# TO-DO: normalize the values, considering the neutral range
				result = messages_score_svm[index]
			
			elif(variables.use_only_RForest_classifier):
				result = messages_score_RFor[index]

			# Textblob default (no semeval training provided) only
			elif(variables.use_only_textblob_no_train):
				result = TextBlob(message).sentiment.polarity

			# NaiveBayes only
			elif(variables.use_only_naive_bayes):
				if messages_score_naive[index] > 0:
					result = variables.neutral_superior_range + 1
				elif messages_score_naive[index] < 0:
					result = variables.neutral_inferior_range - 1
				elif messages_score_naive[index] == 0:
					result = random.uniform(variables.neutral_inferior_range, variables.neutral_superior_range)
				#result = messages_score_naive[index]

			# MS classifier only
			elif(variables.use_only_MS_classifier):
				# TO-DO: normalize the values, considering the neutral range
				result = messages_score_MS[index]

			# Logistic Regression only
			elif(variables.use_only_LReg_classifier):
				result = messages_score_LReg[index]

			# SGD only
			elif(variables.use_only_SGD_classifier):
				result = messages_score_SGD[index]

			elif(variables.use_all_classifiers_nopg_sum):
				result = messages_score_ESumNoPG[index]

			# Use all classifiers - get the majority
			elif(variables.use_all_classifiers):
				all_classifiers = []
				
				if base == "livejournal":
					all_classifiers.append(floatToStr_polarity_value(messages_score_svm[index],   variables.neutral_inferior_range, variables.neutral_superior_range))
					#all_classifiers.append(floatToStr_polarity_value(messages_score_naive[index], variables.neutral_inferior_range, variables.neutral_superior_range))
					all_classifiers.append(floatToStr_polarity_value(messages_score_LReg[index], variables.neutral_inferior_range, variables.neutral_superior_range))
					#all_classifiers.append(floatToStr_polarity_value(messages_score_RFor[index],  variables.neutral_inferior_range, variables.neutral_superior_range))
					all_classifiers.append(floatToStr_polarity_value(messages_score_SGD[index],   variables.neutral_inferior_range, variables.neutral_superior_range))
					all_classifiers.append(floatToStr_polarity_value(float(eval(model_analysis)), variables.neutral_inferior_range, variables.neutral_superior_range))
					#all_classifiers.append(floatToStr_polarity_value(messages_score_MS[index], variables.neutral_inferior_range, variables.neutral_superior_range))
					#all_classifiers.append(floatToStr_polarity_value(messages_score_S140[index], variables.neutral_inferior_range, variables.neutral_superior_range))
					#all_classifiers.append(floatToStr_polarity_value(TextBlob(message).sentiment.polarity, variables.neutral_inferior_range, variables.neutral_superior_range))
					if hasEmoticons(message):
						all_classifiers.append(floatToStr_polarity_value(emoticonsPolaritySum(message), variables.neutral_inferior_range, variables.neutral_superior_range))
					if hasHashtag(message):
						all_classifiers.append(floatToStr_polarity_value(hashtagPolaritySum(message), variables.neutral_inferior_range, variables.neutral_superior_range))
				else:
					all_classifiers.append(floatToStr_polarity_value(messages_score_svm[index],   variables.neutral_inferior_range, variables.neutral_superior_range))
					#all_classifiers.append(floatToStr_polarity_value(messages_score_naive[index], variables.neutral_inferior_range, variables.neutral_superior_range))
					all_classifiers.append(floatToStr_polarity_value(messages_score_LReg[index], variables.neutral_inferior_range, variables.neutral_superior_range))
					all_classifiers.append(floatToStr_polarity_value(messages_score_RFor[index],  variables.neutral_inferior_range, variables.neutral_superior_range))
					all_classifiers.append(floatToStr_polarity_value(messages_score_SGD[index],   variables.neutral_inferior_range, variables.neutral_superior_range))
					all_classifiers.append(floatToStr_polarity_value(float(eval(model_analysis)), variables.neutral_inferior_range, variables.neutral_superior_range))
					#all_classifiers.append(floatToStr_polarity_value(messages_score_MS[index], variables.neutral_inferior_range, variables.neutral_superior_range))
					#all_classifiers.append(floatToStr_polarity_value(messages_score_S140[index], variables.neutral_inferior_range, variables.neutral_superior_range))
					#all_classifiers.append(floatToStr_polarity_value(TextBlob(message).sentiment.polarity, variables.neutral_inferior_range, variables.neutral_superior_range))
					if hasEmoticons(message):
						all_classifiers.append(floatToStr_polarity_value(emoticonsPolaritySum(message), variables.neutral_inferior_range, variables.neutral_superior_range))
					if hasHashtag(message):
						all_classifiers.append(floatToStr_polarity_value(hashtagPolaritySum(message), variables.neutral_inferior_range, variables.neutral_superior_range))

				r = get_best_evaluation(all_classifiers)            

				if r == "positive":
					result = variables.neutral_superior_range + 1
				elif r == "negative":
					result = variables.neutral_inferior_range - 1
				elif r == "neutral":
					result = random.uniform(variables.neutral_inferior_range, variables.neutral_superior_range)
				elif r[:4] == "DRAW":
					
					# Equal Draw - classify again without RF
					if r.split("_")[1] == "all":
						all_classifiers = []
						all_classifiers.append(floatToStr_polarity_value(messages_score_svm[index], variables.neutral_inferior_range, variables.neutral_superior_range))
						all_classifiers.append(floatToStr_polarity_value(messages_score_naive[index], variables.neutral_inferior_range, variables.neutral_superior_range))
						all_classifiers.append(floatToStr_polarity_value(messages_score_LReg[index], variables.neutral_inferior_range, variables.neutral_superior_range))
						all_classifiers.append(floatToStr_polarity_value(messages_score_SGD[index], variables.neutral_inferior_range, variables.neutral_superior_range))
						all_classifiers.append(floatToStr_polarity_value(float(eval(model_analysis)), variables.neutral_inferior_range, variables.neutral_superior_range))
						if hasEmoticons(message):
							all_classifiers.append(floatToStr_polarity_value(emoticonsPolaritySum(message), variables.neutral_inferior_range, variables.neutral_superior_range))
						if hasHashtag(message):
							all_classifiers.append(floatToStr_polarity_value(hashtagPolaritySum(message), variables.neutral_inferior_range, variables.neutral_superior_range))

						r = get_best_evaluation(all_classifiers)      
						if r == "positive":
							result = variables.neutral_superior_range + 1
						elif r == "negative":
							result = variables.neutral_inferior_range - 1
						elif r == "neutral":
							result = random.uniform(variables.neutral_inferior_range, variables.neutral_superior_range)      
						elif r[:4] == "DRAW":
							if base == "sms":
								result = messages_score_SGD[index]
								if floatToStr_polarity_value(messages_score_SGD[index], variables.neutral_inferior_range, variables.neutral_superior_range) == r.split("_")[1].strip():
									result = float(eval(model_analysis))
							else:
								result = float(eval(model_analysis))
								if floatToStr_polarity_value(float(eval(model_analysis)), variables.neutral_inferior_range, variables.neutral_superior_range) == r.split("_")[1].strip():
									result = messages_score_SGD[index]   
									#result = messages_score_svm[index]   
					else:
						result = float(eval(model_analysis))
						if floatToStr_polarity_value(float(eval(model_analysis)), variables.neutral_inferior_range, variables.neutral_superior_range) == r.split("_")[1].strip():
							result = messages_score_SGD[index]
							#result = messages_score_svm[index]

				#if base == "sarcasm":
				#	result = float(eval(model_analysis))

			# GP only
			else:
				if model_ensemble:
					results_models_ensemble = []
					for m in models_analysis:
						res = float(eval(m))
						if res > variables.neutral_superior_range:
							results_models_ensemble.append("positive")
						elif res < variables.neutral_inferior_range:
							results_models_ensemble.append("negative")
						else:
							results_models_ensemble.append("neutral")
					
					#print("results models ensemble " + str(results_models_ensemble))

					ensemble_result = get_best_evaluation(results_models_ensemble)
					if ensemble_result == "positive":
						result = variables.neutral_superior_range + 1
					elif ensemble_result == "negative":
						result = variables.neutral_inferior_range - 1
					elif ensemble_result == "neutral":
						result = random.uniform(variables.neutral_inferior_range, variables.neutral_superior_range)
					elif ensemble_result[:4] == "DRAW":
						print(ensemble_result)
						result = messages_score_svm[index]

				else:
					#print("[evaluate using only GP in normal mode (no ensemble)")
					result = float(eval(model_analysis))

					if base == "tweets2013":
						variables.t2k13_outputs.append(result)
					elif base == "tweets2014":
						variables.t2k14_outputs.append(result)
					elif base == "sms":
						variables.sms_outputs.append(result)
					elif base == "livejournal":
						variables.lj_outputs.append(result)
					elif base == "sarcasm":
						variables.sar_outputs.append(result)

					variables.all_model_outputs.append(result)

					if result <= variables.neutral_superior_range and result >= variables.neutral_inferior_range:
						result = messages_score_svm[index]
					
					#testing the svm on the bases that it's the best
					#if base == "tweets2014" or base == "sms":
					#    result = messages_score_svm[index]
					#else:
					#    result = float(eval(model_analysis))

					#if result == 0:
					#    if(variables.use_hashtag_analysis and hasHashtag(message)):
					#        result = hashtagPolaritySum(message)
					#    elif base == "sarcasm":
					#        result = TextBlob(message).sentiment.polarity
					#    else:
					#        result = messages_score_svm[index]



		except Exception as e:
			print("exception 2: " + str(e))
			#print("\n\n[WARNING] eval(model_analysis) exception for the message: " + message + "\n\n")
			continue

		if(base == "all"):
			model_results_to_count_occurrences.append(result)


		if base == "sarcasm" and variables.INVERT_SARCASM and result != 0:
			result *= -1

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

				if base != "all" and variables.SAVE_INCORRECT_EVALUATIONS:
					with open(variables.INCORRECT_EVALUATIONS, 'a') as f_incorrect:
						f_incorrect.write(str(messages_score[index]) + "\t" + str(result) + "\t[" + str(variables.neutral_inferior_range) + ", " + str(variables.neutral_superior_range) + "]\t" + base + "\t" + message + "\n")
				
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

				if base != "all" and variables.SAVE_INCORRECT_EVALUATIONS:
					with open(variables.INCORRECT_EVALUATIONS, 'a') as f_incorrect:
						f_incorrect.write(str(messages_score[index]) + "\t" + str(result) + "\t[" + str(variables.neutral_inferior_range) + ", " + str(variables.neutral_superior_range) + "]\t" + base + "\t" + message + "\n")

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

				elif result > variables.neutral_superior_range:
					false_positive += 1
					goldNeu_classPos += 1
					if base == "all":
						goldNeu_classPos_value.append(result)

				if base != "all" and variables.SAVE_INCORRECT_EVALUATIONS:
					with open(variables.INCORRECT_EVALUATIONS, 'a') as f_incorrect:
						f_incorrect.write(str(messages_score[index]) + "\t" + str(result) + "\t[" + str(variables.neutral_inferior_range) + ", " + str(variables.neutral_superior_range) + "]\t" + base + "\t" + message + "\n")


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
	if not model_ensemble:
		print("[model]: " + str(model))
	else:
		print("[ensemble of " + str(len(model)) + " models]")
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
				if not model_ensemble:
					f.write("[Model]\t" + model + "\n")
				else:
					f.write("[Ensemble of " + str(len(models_analysis)) + " models]\n")
				f.write("# [results - f1]\n")

			f.write(base + "\t" + str(round(f1_positive_negative_avg, 4)) + "\n")
			if base == "all":
				if not model_ensemble:
					f.write("\n# [weights]\n")
					if (float(len(variables.w1)) > 0):
						f.write("# _w1\tall values:\t" + str(set(variables.w1))  + "\tmean:\t" + str(sum(variables.w1) / float(len(variables.w1))) + "\n")
						f.write("# _w2\tall values:\t" + str(set(variables.w2))  + "\tmean:\t" + str(sum(variables.w2) / float(len(variables.w2))) + "\n")
						f.write("# _w3\tall values:\t" + str(set(variables.w3))  + "\tmean:\t" + str(sum(variables.w3) / float(len(variables.w3))) + "\n")
						f.write("# _w4\tall values:\t" + str(set(variables.w4))  + "\tmean:\t" + str(sum(variables.w4) / float(len(variables.w4))) + "\n")
						f.write("# _w5\tall values:\t" + str(set(variables.w5))  + "\tmean:\t" + str(sum(variables.w5) / float(len(variables.w5))) + "\n")
						f.write("# _w6\tall values:\t" + str(set(variables.w6))  + "\tmean:\t" + str(sum(variables.w6) / float(len(variables.w6))) + "\n")
						f.write("# _w7\tall values:\t" + str(set(variables.w7))  + "\tmean:\t" + str(sum(variables.w7) / float(len(variables.w7))) + "\n")
						f.write("# _w8\tall values:\t" + str(set(variables.w8))  + "\tmean:\t" + str(sum(variables.w8) / float(len(variables.w8))) + "\n")
						f.write("# _w9\tall values:\t" + str(set(variables.w9))  + "\tmean:\t " + str(sum(variables.w9) / float(len(variables.w9))) + "\n")
						f.write("# _w10\tall values:\t" + str(set(variables.w10)) + "\tmean:\t" + str(sum(variables.w10) / float(len(variables.w10))) + "\n")
						f.write("# _w11\tall values:\t" + str(set(variables.w11)) + "\tmean:\t" + str(sum(variables.w11) / float(len(variables.w11))) + "\n\n")

					f.write("# [neutral ranges]\n")
					f.write("# " + str(set(variables.neutral_values)) + "\n\n")

				f.write("# [confusion matrix]\n")
				f.write("#           |  Gold_Pos  |  Gold_Neg  |  Gold_Neu  |\n")
				f.write("# --------------------------------------------------\n")
				f.write("# Pred_Pos  |  " + '{message: <{width}}'.format(message=str(true_positive), width=8) + "  |  " + '{message: <{width}}'.format(message=str(goldNeg_classPos), width=8) + "  |  " + '{message: <{width}}'.format(message=str(goldNeu_classPos), width=8) + "  |\n")
				f.write("# Pred_Neg  |  " + '{message: <{width}}'.format(message=str(goldPos_classNeg), width=8) + "  |  " + '{message: <{width}}'.format(message=str(true_negative), width=8) + "  |  " + '{message: <{width}}'.format(message=str(goldNeu_classNeg), width=8) + "  |\n")
				f.write("# Pred_Neu  |  " + '{message: <{width}}'.format(message=str(goldPos_classNeu), width=8) + "  |  " + '{message: <{width}}'.format(message=str(goldNeg_classNeu), width=8) + "  |  " + '{message: <{width}}'.format(message=str(true_neutral), width=8)  + "  |\n\n")
				
				if not model_ensemble:
					f.write("# [all outputs] (" + str(len(variables.all_model_outputs)) + " outputs)\n")
					f.write("# [Tweets2013] "  + str(variables.t2k13_outputs) + "\n\n")
					f.write("# [Tweets2014] "  + str(variables.t2k14_outputs) + "\n\n")
					f.write("# [SMS] "         + str(variables.sms_outputs)   + "\n\n")
					f.write("# [LiveJournal] " + str(variables.lj_outputs)    + "\n\n")
					f.write("# [Sarcasm] "     + str(variables.sar_outputs)   + "\n\n")
					#f.write("# " + str(variables.all_model_outputs) + "\n")
					
					import seaborn as sns
					import matplotlib.pyplot as plt

					#plt.interactive(True)


					sns.set(style="whitegrid")
					tips = sns.load_dataset("tips")
					#ax = sns.boxplot(x=tips["total_bill"])

					all_outs = []

					#labels = ['Twitter2013', 'Twitter2014', 'Sarcasm', 'SMS', 'LiveJournal', 'Todas']

					all_outs.append(variables.t2k13_outputs)
					all_outs.append(variables.t2k14_outputs)
					all_outs.append(variables.sar_outputs)
					all_outs.append(variables.sms_outputs)
					all_outs.append(variables.lj_outputs)
					all_outs.append(variables.all_model_outputs)

					ax = sns.boxplot(data=all_outs, orient="v", palette="Set2", showmeans=True)
					
					plt.ylim(-6, 6)


					#plt.ion()
					#plt.show()

					#input("Press enter to continue")

				f.write("# ---------//---------\n\n")

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

	all_w1_means = []
	all_w2_means = []
	all_w3_means = []
	all_w4_means = []
	all_w5_means = []
	all_w6_means = []
	all_w7_means = []
	all_w8_means = []
	all_w9_means = []
	all_w10_means = []
	all_w11_means = []

	with open(variables.FILE_RESULTS, 'r') as f:
		for line in f:
			w_index = 0
			if line.startswith("["):
				models += 1
			
			elif len(line) > 1 and not line.startswith("#"):
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
			
			elif len(line) > 1 and line.startswith("# _w") and variables.save_dic_means_on_result_file:
				if (float(line.split('\t')[4].strip()) > variables.limit_dictionary_weight):
					mean = variables.limit_dictionary_weight
				else:
					mean = float(line.split('\t')[4].strip())

				if line.startswith("# _w1\t"):
					all_w1_means.append(mean)
				if line.startswith("# _w2\t"):
					all_w2_means.append(mean) 
				if line.startswith("# _w3\t"):
					all_w3_means.append(mean)                                   
				if line.startswith("# _w4\t"):
					all_w4_means.append(mean)
				if line.startswith("# _w5\t"):
					all_w5_means.append(mean)
				if line.startswith("# _w6\t"):
					all_w6_means.append(mean)
				if line.startswith("# _w7\t"):
					all_w7_means.append(mean)
				if line.startswith("# _w8\t"):
					all_w8_means.append(mean)
				if line.startswith("# _w9\t"):
					all_w9_means.append(mean)
				if line.startswith("# _w10\t"):
					all_w10_means.append(mean)
				if line.startswith("# _w11\t"):
					all_w11_means.append(mean)                                                       

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
			f.write("\n\nStandard Deviation")
			f.write("\nStandard Deviation Tweets2013\t" + str(calcStdDeviation(calcVariance(t2k13_list, models))))
			f.write("\nStandard Deviation Tweets2014\t" + str(calcStdDeviation(calcVariance(t2k14_list, models))))
			f.write("\nStandard Deviation SMS\t" + str(calcStdDeviation(calcVariance(sms_list, models))))
			f.write("\nStandard Deviation Live Journal\t" + str(calcStdDeviation(calcVariance(liveJ_list, models))))
			f.write("\nStandard Deviation Sarcasm\t" + str(calcStdDeviation(calcVariance(sarcasm_list, models))))
			f.write("\nStandard Deviation All\t" + str(calcStdDeviation(calcVariance(allB_list, models))))

			if variables.save_dic_means_on_result_file:
				if variables.save_all_dic_values_on_result_file:
					f.write("\n\nAll dictionaries weights values")
					f.write("\nw1 values\t" + str(all_w1_means))           
					f.write("\nw2 values\t" + str(all_w2_means))
					f.write("\nw3 values\t" + str(all_w3_means))
					f.write("\nw4 values\t" + str(all_w4_means))
					f.write("\nw5 values\t" + str(all_w5_means))
					f.write("\nw6 values\t" + str(all_w6_means))
					f.write("\nw7 values\t" + str(all_w7_means))                
					f.write("\nw8 values\t" + str(all_w8_means))                
					f.write("\nw9 values\t" + str(all_w9_means))                                
					f.write("\nw10 values\t" + str(all_w10_means))                            
					f.write("\nw11 values\t" + str(all_w11_means))
				f.write("\n\nAll dictionaries weights means")
				
				f.write("\nw1 mean\t" + str(round(sum(all_w1_means) / float(len(all_w1_means)), 4)))
				f.write("\nw2 mean\t" + str(round(sum(all_w2_means) / float(len(all_w2_means)), 4)))
				f.write("\nw3 mean\t" + str(round(sum(all_w3_means) / float(len(all_w3_means)), 4)))
				f.write("\nw4 mean\t" + str(round(sum(all_w4_means) / float(len(all_w4_means)), 4)))
				f.write("\nw5 mean\t" + str(round(sum(all_w5_means) / float(len(all_w5_means)), 4)))
				f.write("\nw6 mean\t" + str(round(sum(all_w6_means) / float(len(all_w6_means)), 4)))
				f.write("\nw7 mean\t" + str(round(sum(all_w7_means) / float(len(all_w7_means)), 4)))
				f.write("\nw8 mean\t" + str(round(sum(all_w8_means) / float(len(all_w8_means)), 4)))
				f.write("\nw9 mean\t" + str(round(sum(all_w9_means) / float(len(all_w9_means)), 4)))
				f.write("\nw10 mean\t" + str(round(sum(all_w10_means) / float(len(all_w10_means)), 4)))
				f.write("\nw11 mean\t" + str(round(sum(all_w11_means) / float(len(all_w11_means)), 4)))
				#f.write("\nw2 mean\t" + str(sum(all_w2_means) / float(len(all_w2_means))))
				#f.write("\nw3 mean\t" + str(sum(all_w3_means) / float(len(all_w3_means))))
				#f.write("\nw4 mean\t" + str(sum(all_w4_means) / float(len(all_w4_means))))
				#f.write("\nw5 mean\t" + str(sum(all_w5_means) / float(len(all_w5_means))))
				#f.write("\nw6 mean\t" + str(sum(all_w6_means) / float(len(all_w6_means))))
				#f.write("\nw7 mean\t" + str(sum(all_w7_means) / float(len(all_w7_means))))
				#f.write("\nw8 mean\t" + str(sum(all_w8_means) / float(len(all_w8_means))))
				#f.write("\nw9 mean\t" + str(sum(all_w9_means) / float(len(all_w9_means))))
				#f.write("\nw10 mean\t" + str(sum(all_w10_means) / float(len(all_w10_means))))
				#f.write("\nw11 mean\t" + str(sum(all_w11_means) / float(len(all_w11_means))))

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


#####################################################################################
## 2 classes ## 
# TO-DO: refactory the code/functions
#####################################################################################

# Evaluate the test messages using the model
# http://text-analytics101.rxnlp.com/2014/10/computing-precision-and-recall-for.html
def evaluateMessages_2classes(model):
	global model_results_to_count_occurrences
	print("[starting evaluation of messages (2 classes)]")
	
	# test
	count_has_date = 0
	# test

	# test
	variables.neutral_inferior_range = 0
	variables.neutral_superior_range = 0
	# test

	# parameters to calc the metrics
	true_positive  = 0
	true_negative  = 0
	false_positive = 0
	false_negative = 0

	# confusion matrix 
	goldPos_classNeg = 0
	goldNeg_classPos = 0
	# confusion matrix 

	# calc of mode
	goldPos_classPos_value = []
	goldPos_classNeg_value = []
	goldNeg_classPos_value = []
	goldNeg_classNeg_value = []
	# calc of mode
	
	accuracy = 0

	precision_positive = 0
	precision_negative = 0
	precision_avg = 0

	recall_positive = 0
	recall_negative = 0
	recall_avg = 0

	f1_positive = 0
	f1_negative = 0
	
	f1_avg = 0
	f1_positive_negative_avg = 0

	false_negative_log = 0

	message = ""
	model_analysis = ""
	result = 0

	messages = []
	messages_score = []
	messages_positive = 0
	messages_negative = 0
	
	if len(variables.tweets_2013) == 0:
		loadTestTweets_STS()
		
	messages = variables.tweets_sts_test
	messages_score = variables.tweets_sts_score_test
	messages_positive = variables.tweets_sts_positive
	messages_negative = variables.tweets_sts_negative

	for index, item in enumerate(messages): 
		message = str(messages[index]).strip().replace("'", "")
		message = message.replace("\\u2018", "").replace("\\u2019", "").replace("\\u002c", "")        
		message = "'" + message + "'"

		model_analysis = model.replace("(x", "(" + message)

		if not len(message) > 0:
			continue
		
		try:
			if(variables.use_emoticon_analysis and hasEmoticons(message)):
				result = emoticonsPolaritySum(message)

			# SVM only
			elif(variables.use_only_svm):
				result = variables.svm_normalized_values[index]
			
			# GP only
			else:
				result = float(eval(model_analysis))


		except Exception as e:
			print("exception 2: " + str(e))
			continue
		
		if messages_score[index] > 0:
			if result > variables.neutral_superior_range:
				true_positive += 1
			else:
				false_negative += 1
				goldPos_classNeg += 1
				
		elif messages_score[index] < 0:
			if result < variables.neutral_inferior_range:
				true_negative += 1
			else:
				false_positive += 1
				goldNeg_classPos += 1


	if true_positive + false_positive + true_negative + false_negative > 0:
		accuracy = (true_positive + true_negative) / (true_positive + false_positive + true_negative + false_negative)

	# Begin PRECISION
	if true_positive + false_positive > 0:
		precision_positive = true_positive / (true_positive + false_positive)


	if true_negative + false_negative > 0:
		precision_negative = true_negative / (true_negative + false_negative)
	# End PRECISION

	# Begin RECALL
	if messages_positive > 0:
		recall_positive = true_positive / messages_positive


	if messages_negative > 0:
		recall_negative = true_negative / messages_negative
	# End RECALL

	# Begin F1
	if precision_positive + recall_positive > 0:
		f1_positive = 2 * (precision_positive * recall_positive) / (precision_positive + recall_positive)

	if precision_negative + recall_negative > 0:
		f1_negative = 2 * (precision_negative * recall_negative) / (precision_negative + recall_negative)                
	# End F1        

	# Precision, Recall and f1 means
	precision_avg = (precision_positive + precision_negative) / 2
	
	recall_avg = (recall_positive + recall_negative) / 2

	f1_avg = (f1_positive + f1_negative) / 2

	f1_positive_negative_avg = (f1_positive + f1_negative) / 2         

	print("\n")

	print("[messages evaluated]: " + str(len(messages)) + " (" + str(messages_positive) + " positives, " + str(messages_negative) + " negatives)")
	print("[correct evaluations]: " + str(true_positive + true_negative) + " (" + str(true_positive) + " positives, " + str(true_negative) + " negatives)")
	print("[model]: " + str(model))
	print("[accuracy]: " + str(round(accuracy, 4)))
	print("[precision_positive]: " + str(round(precision_positive, 4)))
	print("[precision_negative]: " + str(round(precision_negative, 4)))
	print("[precision_avg]: " + str(round(precision_avg, 4)))
	print("[recall_positive]: " + str(round(recall_positive, 4)))
	print("[recall_negative]: " + str(round(recall_negative, 4)))
	print("[recall avg]: " + str(round(recall_avg, 4)))
	print("[f1_positive]: " + str(round(f1_positive, 4)))
	print("[f1_negative]: " + str(round(f1_negative, 4)))
	print("[f1 avg]: " + str(round(f1_avg, 4)))
	print("[f1 avg SemEval (positive and negative)]: " + str(round(f1_positive_negative_avg, 4)))    
	print("[true_positive]: " + str(true_positive))
	print("[false_positive]: " + str(false_positive))
	print("[true_negative]: " + str(true_negative))
	print("[false_negative]: " + str(false_negative))
	print("[dictionary quantity]: " + str(variables.dic_loaded_total))

	
	print("\n")
	print("Confusion Matrix\n")
	print("          |  Gold_Pos  |  Gold_Neg  |")
	print("-------------------------------------")
	print("Pred_Pos  |  " + '{message: <{width}}'.format(message=str(true_positive), width=8) + "  |  " + '{message: <{width}}'.format(message=str(goldNeg_classPos), width=8) + "  |")
	print("Pred_Neg  |  " + '{message: <{width}}'.format(message=str(goldPos_classNeg), width=8) + "  |  " + '{message: <{width}}'.format(message=str(true_negative), width=8) + "  |")

	print("\n")

	if variables.save_file_results:
		with open(variables.FILE_RESULTS_2CLASSES, 'a') as f:
			f.write("[Model]\t" + model + "\n")
			f.write(str(round(f1_positive_negative_avg, 4)) + "\n")
			f.write("\n")


def resultsAnalysis_2classes():
	models = 0

	array_values = []

	with open(variables.FILE_RESULTS_2CLASSES, 'r') as f:
		for line in f:
			if line.startswith("["):
				models += 1
			elif len(line) > 1:
				value = float(line)
				array_values.append(value)

	with open(variables.FILE_RESULTS_2CLASSES, 'a') as f:
		if models > 0:
			f.write("\n\n##Statistics##\n\n")
			f.write(str(models) + " models evaluated\n")
			f.write(str(variables.dic_loaded_total) + " dictionaries\n\n")
			f.write("AVG F1-measure:\t" + str(round((sum(array_values) / models), 4)))
			f.write("\nBest F1-measure Value:\t" + str(round(max(array_values), 4)))
			f.write("\n\nValues by model")
			f.write("\n" + str(array_values))
			f.write("\nStandard Deviation:\t" + str(calcStdDeviation(calcVariance(array_values, models))))


def calcVariance(base, total_models):
	diffs = []
	
	avg = sum(base) / total_models
	
	for base_value in base:
		diffs.append(math.pow(base_value - avg, 2))

	variance = sum(diffs) / total_models
	return variance


def calcStdDeviation(variance):
	return math.sqrt(variance)