""" 
<Author> Airton Bordin Junior
<Email>  airtonbjunior@gmail.com

Federal University of Goias (UFG)
Computer Science Master's Degree


loadFunctions.py
	Functions that handle loads files/dictionaries/etc

"""
import time
import variables

def loadDictionaries():
	"""Load the dictionaries words/polarities

	The dic_words dictionary (defined in variables.py) will be populated using the follow format: 
	
		DIC_WORDS['dictionary_name']['positive']['word'] = polarity
		DIC_WORDS['dictionary_name']['negative']['word'] = polarity

		Example:
			
			Dictionary Sentiwordnet (word<tab>polarity)
				able	0.125

				DIC_WORDS['sentiwordnet']['positive']['able'] = 0.125
	"""
	start = time.time()
	print("\n[loading dictionaries]")

	# Saving all pos/neg messages on unique array - I'll use this in some gp functions
	variables.DIC_WORDS["all"] = {}
	variables.DIC_WORDS["all"]["positive"], variables.DIC_WORDS["all"]["negative"] = {}, {}

	for dic in (variables.DICTIONARIES + variables.SPECIAL_DICTIONARIES):
		if dic in variables.SPECIAL_DICTIONARIES or variables.USE_DIC[dic]:
			variables.DIC_LOADED[dic] = True
			variables.DIC_WORDS[dic]["positive"], variables.DIC_WORDS[dic]["negative"] = loadDictionaryValues(dic, variables.DIC_PATH[dic])

			variables.DIC_WORDS["all"]["positive"].update(variables.DIC_WORDS[dic]["positive"])
			variables.DIC_WORDS["all"]["negative"].update(variables.DIC_WORDS[dic]["negative"])

	for dic in variables.CLASSLESS_DICTIONARIES:
		variables.DIC_WORDS[dic] = loadClasslessDictionaries(dic, variables.DIC_PATH[dic])

	print("[all dictionaries (" + str(variables.DIC_LOADED['total']) + ") loaded][" + str(format(time.time() - start, '.3g')) + " seconds]\n")


def loadMessages(module, file_path):
	"""Load the messages

		Args:
			module    (str): train or test
			file_path (str): path of the file
	"""
	start = time.time()
	print("\n[loading " + module + " messages]")

	msgs = {}
	msgs_all = []

	with open(file_path, "r") as f:
		for line in f:
			if (variables.MAX[module]['all'] == 0) or (len(msgs_all) < variables.MAX[module]['all']):
				msgs['message']       = line.split('\t')[0].strip()
				msgs['label']         = line.split('\t')[1].strip()
				msgs['num_label']     = convertLabelToNum(line.split('\t')[1].strip())

				if msgs['label'] == 'positive':
					if variables.POSITIVE_MESSAGES >= variables.MAX[module]['positive']:
						continue
					variables.POSITIVE_MESSAGES += 1
				
				elif msgs['label'] == 'negative':
					if variables.NEGATIVE_MESSAGES >= variables.MAX[module]['negative']:
						continue
					variables.NEGATIVE_MESSAGES += 1
				
				elif msgs['label'] == 'neutral':
					if variables.NEUTRAL_MESSAGES >= variables.MAX[module]['neutral']:
						continue
					variables.NEUTRAL_MESSAGES  += 1

				msgs_all.append(msgs)
				msgs = {}

	print("  [" + module + " messages loaded (" + str(len(msgs_all)) + " messages)][" + str(variables.POSITIVE_MESSAGES) + " positive, " + str(variables.NEGATIVE_MESSAGES) + " negative and " + str(variables.NEUTRAL_MESSAGES ) + " neutral][" + str(format(time.time() - start, '.3g')) + " seconds]\n")
	return msgs_all


def loadTrainMessages():
	""" Load the train messages

	"""
	variables.MESSAGES['train'] = loadMessages("train", variables.DATASET_PATH['train'])


"""
	Aux functions
"""
def loadDictionaryValues(dictionary_name, dictionary_path):
	"""
		Args:
			dictionary_name (str):
			dictionary_path (str):

		Return
			pos_words (dict):
			neg_words (dict):


		The dictionaries name are: "liu", "sentiwordnet", "afinn", "vader", "slang", "effect", "semeval2015", "nrc", "gi", "s140", "mpqa"
	"""
	startDic = time.time()
	print("  [loading " + str(dictionary_name.strip().lower()) + "]")
	variables.DIC_LOADED['total'] += 1
	pos_words, neg_words = {}, {}

	variables.DIC_WORDS[dictionary_name.lower()] = {}
	variables.DIC_WORDS[dictionary_name.lower()]["positive"], variables.DIC_WORDS[dictionary_name.lower()]["negative"] = {}, {}

	with open(dictionary_path + dictionary_name.lower() + "_positive.txt", "r") as f:
		for line in f:
			pos_words[line.split("\t")[0].strip()] = line.split("\t")[1].strip()

	with open(dictionary_path + dictionary_name.lower() + "_negative.txt", "r") as f:
		for line in f:
			neg_words[line.split("\t")[0].strip()] = line.split("\t")[1].strip()

	if variables.LOG['loads']:
		print("    [" + str(len(pos_words) + len(neg_words)) + " words loaded]")
		print("    [" + str(len(pos_words)) + " positive and " + str(len(neg_words)) + " negative]")
		print("      [" + str(dictionary_name.strip().lower()) + " dictionary loaded][" + str(format(time.time() - startDic, '.3g')) + " seconds]\n")

	return pos_words, neg_words


def loadClasslessDictionaries(dictionary_name, dictionary_path):
	startDic = time.time()
	words = []
	print("  [loading " + str(dictionary_name.strip().lower()) + "]")	
	with open(dictionary_path + dictionary_name.strip().lower() + ".txt", "r") as f:
		for line in f:
			words.append(line.strip().lower())
	
	if variables.LOG['loads']:
		print("    [" + str(len(words)) + " words]")
		print("      [" + str(dictionary_name.strip().lower()) + " dictionary loaded][" + str(format(time.time() - startDic, '.3g')) + " seconds]\n")
	
	return words


def convertLabelToNum(label):
	if label == "positive":
		return 1
	elif label == "negative":
		return -1
	elif label == "neutral":
		return 0
