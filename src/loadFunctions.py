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
	
		dic_words['dictionary_name']['positive']['word'] = polarity
		dic_words['dictionary_name']['negative']['word'] = polarity

		Example:
			
			Dictionary Sentiwordnet (word<tab>polarity)
				able	0.125

				dic_words['sentiwordnet']['positive']['able'] = 0.125
	"""
	start = time.time()
	print("\n[loading dictionaries]")

	# Saving all pos/neg messages on unique array - I'll use this in some gp functions
	variables.dic_words["all"] = {}
	variables.dic_words["all"]["positive"], variables.dic_words["all"]["negative"] = {}, {}

	for dic in (variables.DICTIONARIES + variables.SPECIAL_DICTIONARIES):
		if dic in variables.SPECIAL_DICTIONARIES or variables.USE_DIC[dic]:
			variables.DIC_LOADED[dic] = True
			variables.dic_words[dic]["positive"], variables.dic_words[dic]["negative"] = loadDictionaryValues(dic, variables.DIC_PATH[dic])

			variables.dic_words["all"]["positive"].update(variables.dic_words[dic]["positive"])
			variables.dic_words["all"]["negative"].update(variables.dic_words[dic]["negative"])

	for dic in variables.CLASSLESS_DICTIONARIES:
		variables.dic_words[dic] = loadClasslessDictionaries(dic, variables.DIC_PATH[dic])

	print("[all dictionaries (" + str(variables.dic_loaded_total) + ") loaded][" + str(format(time.time() - start, '.3g')) + " seconds]\n")


def loadMessages(module, file_path, limit=0):
	"""Load the messages

		Args:
			module    (str): train or test
			file_path (str): path of the file
			limit     (int): limit of messages loaded
	"""
	start = time.time()
	print("\n[loading " + module + " tweets]")

	msgs = {}
	msgs_all = []

	with open(file_path, "r") as f:
		for line in f:
			if limit == 0 or (limit != 0 and len(msgs_all) < limit):
				msgs['message']   = line.split('\t')[0].strip()
				msgs['label']     = line.split('\t')[1].strip()
				msgs['num_label'] = convertLabelToNum(line.split('\t')[1].strip())

				if msgs['label'] == 'positive':
					variables.POSITIVE_MESSAGES += 1
				elif msgs['label'] == 'negative':
					variables.NEGATIVE_MESSAGES += 1
				elif msgs['label'] == 'neutral':
					variables.NEUTRAL_MESSAGES  += 1

				msgs_all.append(msgs)
				msgs = {}

	return msgs_all


def loadTrainMessages(limit=0):
	""" Load the train messages
	
		Limit is the max messages to load

	"""
	variables.messages['train'] = loadMessages("train", variables.DATASET_PATH['train'], limit)


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
	variables.dic_loaded_total += 1
	pos_words, neg_words = {}, {}

	variables.dic_words[dictionary_name.lower()] = {}
	variables.dic_words[dictionary_name.lower()]["positive"], variables.dic_words[dictionary_name.lower()]["negative"] = {}, {}

	with open(dictionary_path + dictionary_name.lower() + "_positive.txt", "r") as f:
		for line in f:
			pos_words[line.split("\t")[0].strip()] = line.split("\t")[1].strip()

	with open(dictionary_path + dictionary_name.lower() + "_negative.txt", "r") as f:
		for line in f:
			neg_words[line.split("\t")[0].strip()] = line.split("\t")[1].strip()

	if variables.log_loads:
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
	
	if variables.log_loads:
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
