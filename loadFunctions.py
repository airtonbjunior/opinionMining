# Airton Bordin Junior
# airtonbjunior@gmail.com
# Federal University of Goias (UFG)
# Computer Science Master's Degree

import time
import variables

def loadDictionaryValues(dictionary_name, dictionary_path):
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