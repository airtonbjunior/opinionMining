""" 
<Author> Airton Bordin Junior
<Email>  airtonbjunior@gmail.com

Federal University of Goias (UFG)
Computer Science Master's Degree


gpFunctions.py
	Functions used by the Genetic Programming Algorithm

	This functions will be called on classifier.py by the DEAP primitives

"""
import re
import variables
import string
from validate_email import validate_email

"""
	Aux functions
"""
def getURLs(message):
	#improve this Regular Expression to get www.something.com and others
	return re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', message)

"""
	Math functions (numeric -> numeric)
"""
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

def protectedDiv(left, right):
	try:
		return left / right
	except:
		return 1

def protectedLog(value):
	try:
		return math.log10(value)
	except:
		return 1    

def protectedSqrt(value):
	try:
		return math.sqrt(value)
	except:
		return 1  

def invertSignal(val):
	return -val


def neutralRange(inferior, superior):
	"""
		TO-DO: change the strategy of this function. See how to handle the constraint and the return value
	"""
	if float(inferior) > float(superior):
		variables.neutral_inferior_range, variables.neutral_superior_range = 0, 0
	else:
		variables.neutral_inferior_range = inferior
		variables.neutral_superior_range = superior

	return 0 # try not be used in other branches of the tree

"""
	Verification functions (string -> boolean)
"""
def hasHashtag(message):
	for word in message.strip().split():
		if word[0] == "#":
			return True

	return False


def hasEmail(message):
	for word in message.strip().split():
		if validate_email(word):
			return True

	return False


def hasURL(message):
	if len(getURLs(message)) > 0:
		return True
	
	return False


def hasEmoticon(message):
	"""
		TO-DO: detect emoticon inside the words - In this version the detection occurs only on separate tokens (split)
	"""
	for word in message.strip().split():
		if (word.replace("'","") in variables.dic_words["emoticon"]["positive"]) or (word.replace("'","") in variables.dic_words["emoticon"]["negative"]):
			return True	

	return False

"""
	String properties functions (string -> numeric)
"""
def messageLength(message):
	return len(message.strip())


def wordCount(message):
	return len(message.split())


def posHashtagCount(message):
	"""
		TO-DO: I'm only checking on 'hashtag' dictionary. Maybe it would be intersting search in all dictionaries
	"""
	if not hasHashtag(message):
		return 0

	return len([word for word in message.strip().split() if word[1:] in variables.dic_words["hashtag"]["positive"] and word.startswith("#")])


def negHashtagCount(message):
	"""
		TO-DO: I'm only checking on 'hashtag' dictionary. Maybe it would be intersting search in all dictionaries
	"""
	if not hasHashtag(message):
		return 0

	return len([word for word in message.strip().split() if word[1:] in variables.dic_words["hashtag"]["negative"] and word.startswith("#")])


def posEmoticonCount(message):
	"""
		Return the # of positive emoticons
	"""
	count = 0
	for word in message.strip().split():
		count = count + 1 if word in variables.dic_words["emoticon"]["positive"] else count
	
	return count


def negEmoticonCount(message):
	"""
		Return the # of negative emoticons
	"""
	count = 0
	for word in message.strip().split():
		count = count + 1 if word in variables.dic_words["emoticon"]["negative"] else count
	
	return count


def posWordCount(message):
	"""
		Return the # of positive words
	"""
	count = 0
	for word in message.strip().split():
		if word in variables.dic_words["all"]["positive"]:
			count += 1
	return count


def negWordCount(message):
	"""
		Return the # of negative words
	"""
	count = 0
	for word in message.strip().split():
		if word in variables.dic_words["all"]["negative"]:
			count += 1
	return count


def negateWords(message):
	""" Check and replace the inverter words of the message. This can be used on polSum functions to increase the polarity

		Args:
			message (str): message to be evaluated

		TO-DO: check if the message was already inverted before

	"""	
	if "insidenoteinverterword" in message:
		return message

	for negate in variables.dic_words["negating"]:
		message = re.sub(r'\b%s\b' % re.escape(negate), "insidenoteinverterword", message)
	
	return message


"""
	Manipulation functions (string -> string)
"""
def removeStopWords(message):
	return ' '.join([word for word in message.strip().split() if word not in variables.STOP_WORDS])


def removeURLs(message):
	return re.sub(r'http\S+', '', message, flags=re.MULTILINE).strip()


def removeLinks(message):
	return  re.sub(r'http\S+', '', message, flags=re.MULTILINE)


def removeAllPonctuation(message):
	return message.translate(str.maketrans('','',string.punctuation.replace("-", "").replace("#", ""))) # keep hyphens


def if_then_else(arg, output_true, output_false):
	if arg: 
		return output_true
	else: 
		return output_false


def boostWords(message):
	""" Check and replace the boost words of the message. This can be used on polSum functions to increase the polarity

		Args:
			message (str): message to be evaluated

		TO-DO: check if the message was already boosted before

	"""
	if "insidenoteboosterword" in message:
		return message

	for booster in variables.dic_words["booster"]:
		message = message.replace(booster, "insidenoteboosterword")
	
	return message


def boostUpper(message):
	""" Check and replace the uppercase words of the message. This can be used on polSum functions to increase the polarity

		Args:
			message (str): message to be evaluated

	"""
	if "insidenoteboosteruppercase" in message:
		return message

	uppers = [word for word in message.split() if word.isupper()]

	for upper in uppers:
		message = message.replace(upper, "insidenoteboosteruppercase " + upper)

	return message


def checkBoosterAndInverter(message, index, polarity):
	"""Check for booster and inverter words

		Args:
			message  (str): message to be evaluatedmessage
			index    (int): index of word on message
			polarity (float): polarity of the word

		Return:
			polarity updated

	"""
	words = message.strip().split()

	for word in words:
		if index > 0 and words[index-1]    == "insidenoteboosterword" and (words[index-2] == "insidenoteinverterword" or words[index-3] == "insidenoteinverterword"):
			return variables.BOOSTER_FACTOR * (polarity * -1)
		
		elif index > 0 and words[index-1]  == "insidenoteinverterword":
			return polarity * -1
		
		elif (index > 0 and words[index-1] == "insidenoteboosterword") or (index < len(words) - 1 and words[index+1] == "insidenoteboosterword" and (words[index-1] != "insidenoteboosterword" or index == 0)):
			return polarity * variables.BOOSTER_FACTOR

		elif (index > 0 and words[index-1] == "insidenoteboosteruppercase") or (index < len(words) - 1 and words[index+1] == "insidenoteboosteruppercase" and (words[index-1] != "insidenoteboosteruppercase" or index == 0)):	
			return polarity * variables.BOOSTER_FACTOR

		else:
			return polarity


def hashtagPolSum(message):
	"""Calc the polarity sum of the hashtags of the message

		Args:
			message (str): message to be evaluated

		Return 
			hashtag polarity sum

	"""
	polSum = 0
	for word in message.strip().split():
		p = [float(variables.dic_words["hashtag"]["positive"][w]) for w in variables.dic_words["hashtag"]["positive"] if word == w]
		n = [float(variables.dic_words["hashtag"]["negative"][w]) for w in variables.dic_words["hashtag"]["negative"] if word == w]

		polSum += sum(p) + sum(n)

	return polSum


def polSum(message):
	"""Calc the polarity sum of the message

		Args:
			message (str): message to be evaluated

		Return:
			polarity sum

	"""
	total_sum = 0

	for word in message.strip().split():
		for dic in variables.DICTIONARIES:
			if variables.use_dic[dic] and variables.dic_loaded[dic]:
				
				p = [float(variables.dic_words[dic.lower()]["positive"][w]) for w in variables.dic_words[dic.lower()]["positive"] if word == w]
				n = [float(variables.dic_words[dic.lower()]["negative"][w]) for w in variables.dic_words[dic.lower()]["negative"] if word == w]

				if len(p) > 0:
					total_sum += float(p[0])

				if len(n) > 0:
					total_sum += float(n[0])

	return total_sum


def polSumAVGWeights(message, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10=0, w11=0):
	"""Calc the weighted polarity sum of the message

		Args:
			message   (str)  : message to be evaluated
			w1 .. w11 (float): weights for the i dictionary 

		The dictionary sequence is: ["liu", "sentiwordnet", "afinn", "vader", "slang", "effect", "semeval2015", "nrc", "gi", "s140", "mpqa"]

		TO-DO: test this function - check if the output is correct
	"""
	total_sum, accumulated_p, accumulated_n, dic_quantity, index = 0, 0, 0, 0, 0
   	
	ws = [w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11] # list of weights (parameters)
	wi, w_sum_p, w_sum_n, p, n = 0, 0, 0, 0, 0

	for word in message.strip().split():
		for dic in variables.DICTIONARIES:
			if variables.use_dic[dic] and variables.dic_loaded[dic] and ws[wi] != 0:
				
				if word in variables.dic_words[dic.lower()]["positive"]:
					p = float(variables.dic_words[dic.lower()]["positive"][word]) * ws[wi]
				
				elif word in variables.dic_words[dic.lower()]["negative"]:
					n = float(variables.dic_words[dic.lower()]["negative"][word]) * ws[wi]

				# splitted for didact reasons
				accumulated_p += p
				w_sum_p       += ws[wi]

				accumulated_n += n
				w_sum_n       += ws[wi]

			wi += 1
		
		if w_sum_p > 0:
			total_sum += checkBoosterAndInverter(message, index, accumulated_p) / w_sum_p
		
		if w_sum_n > 0:
			total_sum += checkBoosterAndInverter(message, index, accumulated_n) / w_sum_n

		wi, accumulated_p, accumulated_n, w_sum_p, w_sum_n = 0, 0, 0, 0, 0
		index += 1

	return total_sum