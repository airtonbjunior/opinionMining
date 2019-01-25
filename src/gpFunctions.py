""" 
<Author> Airton Bordin Junior
<Email>  airtonbjunior@gmail.com

Federal University of Goias (UFG)
Computer Science Master's Degree


gpFunctions.py
	Functions used by the Genetic Programming Algorithm

"""
import re
import variables


"""
	Aux functions
"""
def getURLs(phrase):
	#improve this Regular Expression to get www.something.com and others
	return re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', phrase)


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


"""
	Verification functions (string -> numeric)
"""
def hasHashtag(message):
	for word in message.strip().split():
		if word[0] == "#":
			return True

	return False


#def hasEmails(message):
#	for word in message.strip().split():
#		if validate_email(word):
#			return True
#
#	return False


def hasURLs(phrase):
	if len(getURLs(phrase)) > 0:
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

	"""
	total_sum, accumulated_p, accumulated_n, dic_quantity, index = 0, 0, 0, 0, 0
   	
	ws = [w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11] # list of weights (parameters)
	wi, w_sum_p, w_sum_n = 0, 0, 0

	#print(str(ws))

	for word in message.strip().split():
		for dic in variables.DICTIONARIES:
			if variables.use_dic[dic] and variables.dic_loaded[dic] and ws[wi] != 0:

				[print("word " + word + " on " + dic + " with value " + variables.dic_words[dic.lower()]["positive"][w] + " [new]") for w in variables.dic_words[dic.lower()]["positive"] if word.strip() == w]
				[print("word " + word + " on " + dic + " with value " + variables.dic_words[dic.lower()]["negative"][w] + " [new]") for w in variables.dic_words[dic.lower()]["negative"] if word.strip() == w]
				#print("word " + word + " on mpqa with the value " + str(variables.dic_negative_mpqa[word]))

				p = [float(variables.dic_words[dic.lower()]["positive"][w]) * float(ws[wi]) for w in variables.dic_words[dic.lower()]["positive"] if word == w]
				n = [float(variables.dic_words[dic.lower()]["negative"][w]) * float(ws[wi]) for w in variables.dic_words[dic.lower()]["negative"] if word == w]

				# splitted for didact reasons - I'll improve this later
				if len(p) > 0:
					accumulated_p += p[0]
					w_sum_p += ws[wi]

				if len(n) > 0:
					accumulated_n += n[0]
					w_sum_n += ws[wi]

			wi += 1
		
		if w_sum_p > 0:
			total_sum += checkBoosterAndInverter(message, index, accumulated_p) / w_sum_p
		
		if w_sum_n > 0:
			total_sum += checkBoosterAndInverter(message, index, accumulated_n) / w_sum_n

		print("total_sum -> " + str(total_sum))

		wi, accumulated_p, accumulated_n, w_sum_p, w_sum_n = 0, 0, 0, 0, 0
		index += 1

	return total_sum