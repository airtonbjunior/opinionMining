""" 
<Author> Airton Bordin Junior
<Email>  airtonbjunior@gmail.com

Federal University of Goias (UFG)
Computer Science Master's Degree


classifier.py
	Main file of the classifier

Genetic Programming lib: DEAP
References: https://github.com/DEAP/deap/blob/08986fc3848144903048c722564b7b1d92db33a1/examples/gp/symbreg.py
			https://github.com/DEAP/deap/blob/08986fc3848144903048c722564b7b1d92db33a1/examples/gp/spambase.py

"""
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from datetime import datetime

import time
import operator
import random
import sys
import variables as var
import numpy as np
from gpFunctions import *
from loadFunctions import *
from sendMail import *
from auxFunctions import *
from evolutionaryFunctions import *

evaluation_acumulated_time = 0

# log time
start = time.time()

pset = gp.PrimitiveSetTyped("MAIN", [str], float)
pset.addPrimitive(operator.add, [float,float], float)
pset.addPrimitive(operator.sub, [float,float], float)
pset.addPrimitive(operator.mul, [float,float], float)
pset.addPrimitive(protectedDiv, [float,float], float)
pset.addPrimitive(posHashtagCount,  [str], float)
pset.addPrimitive(negHashtagCount,  [str], float)
pset.addPrimitive(posEmoticonCount, [str], float)
pset.addPrimitive(negEmoticonCount, [str], float)
pset.addPrimitive(polSum,           [str], float)
pset.addPrimitive(polSumAVG,        [str], float)
pset.addPrimitive(polSumAVGWeights, [str, float, float, float, float, float, float, float, float, float, float, float], float)
pset.addPrimitive(hashtagPolSum,    [str], float)
#pset.addPrimitive(emoticonsPolaritySum,  [str], float)
pset.addPrimitive(posWordCount, [str], float)
pset.addPrimitive(negWordCount, [str], float)
pset.addPrimitive(hasHashtag,   [str], bool)
pset.addPrimitive(hasEmoticon,  [str], bool)
pset.addPrimitive(hasURL,       [str], bool)
pset.addPrimitive(hasEmail,     [str], bool)
pset.addPrimitive(if_then_else, [bool, float, float], float)
pset.addPrimitive(removeStopWords,      [str], str)
pset.addPrimitive(removeAllPonctuation, [str], str)
pset.addPrimitive(negateWords, [str], str)
pset.addPrimitive(boostWords,  [str], str)
pset.addPrimitive(boostUpper,  [str], str)
pset.addPrimitive(neutralRange, [float, float], float)
pset.addPrimitive(messageLength,[str], float)
pset.addPrimitive(wordCount,    [str], float)
pset.addPrimitive(removeLinks,  [str], str)
#pset.addPrimitive(hasDate,     [str], bool)
#pset.addPrimitive(removeEllipsis, [str], str)
#pset.addPrimitive(stemmingText, [str], str)

pset.addTerminal(True,  bool)
pset.addTerminal(False, bool)
pset.addTerminal(0.0,   float)
pset.addEphemeralConstant("rand",  lambda: random.uniform(0,  2), float)
pset.addEphemeralConstant("rand2", lambda: random.uniform(-2, 2), float)
pset.renameArguments(ARG0='x')

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=var.TREE_MIN_HEIGHT, max_=var.TREE_MAX_HEIGHT)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

iterate_count, generation_count, best_of_generation = 1, 1, 0

def evalMessages(individual):
	""" GP evaluation function
			The individual is evaluated using all messages
			The fitness is the F1-measure of the positive/negative messages

		Args:
			individual: individual to be evaluated

		Return:
			Evaluation (fitness)
	"""
	start = time.time()
	global iterate_count
	global generation_count
	global best_of_generation
	new_generation = False

	var.NEUTRAL['range']['inferior'], var.NEUTRAL['range']['superior'] = 0, 0

	# Check max unchanged generations
	if var.GENERATIONS_UNCHANGED >= var.MAX_UNCHANGED_GENERATIONS:
		if(var.GEN_UNCHANGED_REACH_MSG == False):
			print("[Max unchanged generations (" + str(var.MAX_UNCHANGED_GENERATIONS) + ") reached on generation " + str(generation_count) + "]")
		var.GEN_UNCHANGED_REACH_MSG = True
		return 0,

	# Log the number of each individual
	if iterate_count <= var.POPULATION:
		print("[individual " + str(iterate_count) + " of the generation " + str(generation_count) + "]")
		iterate_count += 1
	else:
		generation_count += 1
		var.HISTORY['fitness']['per_generation'].append(best_of_generation)
		print("\n[new generation][start generation " + str(generation_count) + "]\n")
		iterate_count, new_generation, best_of_generation = 1, True, 0

	global evaluation_acumulated_time
	correct_evaluations = 0
	fitnessReturn = 0

	is_positive, is_negative, is_neutral = 0, 0, 0
	conf_matrix = {'true_positive': 0, 'true_negative': 0, 'true_neutral': 0, 'false_positive': 0, 'false_negative': 0, 'false_neutral': 0}
	precision   = {'positive': 0, 'negative': 0, 'neutral': 0, 'avg': 0}
	recall      = {'positive': 0, 'negative': 0, 'neutral': 0, 'avg': 0}
	f1          = {'positive': 0, 'negative': 0, 'neutral': 0, 'avg': 0, 'avg_pn': 0}
	accuracy    = 0
	func_value  = 0

	# Constraint controls 
	breaked, fitness_decreased, double_decreased = False, False, False

	# Transform the tree expression in a callable function
	func = toolbox.compile(expr=individual)

	#MESSAGES_TO_EVALUATE = var.MESSAGES['train']
	MESSAGES_TO_EVALUATE = np.array(var.MESSAGES['train'])

	# Main loop - iterate through the messages
	for index, item in enumerate(MESSAGES_TO_EVALUATE):
	
		if var.CONSTRAINT['active']:
			# massive function - more than massive_functions_max
			if (str(individual).count(var.CONSTRAINT['massive']['function']) > var.CONSTRAINT['massive']['max']) and (var.CONSTRAINT['massive']['active']):
				logConstraint(start, 'massive_function_more_than_one')
				breaked = True
				break

			count_neutral_range = str(individual).count("neutralRange")
			
			# neutralRange - more than one neutralRange
			if count_neutral_range > 1:
				logConstraint(start, 'neutral_range_more_than_one')
				breaked = True
				break            

			# neutralRange constraints
			if(var.CONSTRAINT['neutral_range']['active']) and (count_neutral_range == 0) and (index == 0):
				logConstraint(start, 'neutral_range_inexistent')
				if fitness_decreased:
					double_decreased = True
				fitness_decreased = True

			# root function - functions root_functions (if_then_else hardcoded temporarily)
			if (not str(individual).startswith(var.CONSTRAINT['root']['function']) and not str(individual).startswith("if_then_else")) and var.CONSTRAINT['root']['active'] and index == 0:
				logConstraint(start, 'root_function')
				if fitness_decreased:
					double_decreased = True            
				fitness_decreased = True


		# Check cicle limit
		if var.CICLES_UNCHANGED >= var.MAX_UNCHANGED_CICLES:
			breaked = True
			break

		# Log each new cicle
		if index == 0 and var.LOG['all_each_cicle']:
			print("\n[New cicle]: " + str(len(MESSAGES_TO_EVALUATE)) + " phrases to evaluate [" + str(var.POSITIVE_MESSAGES) + " positives, " + str(var.NEGATIVE_MESSAGES) + " negatives and " + str(var.NEUTRAL_MESSAGES) + " neutrals]")

		try:
			func_value = float(func(MESSAGES_TO_EVALUATE[index]['message'])) # run the model (individual)

			if MESSAGES_TO_EVALUATE[index]['num_label'] > 0:
				if  func_value > var.NEUTRAL['range']['superior']:
					correct_evaluations += 1 
					is_positive   += 1
					conf_matrix['true_positive'] += 1
				else:
					if func_value >= var.NEUTRAL['range']['inferior'] and func_value <= var.NEUTRAL['range']['superior']:
						conf_matrix['false_neutral'] += 1
					elif func_value < var.NEUTRAL['range']['inferior']:
						conf_matrix['false_negative'] += 1

			elif MESSAGES_TO_EVALUATE[index]['num_label'] < 0:
				if func_value < var.NEUTRAL['range']['inferior']:
					correct_evaluations += 1 
					is_negative   += 1
					conf_matrix['true_negative'] += 1
				else:
					if func_value >= var.NEUTRAL['range']['inferior'] and func_value <= var.NEUTRAL['range']['superior']:
						conf_matrix['false_neutral'] += 1
					elif func_value > var.NEUTRAL['range']['superior']:
						conf_matrix['false_positive'] += 1

			elif MESSAGES_TO_EVALUATE[index]['num_label'] == 0:
				if func_value >= var.NEUTRAL['range']['inferior'] and func_value <= var.NEUTRAL['range']['superior']:
					correct_evaluations += 1 
					is_neutral   += 1
					conf_matrix['true_neutral'] += 1
				else:
					if func_value < var.NEUTRAL['range']['inferior']:
						conf_matrix['false_negative'] += 1
					elif func_value > var.NEUTRAL['range']['superior']:
						conf_matrix['false_positive'] += 1

		except Exception as e: 
			print(e)
			continue

		# LOGS
		if var.LOG['all_messages']:
			print("[phrase]: "    + MESSAGES_TO_EVALUATE[index]['message'])
			print("[value]: "     + str(MESSAGES_TO_EVALUATE[index]['label']))
			print("[calculated]:" + func_value)

	if conf_matrix['true_positive'] + conf_matrix['false_positive'] + conf_matrix['true_negative'] + conf_matrix['false_negative'] > 0:
		accuracy = (conf_matrix['true_positive'] + conf_matrix['true_negative'] + conf_matrix['true_neutral']) / (conf_matrix['true_positive'] + conf_matrix['false_positive'] + conf_matrix['true_negative'] + conf_matrix['false_negative'] + conf_matrix['true_neutral'] + conf_matrix['false_neutral'])

	# PRECISION
	if conf_matrix['true_positive'] + conf_matrix['false_positive'] > 0:
		precision['positive'] = conf_matrix['true_positive'] / (conf_matrix['true_positive'] + conf_matrix['false_positive'])

	if conf_matrix['true_negative'] + conf_matrix['false_negative'] > 0:
		precision['negative'] = conf_matrix['true_negative'] / (conf_matrix['true_negative'] + conf_matrix['false_negative'])
	
	if conf_matrix['true_neutral'] + conf_matrix['false_neutral'] > 0:
		precision['neutral'] = conf_matrix['true_neutral'] / (conf_matrix['true_neutral'] + conf_matrix['false_neutral'])
	# PRECISION

	# RECALL
	if var.POSITIVE_MESSAGES > 0:
		recall['positive'] = conf_matrix['true_positive'] / var.POSITIVE_MESSAGES

	if var.NEGATIVE_MESSAGES > 0:
		recall['negative'] = conf_matrix['true_negative'] / var.NEGATIVE_MESSAGES

	if var.NEUTRAL_MESSAGES > 0:
		recall['neutral'] = conf_matrix['true_neutral'] / var.NEUTRAL_MESSAGES
	# RECALL

	# F1
	if precision['positive'] + recall['positive'] > 0:
		f1['positive'] = 2 * (precision['positive'] * recall['positive']) / (precision['positive'] + recall['positive'])

	if precision['negative'] + recall['negative'] > 0:
		f1['negative'] = 2 * (precision['negative'] * recall['negative']) / (precision['negative'] + recall['negative'])

	if precision['neutral'] + recall['neutral'] > 0:
		f1['neutral'] = 2 * (precision['neutral'] * recall['neutral']) / (precision['neutral'] + recall['neutral'])            
	# F1

	# MEANS (PRECISION, RECALL, F1)
	precision['avg'] = (precision['positive'] + precision['negative'] + precision['neutral']) / 3
	recall['avg']    = (recall['positive'] + recall['negative'] + recall['neutral']) / 3
	f1['avg']        = (f1['positive'] + f1['negative'] + f1['neutral']) / 3
	f1['avg_pn']     = (f1['positive'] + f1['negative']) / 2

	saveBestValues(individual, accuracy, precision, recall, f1)

	# FITNESS
	fitnessReturn = f1['avg_pn']
	if fitness_decreased:
		fitnessReturn -= fitnessReturn * var.CONSTRAINT['root']['decrease_rate'] # 80% of the original value
	if double_decreased:
		fitnessReturn -= fitnessReturn * var.CONSTRAINT['root']['decrease_rate'] # Again

	if var.BEST['fitness'] < fitnessReturn:
		saveBestFitness(var.BEST_INDIVIDUAL, individual, fitnessReturn, generation_count, is_positive, is_negative, is_neutral)
		is_positive, is_negative, is_neutral = 0, 0, 0
	else:
		var.CICLES_UNCHANGED += 1
		if new_generation:
			var.GENERATIONS_UNCHANGED += 1

	if not new_generation and best_of_generation < fitnessReturn:
		best_of_generation = fitnessReturn

	var.HISTORY['fitness']['all'].append(fitnessReturn)

	if var.LOG['partial_results'] and not breaked:        
		logCicleValues(start, correct_evaluations, individual, accuracy, precision, recall, f1, fitnessReturn, conf_matrix)

	evaluation_acumulated_time += time.time() - start
	
	return fitnessReturn,


# evaluation function with folds
def evalSymbRegTweetsFromSemeval_folds(individual):
	start = time.time()
	global iterate_count
	global generation_count
	global best_of_generation
	new_generation = False

	# Check max unchanged generations
	if var.GENERATIONS_UNCHANGED >= var.MAX_UNCHANGED_GENERATIONS:
		if(var.GEN_UNCHANGED_REACH_MSG == False):
			print("[Max unchanged generations (" + str(var.MAX_UNCHANGED_GENERATIONS) + ") reached on generation " + str(generation_count) + "]")
		var.GEN_UNCHANGED_REACH_MSG = True
		return 0,

	# Log the number of each individual
	if iterate_count <= var.POPULATION:
		print("[individual " + str(iterate_count) + " of the generation " + str(generation_count) + "]")
		iterate_count += 1
	else:
		generation_count += 1
		iterate_count = 1
		var.HISTORY['fitness']['per_generation'].append(best_of_generation)
		print("\n[new generation][start generation " + str(generation_count) + "]\n")
		new_generation = True
		best_of_generation = 0

	global evaluation_acumulated_time
	correct_evaluations = 0

	fitnessReturn = 0
	accuracy = 0
	is_positive, is_negative, is_neutral = 0, 0, 0
	conf_matrix['true_positive'], conf_matrix['true_negative'], conf_matrix['true_neutral'], conf_matrix['false_positive'], conf_matrix['false_negative'], conf_matrix['false_neutral'] = 0, 0, 0, 0, 0, 0
	precision_positive, precision_negative, precision_neutral, precision_avg                  = 0, 0, 0, 0
	recall['positive'], recall['negative'], recall['neutral'], recall_avg                              = 0, 0, 0, 0
	f1['positive'], f1['negative'], f1['neutral'], f1_avg, f1['avg_pn']                    = 0, 0, 0, 0, 0
	func_value = 0

	# Constraint controls 
	breaked, fitness_decreased, double_decreased = False, False, False

	func = toolbox.compile(expr=individual)

	TWEETS_TO_EVALUATE, TWEETS_SCORE_TO_EVALUATE = var.tweets_semeval, var.tweets_semeval_score

	indexes = list(range(len(TWEETS_TO_EVALUATE)))
	#chunks = createIndexChunks(indexes, 10)
	chunks = createRandomIndexChunks(indexes, 10)

	fitness_list = []

	fold_index = 0

	for fold in chunks:
		correct_evaluations, fitnessReturn, accuracy = 0, 0, 0
		is_positive, is_negative, is_neutral         = 0, 0, 0
		conf_matrix['true_positive'], conf_matrix['true_negative'], conf_matrix['true_neutral'], conf_matrix['false_positive'], conf_matrix['false_negative'], conf_matrix['false_neutral'] = 0, 0, 0, 0, 0, 0
		precision_positive, precision_negative, precision_neutral, precision_avg                  = 0, 0, 0, 0
		recall['positive'], recall['negative'], recall['neutral'], recall_avg                              = 0, 0, 0, 0
		f1['positive'], f1['negative'], f1['neutral'], f1_avg, f1['avg_pn']                    = 0, 0, 0, 0, 0
		first = True
		# Constraint controls 
		breaked, fitness_decreased, double_decreased = False, False, False
		
		# Constraints
		# Constraint 1: massive function - more than massive_functions_max
		if (str(individual).count(var.massive_function) > var.massive_functions_max) and var.massive_functions_constraint:
			print("\n[CONSTRAINT][more than " + str(var.massive_functions_max) + " massive(s) function(s)][bad individual][fitness zero]\n")
			if var.LOG['times']:
				print("[cicle ends after " + str(format(time.time() - start, '.3g')) + " seconds]")     
			print("-----------------------------")
			print("\n") 
			breaked = True
			break

		count_neutral_range = 0
		count_neutral_range = str(individual).count("neutralRange")
		
		# Constraint 2: neutralRange - more than one neutralRange
		if count_neutral_range > 1:
			print("\n[CONSTRAINT][more than one neutralRange function][bad individual][fitness zero]\n")
			if var.LOG['times']:
				print("[cicle ends after " + str(format(time.time() - start, '.3g')) + " seconds]")     
			print("-----------------------------")
			print("\n") 
			breaked = True
			break

		# neutralRange constraints
		if(var.neutral_range_constraint):      
			# Constraint 4: neutralRange - function does not exist in the model
			if count_neutral_range == 0:
				if fold_index == 0: #log only on first index
					print("\n[CONSTRAINT][model does not have neutralRange function][fitness decreased in " + str(var.CONSTRAINT['root']['decrease_rate'] * 100) + "%]\n")
				if fitness_decreased:
					double_decreased = True
				fitness_decreased = True

		# Constraint 5: root function - functions root_functions (if_then_else hardcoded temporarily)
		#if (not str(individual).startswith(var.root_function) and not str(individual).startswith("if_then_else")) and var.root_constraint:# and fold_index == 0:
		if (not str(individual).startswith(var.root_function)) and var.root_constraint:# and fold_index == 0:
			if fold_index == 0: #log only on first index
				print("[CONSTRAINT][root node is not " + var.root_function + "][fitness decreased in " + str(var.CONSTRAINT['root']['decrease_rate'] * 100) + "%]\n")
			if fitness_decreased:
				double_decreased = True            
			fitness_decreased = True


		for index in fold:
			#breaked = False
			#breaked, fitness_decreased, double_decreased = False, False, False

			#if(first):
			#    print("[individual] " + str(individual))
			# Log each new cicle
			if first:
				if var.LOG['all_each_cicle']:
					print("\n[New cicle]: " + str(len(TWEETS_TO_EVALUATE)) + " phrases to evaluate [" + str(var.POSITIVE_MESSAGES) + " positives, " + str(var.NEGATIVE_MESSAGES) + " negatives and " + str(var.NEUTRAL_MESSAGES) + " neutrals]")
					first = False

			try:
				func_value = float(func(TWEETS_TO_EVALUATE[index]))
				
				if float(TWEETS_SCORE_TO_EVALUATE[index]) > 0:
					is_positive   += 1
					if  func_value > var.NEUTRAL['range']['superior']:
						correct_evaluations += 1 
						conf_matrix['true_positive'] += 1
					else:
						if func_value >= var.NEUTRAL['range']['inferior'] and func_value <= var.NEUTRAL['range']['superior']:
							conf_matrix['false_neutral'] += 1
						elif func_value < var.NEUTRAL['range']['inferior']:
							conf_matrix['false_negative'] += 1

				elif float(TWEETS_SCORE_TO_EVALUATE[index]) < 0:
					is_negative   += 1
					if func_value < var.NEUTRAL['range']['inferior']:
						correct_evaluations += 1 
						conf_matrix['true_negative'] += 1
					else:
						if func_value >= var.NEUTRAL['range']['inferior'] and func_value <= var.NEUTRAL['range']['superior']:
							conf_matrix['false_neutral'] += 1
						elif func_value > var.NEUTRAL['range']['superior']:
							conf_matrix['false_positive'] += 1

				elif float(TWEETS_SCORE_TO_EVALUATE[index]) == 0:
					is_neutral   += 1
					if func_value >= var.NEUTRAL['range']['inferior'] and func_value <= var.NEUTRAL['range']['superior']:
						correct_evaluations += 1 
						conf_matrix['true_neutral'] += 1
					else:
						if func_value < var.NEUTRAL['range']['inferior']:
							conf_matrix['false_negative'] += 1
						elif func_value > var.NEUTRAL['range']['superior']:
							conf_matrix['false_positive'] += 1

			except Exception as e: 
				print(e)
				continue

			# LOGS
			if var.LOG['all_messages']:
				print("[phrase]: " + TWEETS_TO_EVALUATE[index])
				print("[value]: " + str(TWEETS_SCORE_TO_EVALUATE[index]))
				print("[calculated]:" + func_value)

		if breaked:
			break

		if conf_matrix['true_positive'] + conf_matrix['false_positive'] + conf_matrix['true_negative'] + conf_matrix['false_negative'] > 0:
			accuracy = (conf_matrix['true_positive'] + conf_matrix['true_negative'] + conf_matrix['true_neutral']) / (conf_matrix['true_positive'] + conf_matrix['false_positive'] + conf_matrix['true_negative'] + conf_matrix['false_negative'] + conf_matrix['true_neutral'] + conf_matrix['false_neutral'])

		if accuracy > var.best_accuracy:
			var.best_accuracy = accuracy 

		# Begin PRECISION
		if conf_matrix['true_positive'] + conf_matrix['false_positive'] > 0:
			precision_positive = conf_matrix['true_positive'] / (conf_matrix['true_positive'] + conf_matrix['false_positive'])
			if precision_positive > var.best_precision_positive:
				var.best_precision_positive = precision_positive


		if conf_matrix['true_negative'] + conf_matrix['false_negative'] > 0:
			precision_negative = conf_matrix['true_negative'] / (conf_matrix['true_negative'] + conf_matrix['false_negative'])
			if precision_negative > var.best_precision_negative:
				var.best_precision_negative = precision_negative
		

		if conf_matrix['true_neutral'] + conf_matrix['false_neutral'] > 0:
			precision_neutral = conf_matrix['true_neutral'] / (conf_matrix['true_neutral'] + conf_matrix['false_neutral'])
			if precision_neutral > var.best_precision_neutral:
				var.best_precision_neutral = precision_neutral
		# End PRECISION

		# Begin RECALL
		if is_positive > 0:
			recall['positive'] = conf_matrix['true_positive'] / is_positive
			if recall['positive'] > var.best_recall_positive:
				var.best_recall_positive = recall['positive']


		if is_negative > 0:
			recall['negative'] = conf_matrix['true_negative'] / is_negative
			if recall['negative'] > var.best_recall_negative:
				var.best_recall_negative = recall['negative']

		if is_neutral > 0:
			recall['neutral'] = conf_matrix['true_neutral'] / is_neutral
			if recall['neutral'] > var.best_recall_neutral:
				var.best_recall_neutral = recall['neutral']
		# End RECALL

		# Begin F1
		if precision_positive + recall['positive'] > 0:
			f1['positive'] = 2 * (precision_positive * recall['positive']) / (precision_positive + recall['positive'])
			if f1['positive'] > var.best_f1_positive:
				var.best_f1_positive = f1['positive']


		if precision_negative + recall['negative'] > 0:
			f1['negative'] = 2 * (precision_negative * recall['negative']) / (precision_negative + recall['negative'])        
			if f1['negative'] > var.best_f1_negative:
				var.best_f1_negative = f1['negative']

		if precision_neutral + recall['neutral'] > 0:
			f1['neutral'] = 2 * (precision_neutral * recall['neutral']) / (precision_neutral + recall['neutral'])        
			if f1['neutral'] > var.best_f1_neutral:
				var.best_f1_neutral = f1['neutral']            
		# End F1

		# Precision, Recall and f1 means
		precision_avg = (precision_positive + precision_negative + precision_neutral) / 3
		if precision_avg > var.best_precision_avg:
			var.best_precision_avg = precision_avg
			var.best_precision_avg_function = str(individual)

		recall_avg = (recall['positive'] + recall['negative'] + recall['neutral']) / 3
		if recall_avg > var.best_recall_avg:
			var.best_recall_avg = recall_avg
			var.best_recall_avg_function = str(individual)

		f1_avg = (f1['positive'] + f1['negative'] + f1['neutral']) / 3
		if f1_avg > var.best_f1_avg:
			var.best_f1_avg = f1_avg
			var.best_f1_avg_function = str(individual)

		f1['avg_pn'] = (f1['positive'] + f1['negative']) / 2
		if f1['avg_pn'] > var.best_f1_positive_negative_avg:
			var.best_f1_positive_negative_avg = f1['avg_pn']

		
		fitnessReturn = f1['avg_pn']
		orig = 0
		if fitness_decreased:
			orig = fitnessReturn
			fitnessReturn -= fitnessReturn * var.CONSTRAINT['root']['decrease_rate'] # 80% of the original value
			print("  [INDIVIDUAL]  " + str(individual))
			print("     [POS: " + str(is_positive) + "] [NEG: " + str(is_negative) + "] [NEU: " + str(is_neutral) +"]")
			print("       [FITNESS DECREASED] [ORIGINAL: " + str(orig) + "] [DECREASED: " + str(fitnessReturn) + "]")
		orig = 0
		if double_decreased:
			orig = fitnessReturn
			fitnessReturn -= fitnessReturn * var.CONSTRAINT['root']['decrease_rate'] # Again
			print("         [FITNESS DOUBLE DECREASED] [ORIGINAL: " + str(orig) + "] [DECREASED: " + str(fitnessReturn) + "]")

		fitness_list.append(fitnessReturn)
		fold_index += 1

	fitnessReturn = sum(fitness_list)/len(chunks)
	
	print("[fitness list] "     + str(fitness_list))
	print("[fitness sum] "      + str(sum(fitness_list)))
	print("[number of chunks] " + str(len(chunks)))
	print("[avg fitness] "      + str(fitnessReturn) + "\n")

	if var.BEST['fitness'] < fitnessReturn:
		if var.BEST['fitness'] != 0:
			with open(var.BEST_INDIVIDUAL, 'w') as f:
				f.write(str(individual))
				f.write("\n\n# Generation -> " + str(generation_count))
				f.write("\n# Neutral Range -> [" + str(var.NEUTRAL['range']['inferior']) + ", " + str(var.NEUTRAL['range']['superior']) + "]")
			var.HISTORY['fitness']['best'].append(var.BEST['fitness'])
		var.BEST['fitness'] = fitnessReturn
		#var.best_fitness_history_dict[generation_count] = fitnessReturn
		var.fitness_positive = is_positive
		var.fitness_negative = is_negative
		var.fitness_neutral  = is_neutral
		is_positive, is_negative, is_neutral, var.CICLES_UNCHANGED, var.GENERATIONS_UNCHANGED = 0, 0, 0, 0, 0
	else:
		var.CICLES_UNCHANGED += 1
		if new_generation:
			var.GENERATIONS_UNCHANGED += 1

	if not new_generation and best_of_generation < fitnessReturn:
		best_of_generation = fitnessReturn

	var.HISTORY['fitness']['all'].append(fitnessReturn)


	# LOGS
	if var.LOG['partial_results'] and not breaked:# and not var.calling_by_ag_file: 
		if var.LOG['all_each_cicle']:
			print("[correct evaluations] " + str(correct_evaluations))
			print('{message: <{width}}'.format(message="[accuracy] ", width=18)   + " -> " + str(round(accuracy, 3)))
			print('{message: <{width}}'.format(message="[precision] ", width=18)  + " -> " + "[positive]: " + '{message: <{width}}'.format(message=str(round(precision_positive, 3)), width=6) + " " + "[negative]: " + '{message: <{width}}'.format(message=str(round(precision_negative, 3)), width=6) + " " + "[neutral]: " + '{message: <{width}}'.format(message=str(round(precision_neutral, 3)), width=6) + " " + "[avg]: " + '{message: <{width}}'.format(message=str(round(precision_avg, 3)), width=6))
			print('{message: <{width}}'.format(message="[recall] ", width=18)     + " -> " + "[positive]: " + '{message: <{width}}'.format(message=str(round(recall['positive'], 3)), width=6) + " " + "[negative]: " + '{message: <{width}}'.format(message=str(round(recall['negative'], 3)), width=6) + " " + "[neutral]: " + '{message: <{width}}'.format(message=str(round(recall['neutral'], 3)), width=6) + " " + "[avg]: " + '{message: <{width}}'.format(message=str(round(recall_avg, 3)), width=6))
			print('{message: <{width}}'.format(message="[f1] ", width=18)         + " -> " + "[positive]: " + '{message: <{width}}'.format(message=str(round(f1['positive'], 3)), width=6) + " " + "[negative]: " + '{message: <{width}}'.format(message=str(round(f1['negative'], 3)), width=6) + " " + "[neutral]: " + '{message: <{width}}'.format(message=str(round(f1['neutral'], 3)), width=6) + " " + "[avg]: " + '{message: <{width}}'.format(message=str(round(f1_avg, 3)), width=6))
			print('{message: <{width}}'.format(message="[f1 SemEval] ", width=18) + " -> " + str(round(f1['avg_pn'], 3)))
		
		print('{message: <{width}}'.format(message="[fitness] ", width=18) + " -> "      + str(round(fitnessReturn, 5)) + " ****")
		print('{message: <{width}}'.format(message="[best fitness] ", width=18) + " -> " + str(round(var.BEST['fitness'], 5)))
		
		print('{message: <{width}}'.format(message="[confusion matrix]", width=18) + " -> " + "[conf_matrix['true_positive']]: " + str(conf_matrix['true_positive']) + " " + "[conf_matrix['false_positive']]: " + str(conf_matrix['false_positive']) + " " + "[conf_matrix['true_negative']]: " + str(conf_matrix['true_negative']) + " " + "[conf_matrix['false_negative']]: " + str(conf_matrix['false_negative']) + " " + "[conf_matrix['true_neutral']]: " + str(conf_matrix['true_neutral']) + " " + "[conf_matrix['false_neutral']]: " + str(conf_matrix['false_neutral']) + "\n")

		if var.LOG['all_each_cicle']:
			print('{message: <{width}}'.format(message="[cicles unmodified]", width=24)  + " -> " + str(var.CICLES_UNCHANGED))
		
		print('{message: <{width}}'.format(message="[generations unmodified]", width=24) + " -> " + str(var.GENERATIONS_UNCHANGED))
		print("[function]: " + str(individual))
		
		if var.LOG['times']:
			print("[cicle ends after " + str(format(time.time() - start, '.3g')) + " seconds]")     
		
		print("-----------------------------\n")
	# LOGS

	evaluation_acumulated_time += time.time() - start
	
	return fitnessReturn,


if var.USE_TRAIN_FOLDS:
	toolbox.register("evaluate", evalSymbRegTweetsFromSemeval_folds)
else:
	toolbox.register("evaluate", evalMessages)

toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genHalfAndHalf, min_=1, max_=3)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.register("mutateEphemeral", gp.mutEphemeral)
toolbox.decorate("mate",   gp.staticLimit(key=operator.attrgetter("height"), max_value=10)) # TO-DO: create a variable to max_value
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))


def main():
	"""
		Main function - call the myEaSimple and print some result logs
	"""
	start = time.time()
	global evaluation_acumulated_time
	random.seed()

	pop = toolbox.population(n=var.POPULATION)
	hof = tools.HallOfFame(var.HOF)

	pop, log = myEaSimple(pop, toolbox, var.CROSSOVER, var.MUTATION, var.GENERATIONS, stats=False,
								   halloffame=hof, verbose=False)

	logFinalResults(hof[0])
	plotResults()

	print("[evaluation function consumed " + str(format(evaluation_acumulated_time, '.3g')) + " seconds]")
	print("[main function ended][" + str(format(time.time() - start, '.3g')) + " seconds]\n")
	
	var.MODEL['bests'].append(hof[0])
	
	if not var.SAVE['only_best']:
		for i in range(var.SAVE['save_n_individuals']):
			var.MODEL['others'].append(hof[i+1])

	return pop, log, hof


if __name__ == "__main__":
	print("[opinionMining v" + var.SYSTEM_VERSION + "][starting classifier module]")
	parameters = str(var.CROSSOVER) + " crossover, " + str(var.MUTATION) + " mutation, " + str(var.POPULATION) + " population, " + str(var.GENERATIONS) + " generation"

	loadDictionaries()
	loadTrainMessages()

	with open(var.TRAIN_RESULTS, 'a') as f:
		f.write("[PARAMS]: " + parameters + "\n" + "[DICTIONARIES]: " + str(var.DIC_LOADED['total']) + "\n\n")
	
	# Main loop - will call the main function <TOTAL_MODEL> times
	for i in range(var.TOTAL_MODELS):
		main()
		saveModel(var.TRAIN_RESULTS)
		send_mail(i+1, var.TOTAL_MODELS, var.POPULATION, var.GENERATIONS, parameters)
		resetVariables()

		iterate_count, generation_count, best_of_generation = 1, 1, 0


logEndSystem(var.TRAIN_RESULTS, start)