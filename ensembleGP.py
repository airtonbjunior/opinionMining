# Airton Bordin Junior
# airtonbjunior@gmail.com
# Federal University of Goias (UFG)
# Computer Science Master's Degree
#
# Genetic Programming lib: DEAP

import time
import operator
import random
import sys

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

import matplotlib.pyplot as plt

import variables
from functions import *
from sendMail import *

# log time
start = time.time()
print("[starting ensembleGP module]\n")


# See this note to create the functions
#In particular, the functions chosen to better combine the classifiers composing the ensemble 
#are non-trainable functions and are listed in the following: average, weighted average, multiplication, 
#maximum and median. They can be applied to a different number of classifiers, i.e. each function is 
#replicated with a different arity, typically from 2 to 5. 

def convertResult(string_result):
	if getPosMax(string_result) == 0:
		return variables.superior_range_gp_ensemble + 1
	elif getPosMax(string_result) == 1:
		return variables.inferior_range_gp_ensemble - 1
	elif getPosMax(string_result) == 2:
		return random.uniform(variables.inferior_range_gp_ensemble, variables.superior_range_gp_ensemble)
	else:
		return variables.superior_range_gp_ensemble + 1 ### change this

def safeDiv(left, right):
	try:
		return left / right
	except ZeroDivisionError:
		return 0

def neg(value):
	return -value

def getSum(c1, c2):
	c1p = float(c1.split('\t')[0])
	c1n = float(c1.split('\t')[1])
	c1u = float(c1.split('\t')[2])

	c2p = float(c2.split('\t')[0])
	c2n = float(c2.split('\t')[1])
	c2u = float(c2.split('\t')[2])

	return str(auxSum(c1p, c2p)) + "\t" + str(auxSum(c1n, c2n)) + "\t" + str(auxSum(c1u, c2u))

def getSum3(c1, c2, c3):
	c1p = float(c1.split('\t')[0])
	c1n = float(c1.split('\t')[1])
	c1u = float(c1.split('\t')[2])

	c2p = float(c2.split('\t')[0])
	c2n = float(c2.split('\t')[1])
	c2u = float(c2.split('\t')[2])

	c3p = float(c3.split('\t')[0])
	c3n = float(c3.split('\t')[1])
	c3u = float(c3.split('\t')[2])	

	return str(auxSum(c1p, c2p, c3p)) + "\t" + str(auxSum(c1n, c2n, c3n)) + "\t" + str(auxSum(c1u, c2u, c3u))	


def getMax(c1, c2):
	c1p = float(c1.split('\t')[0])
	c1n = float(c1.split('\t')[1])
	c1u = float(c1.split('\t')[2])

	c2p = float(c2.split('\t')[0])
	c2n = float(c2.split('\t')[1])
	c2u = float(c2.split('\t')[2])

	return str(max(c1p, c2p)) + "\t" + str(max(c1n, c2n)) + "\t" + str(max(c1u, c2u))


def getMin(c1, c2):
	c1p = float(c1.split('\t')[0])
	c1n = float(c1.split('\t')[1])
	c1u = float(c1.split('\t')[2])

	c2p = float(c2.split('\t')[0])
	c2n = float(c2.split('\t')[1])
	c2u = float(c2.split('\t')[2])

	return str(min(c1p, c2p)) + "\t" + str(min(c1n, c2n)) + "\t" + str(min(c1u, c2u))

def getMax3(c1, c2, c3):
	c1p = float(c1.split('\t')[0])
	c1n = float(c1.split('\t')[1])
	c1u = float(c1.split('\t')[2])

	c2p = float(c2.split('\t')[0])
	c2n = float(c2.split('\t')[1])
	c2u = float(c2.split('\t')[2])

	c3p = float(c3.split('\t')[0])
	c3n = float(c3.split('\t')[1])
	c3u = float(c3.split('\t')[2])	

	return str(max(c1p, c2p, c3p)) + "\t" + str(max(c1n, c2n, c3n)) + "\t" + str(max(c1u, c2u, c3u))

def getMin3(c1, c2, c3):
	c1p = float(c1.split('\t')[0])
	c1n = float(c1.split('\t')[1])
	c1u = float(c1.split('\t')[2])

	c2p = float(c2.split('\t')[0])
	c2n = float(c2.split('\t')[1])
	c2u = float(c2.split('\t')[2])

	c3p = float(c3.split('\t')[0])
	c3n = float(c3.split('\t')[1])
	c3u = float(c3.split('\t')[2])	

	return str(min(c1p, c2p, c3p)) + "\t" + str(min(c1n, c2n, c3n)) + "\t" + str(min(c1u, c2u, c3u))


def getMajority3(c1, c2, c3):
	vp, vn, vu = 0, 0, 0
	p = getPosMax(c1)
	if p == 0:
		vp += 1
	elif p == 1:
		vn += 1
	elif p == 2:
		vu += 1

	p = getPosMax(c2)
	if p == 0:
		vp += 1
	elif p == 1:
		vn += 1
	elif p == 2:
		vu += 1

	p = getPosMax(c3)
	if p == 0:
		vp += 1
	elif p == 1:
		vn += 1
	elif p == 2:
		vu += 1		

	if vp > vn and vp > vu:
		return "1	0	0"
	elif vn > vp and vn > vu:
		return "0	1	0"
	elif vu > vp and vu > vn:
		return "0	0	1"
	else:
		return "0	0	0"



def getMajority(c1, c2):
	vp, vn, vu = 0, 0, 0
	p = getPosMax(c1)
	if p == 0:
		vp += 1
	elif p == 1:
		vn += 1
	elif p == 2:
		vu += 1

	p = getPosMax(c2)
	if p == 0:
		vp += 1
	elif p == 1:
		vn += 1
	elif p == 2:
		vu += 1

	if vp > vn and vp > vu:
		return "1	0	0"
	elif vn > vp and vn > vu:
		return "0	1	0"
	elif vu > vp and vu > vn:
		return "0	0	1"
	else:
		return "0	0	0"


def getPosMax(c):
	if float(c.split('\t')[0]) > float(c.split('\t')[1]) and float(c.split('\t')[0]) > float(c.split('\t')[2]):
		return 0
	elif float(c.split('\t')[1]) > float(c.split('\t')[0]) and float(c.split('\t')[1]) > float(c.split('\t')[2]):
		return 1
	elif float(c.split('\t')[2]) > float(c.split('\t')[0]) and float(c.split('\t')[2]) > float(c.split('\t')[1]):
		return 2
	else:
		return -1


#def auxSum(a, b, c):
#	return a + b + c

def auxSum(a, b, c=0, d=0):
	return a + b + c + d

#pset = gp.PrimitiveSet("MAIN", 6) # one for each classify
#pset = gp.PrimitiveSet("MAIN", 5) # one for each classify

pset = gp.PrimitiveSetTyped("MAIN", [str, str, str, str, str], str)

pset.addPrimitive(getMajority3, [str, str, str], str)
pset.addPrimitive(getMajority,  [str, str], str)
pset.addPrimitive(getMax,  [str, str], str)
pset.addPrimitive(getMax3, [str, str, str], str)
pset.addPrimitive(getMin,  [str, str], str)
pset.addPrimitive(getMin3, [str, str, str], str)
pset.addPrimitive(getSum,  [str, str], str)
pset.addPrimitive(getSum3, [str, str, str], str)
#pset.addPrimitive(operator.add, 2)
#pset.addPrimitive(operator.sub, 2)
#pset.addPrimitive(operator.mul, 2)
#pset.addPrimitive(safeDiv, 2)
#pset.addPrimitive(operator.neg, 1)
#pset.addPrimitive(math.cos, 1)
#pset.addPrimitive(math.sin, 1)
#pset.addPrimitive(math.log10, 1)
#pset.addPrimitive(math.log2, 1)
#pset.addEphemeralConstant("randInt", lambda: random.randint(-2,2))
#pset.addEphemeralConstant("randFloat", lambda: random.uniform(-3,3))


pset.renameArguments(ARG0='RFOR')
pset.renameArguments(ARG1='SVM')
pset.renameArguments(ARG2='NB')
pset.renameArguments(ARG3='RL')
pset.renameArguments(ARG4='SGD')
#pset.renameArguments(ARG5='GP')
#pset.renameArguments(ARG0='SVM1')
#pset.renameArguments(ARG1='SVM2')
#pset.renameArguments(ARG2='SVM3')
#pset.renameArguments(ARG3='NAIVE1')
#pset.renameArguments(ARG4='NAIVE2')
#pset.renameArguments(ARG5='NAIVE3')
#pset.renameArguments(ARG6='RFOR1')
#pset.renameArguments(ARG7='RFOR2')
#pset.renameArguments(ARG8='RFOR3')

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def evalEnsemble_folds(individual):
	start = time.time()
	global test_values
	global svm_probs
	global naive_probs
	global rforest_probs
	global population
	global iterate_count
	global generation_count
	global best_f1_pos_neg
	global best_model

	indexes = list(range(len(test_values)))
	folds   = createRandomIndexChunks(indexes, 10)

	fitness_list = []
	#fold_index   = 0
	
	if iterate_count <= population:
		print("[individual " + str(iterate_count) + " of the generation " + str(generation_count) + "]\n")
		iterate_count += 1
	else:
		generation_count += 1
		iterate_count = 1
		print("\n[new generation][start generation " + str(generation_count) + "]\n")

	for fold in folds:
		correct  = 0
		accuracy = 0
		is_positive, is_negative, is_neutral = 0, 0, 0
		true_positive, true_negative, true_neutral, false_positive, false_negative, false_neutral  = 0, 0, 0, 0, 0, 0
		precision_positive, precision_negative, precision_neutral, precision_avg                   = 0, 0, 0, 0
		recall_positive, recall_negative, recall_neutral, recall_avg                               = 0, 0, 0, 0
		f1_positive, f1_negative, f1_neutral, f1_avg, f1_pos_neg_avg                               = 0, 0, 0, 0, 0
		first = True

		for i in fold:
			try:
				func        = toolbox.compile(expr=individual)
				#func_result = float(func(svm_probs[0][i], svm_probs[1][i], svm_probs[2][i], naive_probs[0][i], naive_probs[1][i], naive_probs[2][i], rforest_probs[0][i], rforest_probs[1][i], rforest_probs[2][i]))
				func_result = float(func(rforest_probs[i], svm_probs[i], naive_probs[i], lreg_probs[i], sgd_probs[i])) #, gp_probs[i]))
				if func_result >= variables.superior_range_gp_ensemble:
					if test_values[i]   == 1:
						is_positive     += 1
						correct         += 1
						true_positive   += 1
					elif test_values[i] == -1:
						is_negative     += 1
						false_negative  += 1
					elif test_values[i] == 0:
						is_neutral      += 1
						false_neutral   += 1

				elif func_result <= variables.inferior_range_gp_ensemble:
					if test_values[i]   == -1:
						is_negative     += 1
						correct         += 1
						true_negative   += 1
					elif test_values[i] == 1:
						is_positive     += 1
						false_positive  += 1
					elif test_values[i] == 0:
						is_neutral      += 1
						false_neutral   += 0 

				elif (func_result > variables.inferior_range_gp_ensemble and func_result < variables.superior_range_gp_ensemble):
					if test_values[i]   == 0:
						is_neutral      += 1
						true_neutral    += 1
						correct         += 1
					elif test_values[i] == 1:
						is_positive     += 1
						false_positive  += 1
					elif test_values[i] == -1:
						is_negative     += 1
						false_negative  += 1


			except Exception as e:
				print("Exception evalEnsemble_folds")
				print(e)
				continue
		
		accuracy = float(correct / len(fold))

		if true_positive + false_positive > 0:
			precision_positive = true_positive / (true_positive + false_positive)
		if true_negative + false_negative > 0:
			precision_negative = true_negative / (true_negative + false_negative)
		if true_neutral + false_neutral > 0:
			precision_neutral  = true_neutral  / (true_neutral + false_neutral)

		if is_positive > 0:
			recall_positive = true_positive   / is_positive
		if is_negative > 0:
			recall_negative = true_negative   / is_negative
		if is_neutral > 0:
			recall_neutral  = true_neutral    / is_neutral

		if precision_positive + recall_positive > 0:
			f1_positive = 2 * (precision_positive * recall_positive) / (precision_positive + recall_positive)
		if precision_negative + recall_negative > 0:
			f1_negative = 2 * (precision_negative * recall_negative) / (precision_negative + recall_negative)        
		if precision_neutral + recall_neutral > 0:
			f1_neutral  = 2 * (precision_neutral * recall_neutral)   / (precision_neutral + recall_neutral)        

		precision_avg  = (precision_positive + precision_negative + precision_neutral) / 3
		recall_avg     = (recall_positive + recall_negative + recall_neutral) / 3
		f1_avg         = (f1_positive + f1_negative + f1_neutral) / 3
		f1_pos_neg_avg = (f1_positive + f1_negative) / 2
		
		fitness_list.append(f1_pos_neg_avg)
		#fold_index += 1

	if f1_pos_neg_avg > best_f1_pos_neg and f1_pos_neg_avg != 0:
		best_f1_pos_neg = f1_pos_neg_avg
		best_model      = str(individual)
		with open('sandbox/partial_results/' + variables.BEST_INDIVIDUAL_GP_ENSEMBLE, 'w') as f:
			f.write(str(individual))
			f.write("\n\n# Generation -> " + str(generation_count))
			f.write("\n# f1_pos_neg -> " + str(best_f1_pos_neg))
			f.write("\n# Folds -> True")

	print("[fitness list] " + str(fitness_list))
	print("[fitness sum]  " + str(sum(fitness_list)))
	print("[# of folds]   " + str(len(folds)))
	print("[avg fitness]  " + str(sum(fitness_list)/len(folds)) + "\n")

	#print("[correct    ]: " + str(correct) + " of " + str(len(test_values)) + " messages [" + str(is_positive) + " pos, " + str(is_negative) + " neg, " + str(is_neutral) + " neu]")
	print("[f1_pos_neg ]: " + str(f1_pos_neg_avg) + " ***")
	if best_f1_pos_neg != 0:
		print("[best f1_P_N]: " + str(best_f1_pos_neg))
	print("[individual]:  " + str(individual))
	print("\n[evaluated in " + str(format(time.time() - start, '.3g')) + " seconds]")
	print("-------------------\n")

	return sum(fitness_list)/len(folds),


def evalEnsemble(individual):
	start = time.time()
	func = toolbox.compile(expr=individual)
	
	#global train_values
	global test_values
	global svm_probs
	global naive_probs
	global rforest_probs
	global population
	global iterate_count
	global generation_count
	global best_acc
	global best_f1_pos_neg
	global best_model

	is_positive, is_negative, is_neutral = 0, 0, 0
	true_positive, true_negative, true_neutral, false_positive, false_negative, false_neutral  = 0, 0, 0, 0, 0, 0
	precision_positive, precision_negative, precision_neutral, precision_avg                   = 0, 0, 0, 0
	recall_positive, recall_negative, recall_neutral, recall_avg                               = 0, 0, 0, 0
	f1_positive, f1_negative, f1_neutral, f1_avg, f1_pos_neg_avg                               = 0, 0, 0, 0, 0
	correct = 0
	
	print(str(individual) + "\n")

	# Log the number of each individual
	if iterate_count <= population:
		print("[individual " + str(iterate_count) + " of the generation " + str(generation_count) + "]\n")
		iterate_count += 1
	else:
		generation_count += 1
		iterate_count = 1
		print("\n[new generation][start generation " + str(generation_count) + "]\n")
		new_generation = True

	print(str(individual))
	for i in range(len(test_values)):
		try:
			func        = toolbox.compile(expr=individual)
			#func_result = float(func(svm_probs[0][i], svm_probs[1][i], svm_probs[2][i], naive_probs[0][i], naive_probs[1][i], naive_probs[2][i], rforest_probs[0][i], rforest_probs[1][i], rforest_probs[2][i]))
			#func_result = float(func(rforest_probs[i], svm_probs[i], naive_probs[i], lreg_probs[i], sgd_probs[i], gp_probs[i]))
			func_result = convertResult(func(str(rforest_probs[i]), str(svm_probs[i]), str(naive_probs[i]), str(lreg_probs[i]), str(sgd_probs[i]))) #, gp_probs[i]))
			#print(func_result)
			#print("AAAA\n\n\n\n\n")


			# see how normalize this to consider a range of pos, neg and neu values
			if func_result >= variables.superior_range_gp_ensemble:
				if test_values[i]   == 1:
					is_positive     += 1
					correct         += 1
					true_positive   += 1
				elif test_values[i] == -1:
					is_negative     += 1
					false_negative  += 1
				elif test_values[i] == 0:
					is_neutral      += 1
					false_neutral   += 1

			elif func_result <= variables.inferior_range_gp_ensemble:
				if test_values[i]   == -1:
					is_negative     += 1
					correct         += 1
					true_negative   += 1
				elif test_values[i] == 1:
					is_positive     += 1
					false_positive  += 1
				elif test_values[i] == 0:
					is_neutral      += 1
					false_neutral   += 0 

			elif (func_result > variables.inferior_range_gp_ensemble and func_result < variables.superior_range_gp_ensemble):
				if test_values[i]   == 0:
					is_neutral      += 1
					true_neutral    += 1
					correct         += 1
				elif test_values[i] == 1:
					is_positive     += 1
					false_positive  += 1
				elif test_values[i] == -1:
					is_negative     += 1
					false_negative  += 1
			#print("func_result[" + str(i) + "] " + str(func_result))

		except Exception as e: 
			print("Exception evalEnsemble")
			print(e)
			continue

	acc = float(correct / len(test_values))
	if acc > best_acc:
		best_acc = acc		

	if true_positive + false_positive > 0:
		precision_positive = true_positive / (true_positive + false_positive)
	if true_negative + false_negative > 0:
		precision_negative = true_negative / (true_negative + false_negative)
	if true_neutral + false_neutral > 0:
		precision_neutral  = true_neutral  / (true_neutral + false_neutral)

	if is_positive > 0:
		recall_positive = true_positive   / is_positive
	if is_negative > 0:
		recall_negative = true_negative   / is_negative
	if is_neutral > 0:
		recall_neutral  = true_neutral    / is_neutral

	if precision_positive + recall_positive > 0:
		f1_positive = 2 * (precision_positive * recall_positive) / (precision_positive + recall_positive)
	if precision_negative + recall_negative > 0:
		f1_negative = 2 * (precision_negative * recall_negative) / (precision_negative + recall_negative)        
	if precision_neutral + recall_neutral > 0:
		f1_neutral  = 2 * (precision_neutral * recall_neutral)   / (precision_neutral + recall_neutral)        

	precision_avg  = (precision_positive + precision_negative + precision_neutral) / 3
	recall_avg     = (recall_positive + recall_negative + recall_neutral) / 3
	f1_avg         = (f1_positive + f1_negative + f1_neutral) / 3
	f1_pos_neg_avg = (f1_positive + f1_negative) / 2

	if f1_pos_neg_avg > best_f1_pos_neg and f1_pos_neg_avg != 0:
		best_f1_pos_neg = f1_pos_neg_avg
		best_model      = str(individual)
		with open('sandbox/partial_results/' + variables.BEST_INDIVIDUAL_GP_ENSEMBLE, 'w') as f:
			f.write(str(individual))
			f.write("\n\n# Generation -> " + str(generation_count))
			f.write("\n# f1_pos_neg -> " + str(best_f1_pos_neg))
			f.write("\n# Folds -> False")

	print("[correct    ]: " + str(correct) + " of " + str(len(test_values)) + " messages [" + str(is_positive) + " pos, " + str(is_negative) + " neg, " + str(is_neutral) + " neu]")
	#print("[accuracy]: " + str(acc) + " ***")
	print("[f1_pos_neg ]: " + str(f1_pos_neg_avg) + " ***")
	if best_f1_pos_neg != 0:
		print("[best f1_P_N]: " + str(best_f1_pos_neg))
	print("\n[evaluated in " + str(format(time.time() - start, '.3g')) + " seconds]")
	print("-------------------\n")

	return f1_pos_neg_avg,


if variables.train_using_folds_gp_ensemble:
	toolbox.register("evaluate", evalEnsemble_folds)
else:
	toolbox.register("evaluate", evalEnsemble)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=6))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=6))


def loadClassifierPredictions(file_path, classifier=""):
	start = time.time()
	print("[loading classifier predictions][" + classifier + "]")

	predictions = []
	offset = 0
	if classifier == "naive-test": # change the naive file to remove this
		offset += 1
	
	with open(file_path, 'r') as f:
		for line in f:
			if not line.startswith("#"):
				predictions.append(float(line.split('\t')[4 + offset].strip()))
	
	
	end = time.time()
	print("  [classifier predictions loaded][" + str(format(end - start, '.3g')) + " seconds]\n")
	
	return predictions


def loadClassifierProbStr(file_path, classifier):
	start = time.time()
	print("[loading classifier probs string][" + classifier + "]")

	result = []
	offset = 0
	if classifier == "naive-test": # change the naive file to remove this
		offset += 1
	
	with open(file_path, 'r') as f:
		for line in f:
			if not line.startswith("#"):
				result.append(line.split('\t')[0 + offset].strip() + "\t" + line.split('\t')[1 + offset].strip() + "\t" + line.split('\t')[2 + offset].strip())
		
	return result	
	end = time.time()
	print("  [classifier probs string loaded][" + str(format(end - start, '.3g')) + " seconds]\n")


def loadClassifierProbs(file_path, classifier=""):
	start = time.time()
	print("[loading classifier probs][" + classifier + "]")

	v1, v2, v3 = [], [], []
	result = []
	offset = 0
	if classifier == "naive-test": # change the naive file to remove this
		offset += 1
	
	with open(file_path, 'r') as f:
		for line in f:
			if not line.startswith("#"):
				v1.append(float(line.split('\t')[0 + offset].strip()))
				v2.append(float(line.split('\t')[1 + offset].strip()))
				v3.append(float(line.split('\t')[2 + offset].strip()))
	
	result.append(v1)
	result.append(v2)
	result.append(v3)
	
	end = time.time()
	print("  [classifier probs loaded][" + str(format(end - start, '.3g')) + " seconds]\n")
	
	return result


def normalizePolarity(polarity_string):
	if polarity_string.lower() == "positive" or polarity_string.lower() == "pos":
		return 1 # ??
	elif polarity_string.lower() == "negative" or polarity_string.lower() == "neg":
		return -1 # ??
	elif polarity_string.lower() == "neutral" or polarity_string.lower() == "neu":
		return 0 #???


def loadBaseIndexes(file_path, base):
	indexes = []
	i = 0
	with open(file_path, 'r') as f:
		for line in f:
			if str(line.split('\t')[1]).strip().lower() == base.strip().lower():
				indexes.append(i)
			i += 1

	return indexes

def loadValues(file_path, module, base="none"):
	if module == "train":
		index = 2
	elif module == "test":
		index = 0
	
	values = []
	with open(file_path, 'r') as f:
		for line in f:
			values.append(normalizePolarity(line.split('\t')[index]))

	return values


def testModel_ensembleGP(model, base="default"):
	global test_values
	global svm_probs_test
	global naive_probs_test
	global rforest_probs_test

	is_positive, is_negative, is_neutral = 0, 0, 0
	true_positive, true_negative, true_neutral, false_positive, false_negative, false_neutral  = 0, 0, 0, 0, 0, 0
	precision_positive, precision_negative, precision_neutral, precision_avg                   = 0, 0, 0, 0
	recall_positive, recall_negative, recall_neutral, recall_avg                               = 0, 0, 0, 0
	f1_positive, f1_negative, f1_neutral, f1_avg, f1_pos_neg_avg                               = 0, 0, 0, 0, 0
	goldP_predN, goldP_predU, goldN_predP, goldN_predU, goldU_predP,goldU_predN                = 0, 0, 0, 0, 0, 0
	correct = 0

	original_model = model

	if base == "default":
		indexes = range(len(test_values))
	else:
		indexes = loadBaseIndexes('datasets/test/SemEval2014_SVM_Naive_MS_Lreg_S140.txt', base)

	for i in indexes:
		model = original_model
		model = model.replace("(RFOR", "('"  + str(rforest_probs[i]) + "'").replace("RFOR)", "'" + str(rforest_probs[i]) + "')").replace("RFOR,", "'" + str(rforest_probs[i]) + "',")
		model = model.replace("(SVM",  "('"  + str(svm_probs[i])     + "'").replace("SVM)",  "'" + str(svm_probs[i])     + "')").replace("SVM,",  "'" + str(svm_probs[i])     + "',")	
		model = model.replace("(NB",   "('"  + str(naive_probs[i])   + "'").replace("NB)",   "'" + str(naive_probs[i])   + "')").replace("NB,",   "'" + str(naive_probs[i])   + "',")
		model = model.replace("(RL",   "('"  + str(lreg_probs[i])    + "'").replace("RL)",   "'" + str(lreg_probs[i])    + "')").replace("RL,", "'" + str(lreg_probs[i])    + "',")
		model = model.replace("(SGD",  "('"  + str(sgd_probs[i])     + "'").replace("SGD)",  "'" + str(sgd_probs[i])     + "')").replace("SGD,",  "'" + str(sgd_probs[i])     + "',")
		#model = model.replace("(GP", "("   + str(gp_probs[i])).replace("GP)",        str(gp_probs[i]) + ")")	

		#model = model.replace("(SVM1", "(" + str(svm_probs_test[0][i])).replace("SVM1)", str(svm_probs_test[0][i]) + ")")
		#model = model.replace("(SVM2", "(" + str(svm_probs_test[1][i])).replace("SVM2)", str(svm_probs_test[1][i]) + ")")
		#model = model.replace("(SVM3", "(" + str(svm_probs_test[2][i])).replace("SVM3)", str(svm_probs_test[2][i]) + ")")
		#model = model.replace("(NAIVE1", "(" + str(naive_probs_test[0][i])).replace("NAIVE1)", str(naive_probs_test[0][i]) + ")")
		#model = model.replace("(NAIVE2", "(" + str(naive_probs_test[1][i])).replace("NAIVE2)", str(naive_probs_test[1][i]) + ")")
		#model = model.replace("(NAIVE3", "(" + str(naive_probs_test[2][i])).replace("NAIVE3)", str(naive_probs_test[2][i]) + ")")
		#model = model.replace("(RFOR1", "(" + str(rforest_probs_test[0][i])).replace("RFOR1)", str(rforest_probs_test[0][i]) + ")")
		#model = model.replace("(RFOR2", "(" + str(rforest_probs_test[1][i])).replace("RFOR2)", str(rforest_probs_test[1][i]) + ")")
		#model = model.replace("(RFOR3", "(" + str(rforest_probs_test[2][i])).replace("RFOR3)", str(rforest_probs_test[2][i]) + ")")

		result = float(convertResult(eval(model)))
		print(model + "\n")
		print(str(result))
		print(str(test_values[i]) + "\n")

		if result >= variables.superior_range_gp_ensemble:
			if test_values[i]   == 1:
				is_positive     += 1
				correct         += 1
				true_positive   += 1
			elif test_values[i] == -1:
				is_negative     += 1
				false_negative  += 1
				goldN_predP     += 1
			elif test_values[i] == 0:
				is_neutral      += 1
				false_neutral   += 1
				goldU_predP     += 1

		elif result <= variables.inferior_range_gp_ensemble:
			if test_values[i]   == -1:
				is_negative     += 1
				correct         += 1
				true_negative   += 1
			elif test_values[i] == 1:
				is_positive     += 1
				false_positive  += 1
				goldP_predN     += 1
			elif test_values[i] == 0:
				is_neutral      += 1
				false_neutral   += 0
				goldU_predN     += 1

		elif (result > variables.inferior_range_gp_ensemble and result < variables.superior_range_gp_ensemble):
			if test_values[i]   == 0:
				is_neutral      += 1
				true_neutral    += 1
				correct         += 1
			elif test_values[i] == 1:
				is_positive     += 1
				false_positive  += 1
				goldP_predU     += 1
			elif test_values[i] == -1:
				is_negative     += 1
				false_negative  += 1
				goldN_predU     += 1

	acc = float(correct / len(indexes))

	if true_positive + false_positive > 0:
		precision_positive = true_positive / (true_positive + false_positive)
	if true_negative + false_negative > 0:
		precision_negative = true_negative / (true_negative + false_negative)
	if true_neutral + false_neutral > 0:
		precision_neutral  = true_neutral  / (true_neutral + false_neutral)

	if is_positive > 0:
		recall_positive = true_positive   / is_positive
	if is_negative > 0:
		recall_negative = true_negative   / is_negative
	if is_neutral > 0:
		recall_neutral  = true_neutral    / is_neutral

	if precision_positive + recall_positive > 0:
		f1_positive = 2 * (precision_positive * recall_positive) / (precision_positive + recall_positive)
	if precision_negative + recall_negative > 0:
		f1_negative = 2 * (precision_negative * recall_negative) / (precision_negative + recall_negative)        
	if precision_neutral + recall_neutral > 0:
		f1_neutral  = 2 * (precision_neutral * recall_neutral)   / (precision_neutral + recall_neutral)        

	precision_avg  = (precision_positive + precision_negative + precision_neutral) / 3
	recall_avg     = (recall_positive + recall_negative + recall_neutral) / 3
	f1_avg         = (f1_positive + f1_negative + f1_neutral) / 3
	f1_pos_neg_avg = (f1_positive + f1_negative) / 2

	if base == "default":
		print("all messages")
	else:
		print(str(base) + " messages\n")
	print("correct evals: " + str(correct) + " messages of " + str(len(indexes)) + " (" + str(acc) + " accuracy)")
	print("true positive: " + str(true_positive) + " (of " + str(is_positive) + " positive messages)")
	print("true negative: " + str(true_negative) + " (of " + str(is_negative) + " negative messages)")
	print("true neutral:  " + str(true_neutral)  + " (of " + str(is_neutral) + " neutral messages)")
	print("f1 3 classes:  " + str(f1_avg) + " (f1 pos " + str(f1_positive) + ", f1 neg " + str(f1_negative) + ", f1 neu " + str(f1_neutral) + ")")
	print("f1 2 classes:  " + str(f1_pos_neg_avg))

	print("\n")
	print(str(base) + " confusion matrix\n")
	print("          |  Gold_Pos  |  Gold_Neg  |  Gold_Neu  |")
	print("--------------------------------------------------")
	print("Pred_Pos  |  " + '{message: <{width}}'.format(message=str(true_positive), width=8) + "  |  " + '{message: <{width}}'.format(message=str(goldN_predP), width=8) + "  |  " + '{message: <{width}}'.format(message=str(goldU_predP), width=8) + "  |")
	print("Pred_Neg  |  " + '{message: <{width}}'.format(message=str(goldP_predN), width=8) + "  |  " + '{message: <{width}}'.format(message=str(true_negative), width=8) + "  |  " + '{message: <{width}}'.format(message=str(goldU_predN), width=8) + "  |")
	print("Pred_Neu  |  " + '{message: <{width}}'.format(message=str(goldP_predU), width=8) + "  |  " + '{message: <{width}}'.format(message=str(goldN_predU), width=8) + "  |  " + '{message: <{width}}'.format(message=str(true_neutral), width=8)  + "  |")

	print("\n")

	return f1_pos_neg_avg

# Global vars
train_values  = loadValues('datasets/train/twitter-train-cleansed-B.txt', 'train')
test_values   = loadValues('datasets/test/SemEval2014_SVM_Naive_MS_Lreg_S140.txt', 'test')
test_values   = loadValues('datasets/train/twitter-train-cleansed-B.txt', 'train')

#######################################################
# TO-DO: get the train values instead the test values #
#######################################################

#svm_probs_test     = loadClassifierProbs('datasets/test/SVM_test_results.txt', 'svm')
#naive_probs_test   = loadClassifierProbs('datasets/test/Naive_test_results.txt', 'naive-test')
#rforest_probs_test = loadClassifierProbs('datasets/test/RandomForest_test_results.txt', 'rforest')

#svm_probs     = loadClassifierProbs('datasets/train/svm_train_results.txt', 'svm')
#naive_probs   = loadClassifierProbs('datasets/train/naive_train_results.txt', 'naive')
#rforest_probs = loadClassifierProbs('datasets/train/randomForest_train_results.txt', 'rforest')
#lreg_probs    = loadClassifierProbs('datasets/train/lreg_train_results.txt', 'lreg')

svm_probs     = loadClassifierProbStr('datasets/train/svm_train_results.txt', 'svm')
naive_probs   = loadClassifierProbStr('datasets/train/naive_train_results.txt', 'naive')
rforest_probs = loadClassifierProbStr('datasets/train/randomForest_train_results.txt', 'rforest')
lreg_probs    = loadClassifierProbStr('datasets/train/lreg_train_results.txt', 'lreg')
sgd_probs     = loadClassifierProbStr('datasets/train/sgd_train_results.txt', 'sgd')
#gp_probs      = loadClassifierProbStr('datasets/train/gp_train_results.txt', 'gp')

#svm_predictions     = loadClassifiersPredictions('datasets/train/svm_train_results.txt', 'svm')
#naive_predictions   = loadClassifiersPredictions('datasets/train/naive_train_results.txt', 'naive')
#rforest_predictions = loadClassifiersPredictions('datasets/train/randomForest_train_results.txt', 'rforest')
#lreg_predictions    = loadClassifiersPredictions('datasets/train/lreg_train_results.txt', 'lreg')

population = 200

iterate_count    = 1
generation_count = 1

best_acc        = 0
best_f1_pos_neg = 0
best_model      = ""


def myVarAnd(population, toolbox, cxpb, mutpb):
    #print("YAY, I'm in myVarAnd")

    offspring = [toolbox.clone(ind) for ind in population]

    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                          offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    for i in range(len(offspring)):
        if random.random() < mutpb:
            #print("MUTATE NORMAL")
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values
    
    for i in range(len(offspring)):
        if random.random() < variables.MUTATE_EPHEMERAL:
            #print("MUTATE EPHEMERAL")
            offspring[i], = toolbox.mutateEphemeral(offspring[i], "all")
            del offspring[i].fitness.values
    
    # my mutation (only for w or other real values)
    #for i in range(len(offspring)):
    #    if random.random() < variables.MUTATION_W:
    #        offspring[i], = mutateW(offspring[i])
    #        del offspring[i].fitness.values

    return offspring


def myEaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        offspringOriginal = offspring

        # Vary the pool of individuals
        offspring = myVarAnd(offspring, toolbox, cxpb, mutpb)

        #if random.random() < variables.MUTATION_W:
        #    print("####################\n")
        #    print("My W's Mutation HERE")
        #    print("####################\n")


        #for index, item in enumerate(offspring): 
        #    if (offspringOriginal[index] != offspring[index]):
        #        print("DEFAULT MUTATION!!!!!")
        #        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
        #        print("offspring original   -> " + str(offspringOriginal[index]) + "\n")
        #        print("offspring modificado -> " + str(offspring[index]) + "\n")
        #        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")


        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


def main():
	random.seed()
	global population
	global best_acc
	global best_f1_pos_neg
	
	pop = toolbox.population(n=population)
	hof = tools.HallOfFame(2)

	pop, log = myEaSimple(pop, toolbox, 0.5, 0.1, 20, stats=False,
								   halloffame=hof, verbose=False)
	
	print("\n\n[best model ]: " + str(hof[0]))
	#print("[best acc]:   " + str(best_acc) + "\n")
	print("[best f1_P_N]:   " + str(best_f1_pos_neg) + "\n")
	# print log

	with open(variables.TRAIN_RESULTS_GP_ENS, 'a') as f:
		f.write(str(hof[0]) + "\n")

	return pop, log, hof

if __name__ == "__main__":
	main()

	#print("0.8	0.1	0.1", "0.9	0.1	0.0", "0.2	0.4	0.4")
	#print(getMax3("0.8	0.1	0.1", "0.9	0.1	0.0", "0.2	0.4	0.4"))
	#print(getMin3("0.8	0.1	0.1", "0.9	0.1	0.0", "0.2	0.4	0.4"))
	#print(getSum3("0.8	0.1	0.1", "0.9	0.1	0.0", "0.2	0.4	0.4"))
	#print(getMajority3("0.8	0.1	0.1", "0.1	0.8	0.1", "0.2	0.3	0.5"))
	#print(getMajority("0.8	0.1	0.1", "0.1	0.8	0.1"))

	#print(str(loadBaseIndexes('datasets/test/SemEval2014_SVM_Naive_MS_Lreg_S140.txt', 'Twitter2014')))

	#m = "getMajority3(getMajority3(SVM, SGD, RL), NB, RFOR)"
	#testModel_ensembleGP(m, "Twitter2013")
	
	#testModel_ensembleGP(m, "Twitter2013")
	#testModel_ensembleGP(m, "Twitter2014")
	#testModel_ensembleGP(m, "Twitter2014Sarcasm")
	#testModel_ensembleGP(m, "SMS2013")
	#testModel_ensembleGP(m, "LiveJournal2014")
	#testModel_ensembleGP(m)    
	#print(x)