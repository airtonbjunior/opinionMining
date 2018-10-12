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


def safeDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 0

pset = gp.PrimitiveSet("MAIN", 6)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(safeDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("randInt", lambda: random.randint(-2,2))
pset.addEphemeralConstant("randFloat", lambda: random.uniform(-2,2))

pset.renameArguments(ARG0='a')
pset.renameArguments(ARG1='b')
pset.renameArguments(ARG2='c')
pset.renameArguments(ARG3='d')
pset.renameArguments(ARG4='e')
pset.renameArguments(ARG5='f')

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=4, max_=6)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evalEnsemble(individual):
	start = time.time()
	func = toolbox.compile(expr=individual)
	
	#global train_values
	global test_values
	global svm_probs
	global naive_probs
	global population
	global iterate_count
	global generation_count
	global best_acc
	global best_f1_pos_neg

	is_positive, is_negative, is_neutral = 0, 0, 0
	true_positive, true_negative, true_neutral, false_positive, false_negative, false_neutral  = 0, 0, 0, 0, 0, 0
	precision_positive, precision_negative, precision_neutral, precision_avg                   = 0, 0, 0, 0
	recall_positive, recall_negative, recall_neutral, recall_avg                               = 0, 0, 0, 0
	f1_positive, f1_negative, f1_neutral, f1_avg, f1_pos_neg_avg                               = 0, 0, 0, 0, 0

	print(str(individual) + "\n")

	correct = 0 # accuracy, for while

	# Log the number of each individual
	if iterate_count <= population:
	    print("[individual " + str(iterate_count) + " of the generation " + str(generation_count) + "]\n")
	    iterate_count += 1
	else:
	    generation_count += 1
	    iterate_count = 1
	    print("\n[new generation][start generation " + str(generation_count) + "]\n")
	    new_generation = True

	for i in range(len(test_values)):
		try:
			func        = toolbox.compile(expr=individual)
			func_result = float(func(svm_probs[0][i], svm_probs[1][i], svm_probs[2][i], naive_probs[0][i], naive_probs[1][i], naive_probs[2][i]))

			# see how normalize this to consider a range of pos, neg and neu values
			if func_result >= 1:
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

			elif func_result <= -1:
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

			elif (func_result > -1 and func_result < 1):
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
			print("Exception 1")
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
		precision_neutral = true_neutral / (true_neutral + false_neutral)

	if is_positive > 0:
		recall_positive = true_positive / is_positive
	if is_negative > 0:
		recall_negative = true_negative / is_negative
	if is_neutral > 0:
		recall_neutral = true_neutral / is_neutral

	if precision_positive + recall_positive > 0:
		f1_positive = 2 * (precision_positive * recall_positive) / (precision_positive + recall_positive)
	if precision_negative + recall_negative > 0:
		f1_negative = 2 * (precision_negative * recall_negative) / (precision_negative + recall_negative)        
	if precision_neutral + recall_neutral > 0:
		f1_neutral = 2 * (precision_neutral * recall_neutral) / (precision_neutral + recall_neutral)        

	precision_avg  = (precision_positive + precision_negative + precision_neutral) / 3
	recall_avg     = (recall_positive + recall_negative + recall_neutral) / 3
	f1_avg         = (f1_positive + f1_negative + f1_neutral) / 3
	f1_pos_neg_avg = (f1_positive + f1_negative) / 2

	if f1_pos_neg_avg > best_f1_pos_neg:
		best_f1_pos_neg = f1_pos_neg_avg

	print("[correct    ]: " + str(correct) + " of " + str(len(test_values)) + " messages [" + str(is_positive) + " pos, " + str(is_negative) + " neg, " + str(is_neutral) + " neu]")
	#print("[accuracy]: " + str(acc) + " ***")
	print("[f1_pos_neg ]: " + str(f1_pos_neg_avg) + " ***")
	if best_f1_pos_neg != 0:
		print("[best f1_P_N]: " + str(best_f1_pos_neg))
	print("\n[evaluated in " + str(format(time.time() - start, '.3g')) + " seconds]")
	print("-------------------\n")

	return f1_pos_neg_avg,


toolbox.register("evaluate", evalEnsemble)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


def loadClassifierProbs(file_path, classifier=""):
	start = time.time()
	print("[loading classifier probs][" + classifier + "]")

	v1, v2, v3 = [], [], []
	result = []
	offset = 0
	if classifier == "naive": # change the naive file to remove this
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


def loadValues(file_path, module):
	if module == "train":
		index = 2
	elif module == "test":
		index = 0
	
	values = []
	with open(file_path, 'r') as f:
		for line in f:
			values.append(normalizePolarity(line.split('\t')[index]))

	return values


# Global vars
train_values = loadValues('datasets/train/twitter-train-cleansed-B.txt', 'train')
test_values  = loadValues('datasets/test/SemEval2014_SVM_Naive_MS_Lreg_S140.txt', 'test')
svm_probs    = loadClassifierProbs('datasets/test/SVM_test_results.txt', 'svm')
naive_probs  = loadClassifierProbs('datasets/test/Naive_test_results.txt', 'naive')

population = 200

iterate_count    = 1
generation_count = 1

best_acc        = 0
best_f1_pos_neg = 0

def main():
	random.seed()
	global population
	global best_acc
	
	pop = toolbox.population(n=population)
	hof = tools.HallOfFame(2)

	pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 20, stats=False,
	                               halloffame=hof, verbose=False)
	
	print("\n\n[best model]: " + str(hof[0]))
	print("[best acc]:   " + str(best_acc) + "\n")
	# print log
	return pop, log, hof

if __name__ == "__main__":
    main()