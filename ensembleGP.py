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

def safeDiv(left, right):
	try:
		return left / right
	except ZeroDivisionError:
		return 0

def neg(value):
	return -value

def add3(v1, v2, v3):
	return v1 + v2 + v3

def add4(v1, v2, v3, v4):
	return v1 + v2 + v3 + v4

def add5(v1, v2, v3, v4, v5):
	return v1 + v2 + v3 + v4 + v5

def add6(v1, v2, v3, v4, v5, v6):
	return v1 + v2 + v3 + v4 + v5 + v6	

def mul3(v1, v2, v3):
	return v1 * v2 * v3 

def mul4(v1, v2, v3, v4):
	return v1 * v2 * v3 * v4

def mul5(v1, v2, v3, v4, v5):
	return v1 * v2 * v3	* v4* v5

def mul6(v1, v2, v3, v4, v5, v6):
	return v1 * v2 * v3	* v4* v5 * v6

pset = gp.PrimitiveSet("MAIN", 9)
pset.addPrimitive(operator.add, 2)
#pset.addPrimitive(add3, 3)
#pset.addPrimitive(add4, 4)
#pset.addPrimitive(add5, 5)
#pset.addPrimitive(add6, 6)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
#pset.addPrimitive(mul3, 3)
#pset.addPrimitive(mul4, 4)
#pset.addPrimitive(mul5, 5)
#pset.addPrimitive(mul6, 6)
pset.addPrimitive(safeDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
#pset.addPrimitive(math.log10, 1)
#pset.addPrimitive(math.log2, 1)
pset.addEphemeralConstant("randInt", lambda: random.randint(-2,2))
pset.addEphemeralConstant("randFloat", lambda: random.uniform(-2,2))

pset.renameArguments(ARG0='SVM1')
pset.renameArguments(ARG1='SVM2')
pset.renameArguments(ARG2='SVM3')
pset.renameArguments(ARG3='NAIVE1')
pset.renameArguments(ARG4='NAIVE2')
pset.renameArguments(ARG5='NAIVE3')
pset.renameArguments(ARG6='RFOR1')
pset.renameArguments(ARG7='RFOR2')
pset.renameArguments(ARG8='RFOR3')

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=4, max_=6)
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
				func_result = float(func(svm_probs[0][i], svm_probs[1][i], svm_probs[2][i], naive_probs[0][i], naive_probs[1][i], naive_probs[2][i], rforest_probs[0][i], rforest_probs[1][i], rforest_probs[2][i]))

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

	for i in range(len(test_values)):
		try:
			func        = toolbox.compile(expr=individual)
			func_result = float(func(svm_probs[0][i], svm_probs[1][i], svm_probs[2][i], naive_probs[0][i], naive_probs[1][i], naive_probs[2][i], rforest_probs[0][i], rforest_probs[1][i], rforest_probs[2][i]))

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
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


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
		model = model.replace("(SVM1", "(" + str(svm_probs_test[0][i])).replace("SVM1)", str(svm_probs_test[0][i]) + ")")
		model = model.replace("(SVM2", "(" + str(svm_probs_test[1][i])).replace("SVM2)", str(svm_probs_test[1][i]) + ")")
		model = model.replace("(SVM3", "(" + str(svm_probs_test[2][i])).replace("SVM3)", str(svm_probs_test[2][i]) + ")")
		model = model.replace("(NAIVE1", "(" + str(naive_probs_test[0][i])).replace("NAIVE1)", str(naive_probs_test[0][i]) + ")")
		model = model.replace("(NAIVE2", "(" + str(naive_probs_test[1][i])).replace("NAIVE2)", str(naive_probs_test[1][i]) + ")")
		model = model.replace("(NAIVE3", "(" + str(naive_probs_test[2][i])).replace("NAIVE3)", str(naive_probs_test[2][i]) + ")")
		model = model.replace("(RFOR1", "(" + str(rforest_probs_test[0][i])).replace("RFOR1)", str(rforest_probs_test[0][i]) + ")")
		model = model.replace("(RFOR2", "(" + str(rforest_probs_test[1][i])).replace("RFOR2)", str(rforest_probs_test[1][i]) + ")")
		model = model.replace("(RFOR3", "(" + str(rforest_probs_test[2][i])).replace("RFOR3)", str(rforest_probs_test[2][i]) + ")")

		result = float(eval(model))

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

#######################################################
# TO-DO: get the train values instead the test values #
#######################################################

svm_probs_test     = loadClassifierProbs('datasets/test/SVM_test_results.txt', 'svm')
naive_probs_test   = loadClassifierProbs('datasets/test/Naive_test_results.txt', 'naive-test')
rforest_probs_test = loadClassifierProbs('datasets/test/RandomForest_test_results.txt', 'rforest')

svm_probs     = loadClassifierProbs('datasets/train/svm_train_results.txt', 'svm')
naive_probs   = loadClassifierProbs('datasets/train/naive_train_results.txt', 'naive')
rforest_probs = loadClassifierProbs('datasets/train/randomForest_train_results.txt', 'rforest')

population = 200

iterate_count    = 1
generation_count = 1

best_acc        = 0
best_f1_pos_neg = 0
best_model      = ""

def main():
	random.seed()
	global population
	global best_acc
	global best_f1_pos_neg
	
	pop = toolbox.population(n=population)
	hof = tools.HallOfFame(2)

	pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 20, stats=False,
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
	#print(str(loadBaseIndexes('datasets/test/SemEval2014_SVM_Naive_MS_Lreg_S140.txt', 'Twitter2014')))

	#m = "sub(safeDiv(sub(sub(SVM3, 0), mul(sub(sin(mul(add(SVM1, 1.9786153720017947), neg(-2))), add(neg(mul(NAIVE3, NAIVE3)), safeDiv(safeDiv(SVM2, sub(sin(mul(neg(mul(NAIVE3, NAIVE3)), neg(-2))), add(neg(mul(NAIVE3, NAIVE3)), safeDiv(safeDiv(SVM2, NAIVE1), safeDiv(NAIVE1, NAIVE1))))), safeDiv(NAIVE1, SVM3)))), -2)), safeDiv(mul(NAIVE3, NAIVE3), neg(-2))), cos(sub(mul(mul(NAIVE1, mul(add(-0.7083473166495553, -1.4732497725511893), safeDiv(SVM2, -2))), sin(NAIVE2)), sub(cos(NAIVE3), safeDiv(NAIVE1, NAIVE3)))))"
	#m = "sub(safeDiv(sub(neg(cos(mul(sub(sin(mul(add(SVM1, 1.9786153720017947), neg(-2))), add(neg(mul(NAIVE3, NAIVE3)), neg(SVM1))), -2))), mul(sub(sin(mul(add(SVM1, 1.9786153720017947), neg(-2))), add(SVM1, safeDiv(safeDiv(SVM2, sub(sin(mul(neg(mul(NAIVE3, NAIVE3)), neg(-2))), add(neg(mul(NAIVE3, NAIVE3)), safeDiv(safeDiv(SVM2, NAIVE1), safeDiv(NAIVE1, NAIVE1))))), safeDiv(NAIVE1, SVM3)))), -2)), safeDiv(mul(NAIVE3, NAIVE3), neg(-2))), cos(sub(mul(cos(1), sin(NAIVE2)), sub(cos(1), neg(-2)))))"
	#m = "sub(NAIVE3, safeDiv(cos(mul(SVM1, mul(SVM3, NAIVE2))), mul(NAIVE3, sub(SVM1, neg(cos(1))))))"
	
	#m = "safeDiv(add(safeDiv(add(0, safeDiv(NAIVE1, sub(neg(neg(-2)), neg(cos(sin(SVM2)))))), -0.6729775461871736), safeDiv(add(NAIVE2, add(1.6586285319358236, RFOR1)), neg(NAIVE3))), sin(mul(neg(-2), safeDiv(1.7862219926220981, NAIVE2))))"
	#m = "sub(NAIVE3, safeDiv(cos(mul(SVM1, mul(SVM3, NAIVE2))), mul(NAIVE3, sub(SVM1, neg(cos(1))))))"
	#testModel_ensembleGP(m, "Twitter2013")
	#testModel_ensembleGP(m, "Twitter2014")
	#testModel_ensembleGP(m, "Twitter2014Sarcasm")
	#testModel_ensembleGP(m, "SMS2013")
	#testModel_ensembleGP(m, "LiveJournal2014")
	#testModel_ensembleGP(m)    
	#print(x)