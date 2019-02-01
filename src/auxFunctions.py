""" 
<Author> Airton Bordin Junior
<Email>  airtonbjunior@gmail.com

Federal University of Goias (UFG)
Computer Science Master's Degree


auxFunctions.py
	Aux Functions used by the Genetic Programming Algorithm

"""
import time
import variables as var
import matplotlib.pyplot as plt
from datetime import datetime


def saveBestFitness(file_path, individual, fitnessReturn, generation_count, is_positive, is_negative, is_neutral):
	""" Save the best fitness on a file
		
	"""	
	if var.BEST['fitness'] != 0:
		# save partial best individual (in case we need stop evolution)
		with open(file_path, 'w') as f:
			f.write(str(individual))
			f.write("\n\n# Generation -> " + str(generation_count))
			f.write("\n# Neutral Range -> [" + str(var.NEUTRAL['range']['inferior']) + ", " + str(var.NEUTRAL['range']['superior']) + "]")
		var.HISTORY['fitness']['best'].append(var.BEST['fitness'])
	var.BEST['fitness'] = fitnessReturn
	var.fitness_positive, var.fitness_negative, var.fitness_neutral = is_positive, is_negative, is_neutral
	var.CICLES_UNCHANGED, var.GENERATIONS_UNCHANGED = 0, 0


def saveBestValues(individual, accuracy, precision, recall, f1):
	""" Save the best individuals of each metric
		
		Args:
			individual: individual to be evaluated
			accuracy, precision, recall, f1: array of metrics (each of them has positive/negative/neutral indexes)
	"""
	if accuracy > var.BEST['accuracy']:
		var.BEST['accuracy'] = accuracy     

	if precision['positive'] > var.BEST['precision']['positive']:
		var.BEST['precision']['positive'] = precision['positive']

	if precision['negative'] > var.BEST['precision']['negative']:
		var.BEST['precision']['negative'] = precision['negative']
	
	if precision['neutral'] > var.BEST['precision']['neutral']:
		var.BEST['precision']['neutral'] = precision['neutral']

	if precision['avg'] > var.BEST['precision']['avg']:
		var.BEST['precision']['avg'] = precision['avg']
		var.BEST['precision']['avg_function'] = str(individual)        

	if recall['positive'] > var.BEST['recall']['positive']:
		var.BEST['recall']['positive'] = recall['positive']

	if recall['negative'] > var.BEST['recall']['negative']:
		var.BEST['recall']['negative'] = recall['negative']

	if recall['neutral'] > var.BEST['recall']['neutral']:
		var.BEST['recall']['neutral'] = recall['neutral']

	if recall['avg'] > var.BEST['recall']['avg']:
		var.BEST['recall']['avg'] = recall['avg']
		var.BEST['recall']['avg_function'] = str(individual)        

	if f1['positive'] > var.BEST['f1']['positive']:
		var.BEST['f1']['positive'] = f1['positive']
   
	if f1['negative'] > var.BEST['f1']['negative']:
		var.BEST['f1']['negative'] = f1['negative']
   
	if f1['neutral'] > var.BEST['f1']['neutral']:
		var.BEST['f1']['neutral'] = f1['neutral']         
	
	if f1['avg'] > var.BEST['f1']['avg']:
		var.BEST['f1']['avg'] = f1['avg']
		var.BEST['f1']['avg_function'] = str(individual)           

	if f1['avg_pn'] > var.BEST['f1']['avg_pn']:
		var.BEST['f1']['avg_pn'] = f1['avg_pn']


def logFinalResults(function):
	""" Log the final results, after all the cicles

		Arg:
			function (individual): the individual that represents the best model

		TO-DO: use str.format() to the logs
	"""
	print("\n## Results ##\n")
	print("[total tweets]: " + str(var.POSITIVE_MESSAGES + var.NEGATIVE_MESSAGES + var.NEUTRAL_MESSAGES) + " [" + str(var.POSITIVE_MESSAGES) + " positives, " + str(var.NEGATIVE_MESSAGES) + " negatives and " + str(var.NEUTRAL_MESSAGES) + " neutrals]\n")
	print("[best fitness (F1 avg (+/-)]: " + str(var.BEST['fitness']) + " [" + str(var.fitness_positive + var.fitness_negative + var.fitness_neutral) + " correct evaluations] ["+ str(var.fitness_positive) + " positives, " + str(var.fitness_negative) + " negatives and " + str(var.fitness_neutral) + " neutrals]\n")
	print("[function]: " + str(function) + "\n")
	print("[best accuracy]: "           + str(var.BEST['accuracy']) + "\n")
	print("[best precision positive]: " + str(var.BEST['precision']['positive']))
	print("[best precision negative]: " + str(var.BEST['precision']['negative']))
	print("[best precision neutral]: "  + str(var.BEST['precision']['neutral']))    
	print("[best precision avg]: "      + str(var.BEST['precision']['avg']))
	print("[best precision avg function]: " + var.BEST['precision']['avg_function'] + "\n")    
	print("[best recall positive]: "    + str(var.BEST['recall']['positive']))    
	print("[best recall negative]: "    + str(var.BEST['recall']['negative']))
	print("[best recall negative]: "    + str(var.BEST['recall']['neutral']))
	print("[best recall avg]: "         + str(var.BEST['recall']['avg']))
	print("[best recall avg function]: "+     var.BEST['recall']['avg_function'] + "\n")
	print("[best f1 positive]: "        + str(var.BEST['f1']['positive']))    
	print("[best f1 negative]: "        + str(var.BEST['f1']['negative']))
	print("[best f1 avg]: "             + str(var.BEST['f1']['avg']))
	print("[best f1 avg (+/-)]: "       + str(var.BEST['f1']['avg_pn']))
	print("[best f1 avg function]: "    +     var.BEST['f1']['avg_function'])
	print("[best fitness history]: "    + str(var.HISTORY['fitness']['best']) + "\n\n")


def logCicleValues(start_time, correct_evaluations, individual, accuracy, precision, recall, f1, fitnessReturn, conf_matrix):
	""" Log the cicle results

		Args:
			correct_evaluations: number of correct evaluations
			individual: individual to be evaluated
			accuracy, precision, recall, f1: array of metrics (each of them has positive/negative/neutral indexes)
			fitnessReturn: numeric fitness
			conf_matrix: confusion matrix
	"""	
	if var.LOG['all_each_cicle']:
		print("[correct evaluations] " + str(correct_evaluations))
		print('{message: <{width}}'.format(message="[accuracy] ",   width=18) + " -> " + str(round(accuracy, 3)))
		print('{message: <{width}}'.format(message="[precision] ",  width=18) + " -> " + "[positive]: " + '{message: <{width}}'.format(message=str(round(precision['positive'], 3)), width=6) + " " + "[negative]: " + '{message: <{width}}'.format(message=str(round(precision['negative'], 3)), width=6) + " " + "[neutral]: " + '{message: <{width}}'.format(message=str(round(precision['neutral'], 3)), width=6) + " " + "[avg]: " + '{message: <{width}}'.format(message=str(round(precision['avg'], 3)), width=6))
		print('{message: <{width}}'.format(message="[recall] ",     width=18) + " -> " + "[positive]: " + '{message: <{width}}'.format(message=str(round(recall['positive'], 3)), width=6) + " " + "[negative]: " + '{message: <{width}}'.format(message=str(round(recall['negative'], 3)), width=6) + " " + "[neutral]: " + '{message: <{width}}'.format(message=str(round(recall['neutral'], 3)), width=6) + " " + "[avg]: " + '{message: <{width}}'.format(message=str(round(recall['avg'], 3)), width=6))
		print('{message: <{width}}'.format(message="[f1] ",         width=18) + " -> " + "[positive]: " + '{message: <{width}}'.format(message=str(round(f1['positive'], 3)), width=6) + " " + "[negative]: " + '{message: <{width}}'.format(message=str(round(f1['negative'], 3)), width=6) + " " + "[neutral]: " + '{message: <{width}}'.format(message=str(round(f1['neutral'], 3)), width=6) + " " + "[avg]: " + '{message: <{width}}'.format(message=str(round(f1['avg'], 3)), width=6))
		print('{message: <{width}}'.format(message="[f1 SemEval] ", width=18) + " -> " + str(round(f1['avg_pn'], 3)))
	
	print('{message: <{width}}'.format(message="[fitness] "     ,    width=18) + " -> " + str(round(fitnessReturn, 5)) + " ****")
	print('{message: <{width}}'.format(message="[best fitness] ",    width=18) + " -> " + str(round(var.BEST['fitness'], 5)))
	print('{message: <{width}}'.format(message="[confusion matrix]", width=18) + " -> " + "[true_positive]: " + str(conf_matrix['true_positive']) + " " + "[false_positive]: " + str(conf_matrix['false_positive']) + " " + "[true_negative]: " + str(conf_matrix['true_negative']) + " " + "[false_negative]: " + str(conf_matrix['false_negative']) + " " + "[true_neutral]: " + str(conf_matrix['true_neutral']) + " " + "[false_neutral]: " + str(conf_matrix['false_neutral']) + "\n")

	if var.LOG['all_each_cicle']:
		print('{message: <{width}}'.format(message="[cicles unmodified]", width=24) + " -> " + str(var.CICLES_UNCHANGED))
	
	print('{message: <{width}}'.format(message="[generations unmodified]", width=24) + " -> " + str(var.GENERATIONS_UNCHANGED))
	print("[function]: " + str(individual))
	
	if var.LOG['times']:
		print("[cicle ends after " + str(format(time.time() - start_time, '.3g')) + " seconds]")     
	print("-----------------------------\n")
		

def logConstraint(start_time, constraint_type):
	""" Log the constraints

		The GP algorithm has some constraints
			* repeated massive functions
			* neutral range function
			* tree root function

		Args:
			start_time: used to calculate the total time
			constraint_type: the type of the constraint
	"""
	if constraint_type == 'massive_function_more_than_one':
		print("\n[CONSTRAINT][more than " + str(var.CONSTRAINT['massive']['max']) + " massive(s) function(s)][bad individual][fitness zero]\n")
		if var.LOG['times']:
			print("[cicle ends after " + str(format(time.time() - start_time, '.3g')) + " seconds]")
		print("-----------------------------\n")

	elif constraint_type == 'neutral_range_more_than_one':
		print("\n[CONSTRAINT][more than one neutralRange function][bad individual][fitness zero]\n")
		if var.LOG['times']:
			print("[cicle ends after " + str(format(time.time() - start_time, '.3g')) + " seconds]")     
		print("-----------------------------\n")
	
	elif constraint_type == 'neutral_range_inexistent':
		print("\n[CONSTRAINT][model does not have neutralRange function][fitness decreased in " + str(var.CONSTRAINT['root']['decrease_rate'] * 100) + "%]\n")

	elif constraint_type == 'root_function':
		print("\n[CONSTRAINT][root node is not " + var.CONSTRAINT['root']['function'] + "][fitness decreased in " + str(var.CONSTRAINT['root']['decrease_rate'] * 100) + "%]\n")


def plotResults():
	X = range(len(var.HISTORY['fitness']['per_generation']))
	Y = var.HISTORY['fitness']['per_generation']
	plt.plot(X,Y)
	plt.savefig(var.TRAIN_RESULTS_IMG + "-all.png", bbox_inches='tight') # all models in one image
	plt.savefig(var.TRAIN_RESULTS_IMG + str(datetime.now())[14:16] + str(datetime.now())[17:19] + ".png", bbox_inches='tight') # one image per model


def resetVariables():
	var.fitness_positive, var.fitness_negative, var.fitness_neutral = 0, 0, 0
	var.CICLES_UNCHANGED, var.GENERATIONS_UNCHANGED = 0, 0
	var.generations_unchanged_reached_msg = False

	var.BEST = {'fitness': 0, 'accuracy': 0, 'precision': {}, 'recall': {}, 'f1': {}}
	var.BEST['precision']  = {'positive': 0, 'negative': 0, 'neutral': 0, 'avg': 0, 'avg_function': ""}
	var.BEST['recall']     = {'positive': 0, 'negative': 0, 'neutral': 0, 'avg': 0, 'avg_function': ""}
	var.BEST['f1']         = {'positive': 0, 'negative': 0, 'neutral': 0, 'avg': 0, 'avg_function': "", 'avg_pn': 0}
	var.HISTORY['fitness'] = {'all': [], 'per_generation': [], 'best': []} 
	var.MODEL['others']    = []


def saveModel(file_path):
	with open(file_path, 'a') as f:
		f.write(str(var.MODEL['bests'][len(var.MODEL['bests']) - 1]) + "\n")
		
		if not var.SAVE['only_best']:
			for m in var.MODEL['others']:
				f.write(str(m) + "\n")
			f.write("\n")


def logEndSystem(file_path, start_time):
	print("[SYSTEM ENDS AFTER " + str(format(time.time() - start_time, '.3g')) + " SECONDS]")
	print("[RESULTS SAVED ON]: " + file_path)
	print("\n[CONTACTS]")
	print("  [PROBLEMS/SUGGESTIONS]: airtonbjunior@gmail.com")
	print("  [REPOSITORY]:           https://github.com/airtonbjunior/opinionMining/")