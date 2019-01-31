""" 
<Author> Airton Bordin Junior
<Email>  airtonbjunior@gmail.com

Federal University of Goias (UFG)
Computer Science Master's Degree


auxFunctions.py
	Aux Functions used by the Genetic Programming Algorithm

"""
import variables
import matplotlib.pyplot as plt
from datetime import datetime

def saveBestValues(individual, accuracy, precision, recall, f1):
	""" Save the best individuals of each metric
		
		Args:
			individual: individual to be evaluated
			accuracy, precision, recall, f1: array of metrics (each of them has positive/negative/neutral indexes)
	"""
	if accuracy > variables.BEST['accuracy']:
		variables.BEST['accuracy'] = accuracy     

	if precision['positive'] > variables.BEST['precision']['positive']:
		variables.BEST['precision']['positive'] = precision['positive']

	if precision['negative'] > variables.BEST['precision']['negative']:
		variables.BEST['precision']['negative'] = precision['negative']
	
	if precision['neutral'] > variables.BEST['precision']['neutral']:
		variables.BEST['precision']['neutral'] = precision['neutral']

	if precision['avg'] > variables.BEST['precision']['avg']:
		variables.BEST['precision']['avg'] = precision['avg']
		variables.BEST['precision']['avg_function'] = str(individual)        

	if recall['positive'] > variables.BEST['recall']['positive']:
		variables.BEST['recall']['positive'] = recall['positive']

	if recall['negative'] > variables.BEST['recall']['negative']:
		variables.BEST['recall']['negative'] = recall['negative']

	if recall['neutral'] > variables.BEST['recall']['neutral']:
		variables.BEST['recall']['neutral'] = recall['neutral']

	if recall['avg'] > variables.BEST['recall']['avg']:
		variables.BEST['recall']['avg'] = recall['avg']
		variables.BEST['recall']['avg_function'] = str(individual)        

	if f1['positive'] > variables.BEST['f1']['positive']:
		variables.BEST['f1']['positive'] = f1['positive']
   
	if f1['negative'] > variables.BEST['f1']['negative']:
		variables.BEST['f1']['negative'] = f1['negative']
   
	if f1['neutral'] > variables.BEST['f1']['neutral']:
		variables.BEST['f1']['neutral'] = f1['neutral']         
	
	if f1['avg'] > variables.BEST['f1']['avg']:
		variables.BEST['f1']['avg'] = f1['avg']
		variables.BEST['f1']['avg_function'] = str(individual)           

	if f1['avg_pn'] > variables.BEST['f1']['avg_pn']:
		variables.BEST['f1']['avg_pn'] = f1['avg_pn']


def logFinalResults(function):
	""" Log the final results, after all the cicles

		Arg:
			function (individual): the individual that represents the best model
	"""
	print("\n## Results ##\n")
	print("[total tweets]: " + str(variables.POSITIVE_MESSAGES + variables.NEGATIVE_MESSAGES + variables.NEUTRAL_MESSAGES) + " [" + str(variables.POSITIVE_MESSAGES) + " positives, " + str(variables.NEGATIVE_MESSAGES) + " negatives and " + str(variables.NEUTRAL_MESSAGES) + " neutrals]\n")
	print("[best fitness (F1 avg (+/-)]: " + str(variables.BEST['fitness']) + " [" + str(variables.fitness_positive + variables.fitness_negative + variables.fitness_neutral) + " correct evaluations] ["+ str(variables.fitness_positive) + " positives, " + str(variables.fitness_negative) + " negatives and " + str(variables.fitness_neutral) + " neutrals]\n")
	print("[function]: " + str(function) + "\n")
	print("[best accuracy]: "           + str(variables.BEST['accuracy']) + "\n")
	print("[best precision positive]: " + str(variables.BEST['precision']['positive']))
	print("[best precision negative]: " + str(variables.BEST['precision']['negative']))
	print("[best precision neutral]: "  + str(variables.BEST['precision']['neutral']))    
	print("[best precision avg]: "      + str(variables.BEST['precision']['avg']))
	print("[best precision avg function]: " + variables.BEST['precision']['avg_function'] + "\n")    
	print("[best recall positive]: "    + str(variables.BEST['recall']['positive']))    
	print("[best recall negative]: "    + str(variables.BEST['recall']['negative']))
	print("[best recall negative]: "    + str(variables.BEST['recall']['neutral']))
	print("[best recall avg]: "         + str(variables.BEST['recall']['avg']))
	print("[best recall avg function]: "+     variables.BEST['recall']['avg_function'] + "\n")
	print("[best f1 positive]: "        + str(variables.BEST['f1']['positive']))    
	print("[best f1 negative]: "        + str(variables.BEST['f1']['negative']))
	print("[best f1 avg]: "             + str(variables.BEST['f1']['avg']))
	print("[best f1 avg (+/-)]: "       + str(variables.BEST['f1']['avg_pn']))
	print("[best f1 avg function]: "    +     variables.BEST['f1']['avg_function'])
	print("[best fitness history]: "    + str(variables.HISTORY['fitness']['best']) + "\n\n")


def logCicleValues(correct_evaluations, individual, accuracy, precision, recall, f1, fitnessReturn, conf_matrix):
	""" Log the cicle results

		Args:
			correct_evaluations: number of correct evaluations
			individual: individual to be evaluated
			accuracy, precision, recall, f1: array of metrics (each of them has positive/negative/neutral indexes)
			fitnessReturn: numeric fitness
			conf_matrix: confusion matrix
	"""	
	if variables.LOG['all_each_cicle']:
		print("[correct evaluations] " + str(correct_evaluations))
		print('{message: <{width}}'.format(message="[accuracy] ",   width=18) + " -> " + str(round(accuracy, 3)))
		print('{message: <{width}}'.format(message="[precision] ",  width=18) + " -> " + "[positive]: " + '{message: <{width}}'.format(message=str(round(precision['positive'], 3)), width=6) + " " + "[negative]: " + '{message: <{width}}'.format(message=str(round(precision['negative'], 3)), width=6) + " " + "[neutral]: " + '{message: <{width}}'.format(message=str(round(precision['neutral'], 3)), width=6) + " " + "[avg]: " + '{message: <{width}}'.format(message=str(round(precision['avg'], 3)), width=6))
		print('{message: <{width}}'.format(message="[recall] ",     width=18) + " -> " + "[positive]: " + '{message: <{width}}'.format(message=str(round(recall['positive'], 3)), width=6) + " " + "[negative]: " + '{message: <{width}}'.format(message=str(round(recall['negative'], 3)), width=6) + " " + "[neutral]: " + '{message: <{width}}'.format(message=str(round(recall['neutral'], 3)), width=6) + " " + "[avg]: " + '{message: <{width}}'.format(message=str(round(recall['avg'], 3)), width=6))
		print('{message: <{width}}'.format(message="[f1] ",         width=18) + " -> " + "[positive]: " + '{message: <{width}}'.format(message=str(round(f1['positive'], 3)), width=6) + " " + "[negative]: " + '{message: <{width}}'.format(message=str(round(f1['negative'], 3)), width=6) + " " + "[neutral]: " + '{message: <{width}}'.format(message=str(round(f1['neutral'], 3)), width=6) + " " + "[avg]: " + '{message: <{width}}'.format(message=str(round(f1['avg'], 3)), width=6))
		print('{message: <{width}}'.format(message="[f1 SemEval] ", width=18) + " -> " + str(round(f1['avg_pn'], 3)))
	
	print('{message: <{width}}'.format(message="[fitness] "     ,    width=18) + " -> " + str(round(fitnessReturn, 5)) + " ****")
	print('{message: <{width}}'.format(message="[best fitness] ",    width=18) + " -> " + str(round(variables.BEST['fitness'], 5)))
	print('{message: <{width}}'.format(message="[confusion matrix]", width=18) + " -> " + "[true_positive]: " + str(conf_matrix['true_positive']) + " " + "[false_positive]: " + str(conf_matrix['false_positive']) + " " + "[true_negative]: " + str(conf_matrix['true_negative']) + " " + "[false_negative]: " + str(conf_matrix['false_negative']) + " " + "[true_neutral]: " + str(conf_matrix['true_neutral']) + " " + "[false_neutral]: " + str(conf_matrix['false_neutral']) + "\n")

	if variables.LOG['all_each_cicle']:
		print('{message: <{width}}'.format(message="[cicles unmodified]", width=24) + " -> " + str(variables.CICLES_UNCHANGED))
	
	print('{message: <{width}}'.format(message="[generations unmodified]", width=24) + " -> " + str(variables.GENERATIONS_UNCHANGED))
	print("[function]: " + str(individual))
		

def plotResults():
	X = range(len(variables.HISTORY['fitness']['per_generation']))
	Y = variables.HISTORY['fitness']['per_generation']
	plt.plot(X,Y)
	plt.savefig(variables.TRAIN_RESULTS_IMG + "-all.png", bbox_inches='tight') # all models in one image
	plt.savefig(variables.TRAIN_RESULTS_IMG + str(datetime.now())[14:16] + str(datetime.now())[17:19] + ".png", bbox_inches='tight') # one image per model


def resetVariables():
	variables.fitness_positive, variables.fitness_negative, variables.fitness_neutral = 0, 0, 0
	variables.CICLES_UNCHANGED, variables.GENERATIONS_UNCHANGED = 0, 0
	variables.generations_unchanged_reached_msg = False

	variables.BEST = {'fitness': 0, 'accuracy': 0, 'precision': {}, 'recall': {}, 'f1': {}}
	variables.BEST['precision']  = {'positive': 0, 'negative': 0, 'neutral': 0, 'avg': 0, 'avg_function': ""}
	variables.BEST['recall']     = {'positive': 0, 'negative': 0, 'neutral': 0, 'avg': 0, 'avg_function': ""}
	variables.BEST['f1']         = {'positive': 0, 'negative': 0, 'neutral': 0, 'avg': 0, 'avg_function': "", 'avg_pn': 0}
	variables.HISTORY['fitness'] = {'all': [], 'per_generation': [], 'best': []} 
	variables.MODEL['others'] = []