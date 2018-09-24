# Airton Bordin Junior
# airtonbjunior@gmail.com
# Federal University of Goias (UFG)
# Computer Science Master's Degree

import time
import operator
import random
import sys

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

import variables
from functions import *
from classifier import *
from sendMail import *

INDIVIDUAL_SIZE = 0 # Individual size - number of loaded dictionaries (one weight per dictionary)
MODEL = ""

toolbox = base.Toolbox()

creator.create("FitnessMaxAG", base.Fitness, weights=(1.0,))
creator.create("IndividualAG", list, fitness=creator.FitnessMaxAG) # List?

# Individual evaluation
def evaluateIndividual(individual):
	print(str(individual))
	
	variables.ag_w1 = individual[0]
	variables.ag_w2 = individual[1]
	variables.ag_w3 = individual[2]
	variables.ag_w4 = individual[3]
	variables.ag_w5 = individual[4]
	variables.ag_w6 = individual[5]
	variables.ag_w7 = individual[6]
	variables.ag_w8 = individual[7]
	variables.ag_w9 = individual[8]

	variables.calling_by_ag_file = True
	
	fitnessReturn = evalSymbRegTweetsFromSemeval(MODEL)
	print("\n[FITNESS RETURN] " + str(fitnessReturn) + "\n")

	# func = toolbox.compile(expr=individual)

	# fitnessReturn = sum(individual)

	variables.calling_by_ag_file = False
	variables.ag_w1 = 0
	variables.ag_w2 = 0
	variables.ag_w3 = 0
	variables.ag_w4 = 0
	variables.ag_w5 = 0
	variables.ag_w6 = 0
	variables.ag_w7 = 0
	variables.ag_w8 = 0
	variables.ag_w9 = 0

	return int(fitnessReturn),


def main():
	random.seed()
	loadModel() # Get the model and the number of dictionaries (individual size)

	getDictionary("train")
	loadTrainTweets()

	# Register the parameters
	toolbox.register("evaluate", evaluateIndividual)
	toolbox.register("mate", tools.cxTwoPoint)
	toolbox.register("mutate", tools.mutFlipBit, indpb=0.05) # see if this is correct - if the indpb is the mutation
	toolbox.register("select", tools.selTournament, tournsize=3)
	toolbox.register("attr_bool", random.uniform, -3, 3) # possible weights
	toolbox.register("individual", tools.initRepeat, creator.IndividualAG, toolbox.attr_bool, INDIVIDUAL_SIZE)
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)
	# Register the parameters

	pop = toolbox.population(n=variables.AG_POPULATION)

	# Evaluate the entire population
	fitnesses = list(map(toolbox.evaluate, pop))
	for ind, fit in zip(pop, fitnesses):
	    ind.fitness.values = fit

	CXPB, MUTPB = variables.AG_CROSSOVER, variables.AG_MUTATION

	fits = [ind.fitness.values[0] for ind in pop]

	g = 0
	# while fitness (F1-score) is not 100 and limit generation not reached
	while max(fits) < 100 and g < variables.AG_GENERATIONS:
		g = g + 1

		offspring = toolbox.select(pop, len(pop))
		offspring = list(map(toolbox.clone, offspring))

		# Apply crossover and mutation on the offspring
		for child1, child2 in zip(offspring[::2], offspring[1::2]):

		    # cross two individuals with probability CXPB
		    if random.random() < CXPB:
		        toolbox.mate(child1, child2)

		        # fitness values of the children
		        # must be recalculated later
		        del child1.fitness.values
		        del child2.fitness.values

		for mutant in offspring:

		    # mutate an individual with probability MUTPB
		    if random.random() < MUTPB:
		        toolbox.mutate(mutant)
		        del mutant.fitness.values

		# Evaluate the individuals with an invalid fitness
		invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
		fitnesses = map(toolbox.evaluate, invalid_ind)
		for ind, fit in zip(invalid_ind, fitnesses):
		    ind.fitness.values = fit

		# The population is entirely replaced by the offspring
		pop[:] = offspring

		# Gather all the fitnesses in one list and print the stats
		fits = [ind.fitness.values[0] for ind in pop]        

	best_ind = tools.selBest(pop, 1)[0]
	print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))


# --------------------------------
# Load the model by file 
# TO-DO: load more than one model?
def loadModel():
	import tkinter as tk
	from tkinter import filedialog

	global INDIVIDUAL_SIZE
	global MODEL

	# Pick the file with the model
	root = tk.Tk()
	root.withdraw()
	file_path = filedialog.askopenfilename()

	with open(file_path, 'r') as inF:
		for line in inF:
		    if line.startswith("[DICTIONARIES]"):
		    	INDIVIDUAL_SIZE = int(line.split(":")[1].strip())
		    	print(str(INDIVIDUAL_SIZE))
		    elif not line.startswith("[") and line not in ('\n', '\r\n') and not line.startswith("#"):
		    	MODEL = str(line)
		    	#print(str(line))
		    	


if __name__ == "__main__":                
	main()