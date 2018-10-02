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
#from classifier import *
from sendMail import *

INDIVIDUAL_SIZE = 0 # Individual size - number of loaded dictionaries (one weight per dictionary)
MODEL = ""
AG_INDIVIDUAL = []

toolbox = base.Toolbox()

creator.create("FitnessMaxAG", base.Fitness, weights=(1.0,))
creator.create("IndividualAG", list, fitness=creator.FitnessMaxAG) # List?

# Individual evaluation
def evaluateIndividual(individual):
	print(str(individual))
	
	global AG_INDIVIDUAL
	AG_INDIVIDUAL = individual

	variables.ag_w1 = individual[0]
	variables.ag_w2 = individual[1]
	variables.ag_w3 = individual[2]
	variables.ag_w4 = individual[3]
	variables.ag_w5 = individual[4]
	variables.ag_w6 = individual[5]
	variables.ag_w7 = individual[6]
	variables.ag_w8 = individual[7]
	variables.ag_w9 = individual[8]

	variables.neutral_inferior_range = individual[9]
	variables.neutral_superior_range = individual[10]

	variables.calling_by_ag_file = True
	
	fitnessReturn = evaluateAG(MODEL) # TO-DO: get the GA parameter instead of GP
	print("\n[FITNESS RETURN] " + str(fitnessReturn) + "\n")

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

	variables.neutral_inferior_range = 0
	variables.neutral_superior_range = 0

	return float(fitnessReturn),



iterate_count = 1
generation_count = 1
best_of_generation = 0
evaluation_acumulated_time = 0

def evaluateAG(individual):
    start = time.time()
    global iterate_count
    global generation_count
    global best_of_generation
    new_generation = False

    # test
    #variables.neutral_inferior_range = 0
    #variables.neutral_superior_range = 0
    # test

    # Check max unchanged generations
    if variables.generations_unchanged >= variables.max_unchanged_generations:
        if(variables.generations_unchanged_reached_msg == False):
            print("[Max unchanged generations (" + str(variables.max_unchanged_generations) + ") reached on generation " + str(generation_count) + "]")
        variables.generations_unchanged_reached_msg = True
        return 0,

    # Log the number of each individual
    if iterate_count <= variables.AG_POPULATION:
        print("[individual " + str(iterate_count) + " of the generation " + str(generation_count) + "]")
        iterate_count += 1
    else:
        generation_count += 1
        iterate_count = 1
        variables.best_fitness_per_generation_history.append(best_of_generation)
        print("\n[new generation][start generation " + str(generation_count) + "]\n")
        new_generation = True
        best_of_generation = 0

    global evaluation_acumulated_time
    correct_evaluations = 0

    fitnessReturn = 0

    is_positive = 0
    is_negative = 0
    is_neutral  = 0
    # parameters to calc the metrics
    true_positive  = 0
    true_negative  = 0
    true_neutral   = 0

    false_positive = 0
    false_negative = 0
    false_neutral  = 0

    accuracy = 0

    precision_positive = 0
    precision_negative = 0
    precision_neutral  = 0
    precision_avg = 0

    recall_positive = 0
    recall_negative = 0
    recall_neutral  = 0
    recall_avg = 0

    f1_positive = 0
    f1_negative = 0
    f1_neutral  = 0
    f1_avg = 0
    f1_positive_negative_avg = 0

    func_value = 0

    # Constraint controls 
    breaked = False
    fitness_decreased = False
    double_decreased = False

    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    
    # to work using another databases
    TWEETS_TO_EVALUATE       = variables.tweets_semeval
    TWEETS_SCORE_TO_EVALUATE = variables.tweets_semeval_score

    if variables.neutral_inferior_range > variables.neutral_superior_range:
    	print("\n[CONSTRAINT][inferior range grater than superior][fitness zero][TEMPORARY]")
    	return 0

    # Main loop
    # for index, item in enumerate(variables.tweets_semeval):
    for index, item in enumerate(TWEETS_TO_EVALUATE):

        # Log each new cicle
        if index == 0:
            if variables.log_all_metrics_each_cicle:
                #print("\n[New cicle]: " + str(len(variables.tweets_semeval)) + " phrases to evaluate [" + str(variables.positive_tweets) + " positives, " + str(variables.negative_tweets) + " negatives and " + str(variables.neutral_tweets) + " neutrals]")
                print("\n[New cicle]: " + str(len(TWEETS_TO_EVALUATE)) + " phrases to evaluate [" + str(variables.positive_tweets) + " positives, " + str(variables.negative_tweets) + " negatives and " + str(variables.neutral_tweets) + " neutrals]")

        try:
            #func_value = float(func(variables.tweets_semeval[index]))
            func_value = float(func(TWEETS_TO_EVALUATE[index]))
            

            #if float(variables.tweets_semeval_score[index]) > 0:
            if float(TWEETS_SCORE_TO_EVALUATE[index]) > 0:
                if  func_value > variables.neutral_superior_range:
                    correct_evaluations += 1 
                    is_positive   += 1
                    true_positive += 1
                else:
                    if func_value >= variables.neutral_inferior_range and func_value <= variables.neutral_superior_range:
                        false_neutral += 1
                    elif func_value < variables.neutral_inferior_range:
                        false_negative += 1

            #elif float(variables.tweets_semeval_score[index]) < 0:
            elif float(TWEETS_SCORE_TO_EVALUATE[index]) < 0:
                if func_value < variables.neutral_inferior_range:
                    correct_evaluations += 1 
                    is_negative   += 1
                    true_negative += 1
                else:
                    if func_value >= variables.neutral_inferior_range and func_value <= variables.neutral_superior_range:
                        false_neutral += 1
                    elif func_value > variables.neutral_superior_range:
                        false_positive += 1

            #elif float(variables.tweets_semeval_score[index]) == 0:
            elif float(TWEETS_SCORE_TO_EVALUATE[index]) == 0:
                if func_value >= variables.neutral_inferior_range and func_value <= variables.neutral_superior_range:
                    correct_evaluations += 1 
                    is_neutral   += 1
                    true_neutral += 1
                else:
                    if func_value < variables.neutral_inferior_range:
                        false_negative += 1
                    elif func_value > variables.neutral_superior_range:
                        false_positive += 1

        except Exception as e: 
            print(e)
            continue

        #logs
        if variables.log_all_messages:
            #print("[phrase]: " + variables.tweets_semeval[index])
            print("[phrase]: " + TWEETS_TO_EVALUATE[index])
            #print("[value]: " + str(variables.tweets_semeval_score[index]))
            print("[value]: " + str(TWEETS_SCORE_TO_EVALUATE[index]))
            print("[calculated]:" + func_value)


    if true_positive + false_positive + true_negative + false_negative > 0:
        accuracy = (true_positive + true_negative + true_neutral) / (true_positive + false_positive + true_negative + false_negative + true_neutral + false_neutral)

    if accuracy > variables.best_accuracy:
        variables.best_accuracy = accuracy 

    # Begin PRECISION
    if true_positive + false_positive > 0:
        precision_positive = true_positive / (true_positive + false_positive)
        if precision_positive > variables.best_precision_positive:
            variables.best_precision_positive = precision_positive


    if true_negative + false_negative > 0:
        precision_negative = true_negative / (true_negative + false_negative)
        if precision_negative > variables.best_precision_negative:
            variables.best_precision_negative = precision_negative
    

    if true_neutral + false_neutral > 0:
        precision_neutral = true_neutral / (true_neutral + false_neutral)
        if precision_neutral > variables.best_precision_neutral:
            variables.best_precision_neutral = precision_neutral
    # End PRECISION

    # Begin RECALL
    if variables.positive_tweets > 0:
        recall_positive = true_positive / variables.positive_tweets
        if recall_positive > variables.best_recall_positive:
            variables.best_recall_positive = recall_positive


    if variables.negative_tweets > 0:
        recall_negative = true_negative / variables.negative_tweets
        if recall_negative > variables.best_recall_negative:
            variables.best_recall_negative = recall_negative

    if variables.neutral_tweets > 0:
        recall_neutral = true_neutral / variables.neutral_tweets
        if recall_neutral > variables.best_recall_neutral:
            variables.best_recall_neutral = recall_neutral
    # End RECALL

    # Begin F1
    if precision_positive + recall_positive > 0:
        f1_positive = 2 * (precision_positive * recall_positive) / (precision_positive + recall_positive)
        if f1_positive > variables.best_f1_positive:
            variables.best_f1_positive = f1_positive


    if precision_negative + recall_negative > 0:
        f1_negative = 2 * (precision_negative * recall_negative) / (precision_negative + recall_negative)        
        if f1_negative > variables.best_f1_negative:
            variables.best_f1_negative = f1_negative

    if precision_neutral + recall_neutral > 0:
        f1_neutral = 2 * (precision_neutral * recall_neutral) / (precision_neutral + recall_neutral)        
        if f1_neutral > variables.best_f1_neutral:
            variables.best_f1_neutral = f1_neutral            
    # End F1

    # Precision, Recall and f1 means
    precision_avg = (precision_positive + precision_negative + precision_neutral) / 3
    if precision_avg > variables.best_precision_avg:
        variables.best_precision_avg = precision_avg
        variables.best_precision_avg_function = str(individual)

    recall_avg = (recall_positive + recall_negative + recall_neutral) / 3
    if recall_avg > variables.best_recall_avg:
        variables.best_recall_avg = recall_avg
        variables.best_recall_avg_function = str(individual)

    f1_avg = (f1_positive + f1_negative + f1_neutral) / 3
    if f1_avg > variables.best_f1_avg:
        variables.best_f1_avg = f1_avg
        variables.best_f1_avg_function = str(individual)

    f1_positive_negative_avg = (f1_positive + f1_negative) / 2
    if f1_positive_negative_avg > variables.best_f1_positive_negative_avg:
        variables.best_f1_positive_negative_avg = f1_positive_negative_avg

    # The metric that represent the fitness
    # fitnessReturn = accuracy
    fitnessReturn = f1_positive_negative_avg
    if fitness_decreased:
        fitnessReturn -= fitnessReturn * variables.root_decreased_value # 80% of the original value
    if double_decreased:
        fitnessReturn -= fitnessReturn * variables.root_decreased_value # Again

    global AG_INDIVIDUAL

    if variables.best_fitness < fitnessReturn:
        if variables.best_fitness != 0:
            # save partial best individual (in case we need stop evolution)
            with open("AG_" + variables.BEST_INDIVIDUAL_AG, 'w') as f:
                #f.write(str(individual) + "\n")
                f.write(str(AG_INDIVIDUAL))
                f.write("\n\n# Generation -> " + str(generation_count))
                f.write("\n# Neutral Range -> [" + str(variables.neutral_inferior_range) + ", " + str(variables.neutral_superior_range) + "]")
            variables.best_fitness_history.append(variables.best_fitness)
        variables.best_fitness = fitnessReturn
        variables.fitness_positive = is_positive
        variables.fitness_negative = is_negative
        variables.fitness_neutral  = is_neutral
        is_positive = 0
        is_negative = 0
        is_neutral  = 0
        variables.cicles_unchanged = 0
        variables.generations_unchanged = 0
    else:
        variables.cicles_unchanged += 1
        if new_generation:
            variables.generations_unchanged += 1


    if not new_generation and best_of_generation < fitnessReturn:
        best_of_generation = fitnessReturn


    variables.all_fitness_history.append(fitnessReturn)


    #logs   
    if variables.log_parcial_results and not breaked:# and not variables.calling_by_ag_file: 
        if variables.log_all_metrics_each_cicle:
            print("[correct evaluations] " + str(correct_evaluations))
            print('{message: <{width}}'.format(message="[accuracy] ", width=18) + " -> " + str(round(accuracy, 3)))
            print('{message: <{width}}'.format(message="[precision] ", width=18) + " -> " + "[positive]: " + '{message: <{width}}'.format(message=str(round(precision_positive, 3)), width=6) + " " + "[negative]: " + '{message: <{width}}'.format(message=str(round(precision_negative, 3)), width=6) + " " + "[neutral]: " + '{message: <{width}}'.format(message=str(round(precision_neutral, 3)), width=6) + " " + "[avg]: " + '{message: <{width}}'.format(message=str(round(precision_avg, 3)), width=6))
            print('{message: <{width}}'.format(message="[recall] ", width=18) + " -> " + "[positive]: " + '{message: <{width}}'.format(message=str(round(recall_positive, 3)), width=6) + " " + "[negative]: " + '{message: <{width}}'.format(message=str(round(recall_negative, 3)), width=6) + " " + "[neutral]: " + '{message: <{width}}'.format(message=str(round(recall_neutral, 3)), width=6) + " " + "[avg]: " + '{message: <{width}}'.format(message=str(round(recall_avg, 3)), width=6))
            print('{message: <{width}}'.format(message="[f1] ", width=18) + " -> " + "[positive]: " + '{message: <{width}}'.format(message=str(round(f1_positive, 3)), width=6) + " " + "[negative]: " + '{message: <{width}}'.format(message=str(round(f1_negative, 3)), width=6) + " " + "[neutral]: " + '{message: <{width}}'.format(message=str(round(f1_neutral, 3)), width=6) + " " + "[avg]: " + '{message: <{width}}'.format(message=str(round(f1_avg, 3)), width=6))
            print('{message: <{width}}'.format(message="[f1 SemEval] ", width=18) + " -> " + str(round(f1_positive_negative_avg, 3)))
        
        print('{message: <{width}}'.format(message="[fitness] ", width=18) + " -> " + str(round(fitnessReturn, 5)) + " ****")
        print('{message: <{width}}'.format(message="[best fitness] ", width=18) + " -> " + str(round(variables.best_fitness, 5)))
        
        print('{message: <{width}}'.format(message="[confusion matrix]", width=18) + " -> " + "[true_positive]: " + str(true_positive) + " " + "[false_positive]: " + str(false_positive) + " " + "[true_negative]: " + str(true_negative) + " " + "[false_negative]: " + str(false_negative) + " " + "[true_neutral]: " + str(true_neutral) + " " + "[false_neutral]: " + str(false_neutral) + "\n")

        if variables.log_all_metrics_each_cicle:
            print('{message: <{width}}'.format(message="[cicles unmodified]", width=24) + " -> " + str(variables.cicles_unchanged))
        
        print('{message: <{width}}'.format(message="[generations unmodified]", width=24) + " -> " + str(variables.generations_unchanged))
        print("[function]: " + str(individual))
        
        if variables.log_times:
            print("[cicle ends after " + str(format(time.time() - start, '.3g')) + " seconds]")     
        
        print("-----------------------------")
        print("\n")   
    #logs

    evaluation_acumulated_time += time.time() - start

    return fitnessReturn


def main():
	random.seed()
	loadModel() # Get the model and the number of dictionaries (individual size)

	getDictionary("train")
	loadTrainTweets()

	# Register the parameters
	toolbox.register("evaluate", evaluateIndividual)
	toolbox.register("mate", tools.cxTwoPoint)
	toolbox.register("mutate", tools.mutFlipBit, indpb=0.2) # see if this is correct - if the indpb is the mutation
	toolbox.register("select", tools.selTournament, tournsize=3)
	toolbox.register("attr_bool", random.uniform, -3, 3) # possible weights
	toolbox.register("individual", tools.initRepeat, creator.IndividualAG, toolbox.attr_bool, INDIVIDUAL_SIZE)
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)
	
	# pset config (see how remove this and put on a unique place/file)
	pset = gp.PrimitiveSetTyped("MAIN", [str], float)
	pset.addPrimitive(operator.add, [float,float], float)
	pset.addPrimitive(operator.sub, [float,float], float)
	pset.addPrimitive(operator.mul, [float,float], float)
	pset.addPrimitive(polaritySumAVGUsingWeights, [str, float, float, float, float, float, float, float, float, float], float)
	pset.addPrimitive(hashtagPolaritySum, [str], float)
	pset.addPrimitive(emoticonsPolaritySum, [str], float)
	pset.addPrimitive(positiveWordsQuantity, [str], float)
	pset.addPrimitive(negativeWordsQuantity, [str], float)
	pset.addPrimitive(hasHashtag, [str], bool)
	pset.addPrimitive(hasEmoticons, [str], bool)
	pset.addPrimitive(hasURLs, [str], bool)
	pset.addPrimitive(hasDates, [str], bool)
	pset.addPrimitive(if_then_else, [bool, float, float], float)
	pset.addPrimitive(removeStopWords, [str], str)
	pset.addPrimitive(removeLinks, [str], str)
	pset.addPrimitive(removeAllPonctuation, [str], str)
	pset.addPrimitive(replaceNegatingWords, [str], str)
	pset.addPrimitive(replaceBoosterWords, [str], str)
	pset.addPrimitive(boostUpperCase, [str], str)
	pset.addPrimitive(neutralRange, [float, float], float)
	pset.addTerminal(True, bool)
	pset.addTerminal(False, bool)
	pset.addTerminal(0.0, float)
	pset.addEphemeralConstant("rand", lambda: random.uniform(0, 2), float)
	pset.addEphemeralConstant("rand2", lambda: random.uniform(0, 2), float)
	pset.renameArguments(ARG0='x')
	# pset config

	toolbox.register("compile", gp.compile, pset=pset)
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
	variables.best_AG_weights_combination = best_ind.fitness.values

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
		    	INDIVIDUAL_SIZE = int(line.split(":")[1].strip()) + 2
		    	print(str(INDIVIDUAL_SIZE))
		    elif not line.startswith("[") and line not in ('\n', '\r\n') and not line.startswith("#"):
		    	MODEL = str(line)
		    	#print(str(line))
		    	


if __name__ == "__main__":                
	#main()

    getDictionary("test")


    #with open('positivehashtagssaved.txt', 'a') as fsave:
	#    with open('dictionaries/sentiment140_unigram.txt', 'r') as f:
	#    	for line in f:
	#    		if line.split("\t")[0].strip().startswith("#") and float(line.split("\t")[1].strip()) > 0:
	#    			fsave.write(line.split("\t")[0][1:].strip() + "\n")




    #from textblob import TextBlob
    #print(str(TextBlob("not a very great calculation").sentiment.subjectivity))
    #print(str(TextBlob(message).sentiment.subjectivity))
    #print(str(TextBlob(message).sentiment.polarity))

    #message = 'Worst away day sat in a home end V Bolton 0-0 87 mins gone Berba scores I Jump up #awkward   #awaydays'
    #x = 'way too amped to sleep right now. it is physically impossible for me to take the SATs tomorrow'
    x = 'Thunderbolt and lightning may have been very very frightening for Queen but it is exciting for me... http://t.co/EmUHF4Aq'

    #x = 'my dear i will pray for you i love you'

    #x = re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", x)
    

    #print(str(hashtagPolaritySum(removeAllPonctuation(x))))

    #for l in x.split():
    #	print(str(getPOSTag(l)))

    #print(str(getPOSTag(x)[0]))
    #print(str(getPOSTag(y)))

    #print(str(polaritySumAVGUsingWeights(replaceBoosterWords(removeAllPonctuation(replaceNegatingWords(removeStopWords(removeAllPonctuation(replaceNegatingWords(removeLinks(boostUpperCase(x)))))))), 0.17594809303743952, hashtagPolaritySum(replaceBoosterWords(replaceNegatingWords(replaceNegatingWords(removeLinks(x))))), emoticonsPolaritySum(x), 0.19612448652451175, if_then_else(hasURLs(x), if_then_else(hasURLs(removeAllPonctuation(x)), emoticonsPolaritySum(x), emoticonsPolaritySum(removeAllPonctuation(replaceNegatingWords(x)))), 0.0), emoticonsPolaritySum(x), emoticonsPolaritySum(boostUpperCase(x)), emoticonsPolaritySum(replaceBoosterWords(x)), sub(neutralRange(1.6709136855853841, negativeWordsQuantity(x)), if_then_else(hasURLs(x), hashtagPolaritySum(replaceNegatingWords(removeStopWords(replaceNegatingWords(x)))), 0.0)))))

    print(str(polaritySumAVGUsingWeights(replaceNegatingWords(removeAllPonctuation(removeAllPonctuation(replaceNegatingWords(removeStopWords(replaceNegatingWords(replaceNegatingWords(x))))))), neutralRange(0.0, 0.0), 0.0, 1.3113316399583013, 0.5580832687368169, 0.0, 0.0, emoticonsPolaritySum(x), 0.0, 0.0)))

	# saveWordsTokenized("test")

	#print(str(getPOSTag("Hi there, i love you so much")))
	#print(str(getPOSTag("This is a cooool #dummysmiley: :-) :-P <3 and some arrows < > -> <--")))

	#getDictionary("train")
	#loadTrainTweets()

	#chunks = createChunks(variables.tweets_semeval, 10)

	#c1, c2, c3, c4 = [x[0] for x in chunks], [x[1] for x in chunks], [x[2] for x in chunks], [x[3] for x in chunks]

	#print(str(c1) + "\n")
	#print(str(c2) + "\n")
	#print(str(c3) + "\n")
	#print(str(c4) + "\n")