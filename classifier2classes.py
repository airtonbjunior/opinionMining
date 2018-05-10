# Airton Bordin Junior
# airtonbjunior@gmail.com
# Federal University of Goias (UFG)
# Computer Science Master's Degree
#
# Genetic Programming lib: DEAP
# References: https://github.com/DEAP/deap/blob/08986fc3848144903048c722564b7b1d92db33a1/examples/gp/symbreg.py
#             https://github.com/DEAP/deap/blob/08986fc3848144903048c722564b7b1d92db33a1/examples/gp/spambase.py

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

evaluation_acumulated_time = 0

# log time
start = time.time()

# parameters: tweet, neutral_inferior_range, neutral_superior_range
#pset = gp.PrimitiveSetTyped("MAIN", [str, float, float], float)
pset = gp.PrimitiveSetTyped("MAIN", [str], float)
pset.addPrimitive(operator.add, [float,float], float)
#pset.addPrimitive(operator.sub, [float,float], float)
#pset.addPrimitive(operator.mul, [float,float], float)
#pset.addPrimitive(protectedDiv, [float,float], float)
#pset.addPrimitive(math.exp, [float], float)
#pset.addPrimitive(math.cos, [float], float)
#pset.addPrimitive(math.sin, [float], float)
#pset.addPrimitive(protectedSqrt, [float], float)
#pset.addPrimitive(protectedLog, [float], float)
pset.addPrimitive(invertSignal, [float], float)
#pset.addPrimitive(math.pow, [float, float], float)

pset.addPrimitive(positiveHashtags, [str], float)
pset.addPrimitive(negativeHashtags, [str], float)
pset.addPrimitive(positiveEmoticons, [str], float)
pset.addPrimitive(negativeEmoticons, [str], float)

# Temporary removed
#pset.addPrimitive(polaritySum2, [str], float)
#pset.addPrimitive(polaritySumAVG, [str], float)
# Temporary removed

#pset.addPrimitive(passInt, [int], int)
pset.addPrimitive(polaritySumAVGUsingWeights, [str, float, float, float, float, float, float, float], float)
pset.addPrimitive(hashtagPolaritySum, [str], float)
pset.addPrimitive(emoticonsPolaritySum, [str], float)
pset.addPrimitive(positiveWordsQuantity, [str], float)
pset.addPrimitive(negativeWordsQuantity, [str], float)


pset.addPrimitive(hasHashtag, [str], bool)
pset.addPrimitive(hasEmoticons, [str], bool)
pset.addPrimitive(hasURLs, [str], bool)
pset.addPrimitive(hasDates, [str], bool)

pset.addPrimitive(if_then_else, [bool, float, float], float)

pset.addPrimitive(stemmingText, [str], str)
pset.addPrimitive(removeStopWords, [str], str)
pset.addPrimitive(removeLinks, [str], str)
#pset.addPrimitive(removeEllipsis, [str], str)
pset.addPrimitive(removeAllPonctuation, [str], str)
pset.addPrimitive(replaceNegatingWords, [str], str)
pset.addPrimitive(replaceBoosterWords, [str], str)
pset.addPrimitive(boostUpperCase, [str], str)

#pset.addPrimitive(neutralRange, [float, float], float)

pset.addTerminal(True, bool)
pset.addTerminal(False, bool)

#pset.addTerminal(-2.0, float)
#pset.addTerminal(-1.5, float)
#pset.addTerminal(-1.0, float)
#pset.addTerminal(-0.5, float)
pset.addTerminal(0.0, float)
#pset.addTerminal(0.5, float)
#pset.addTerminal(1.0, float)
#pset.addTerminal(1.5, float)
#pset.addTerminal(2.0, float)

pset.addEphemeralConstant("rand", lambda: random.uniform(0, 2), float)
pset.addEphemeralConstant("rand2", lambda: random.uniform(0, 2), float)
#pset.addEphemeralConstant("randInt", lambda: random.randint(0, 3), int)

pset.renameArguments(ARG0='x')
#pset.renameArguments(ARG0='y')
#pset.renameArguments(ARG0='z')

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

iterate_count = 1
generation_count = 1
best_of_generation = 0

# evaluation function 
def evalSymbRegTweetsFromSemeval(individual):
    start = time.time()
    global iterate_count
    global generation_count
    global best_of_generation
    new_generation = False

    # test
    variables.neutral_inferior_range = 0
    variables.neutral_superior_range = 0
    # test

    # Check max unchanged generations
    if variables.generations_unchanged >= variables.max_unchanged_generations:
        if(variables.generations_unchanged_reached_msg == False):
            print("[Max unchanged generations (" + str(variables.max_unchanged_generations) + ") reached on generation " + str(generation_count) + "]")
        variables.generations_unchanged_reached_msg = True
        return 0,

    # Log the number of each individual
    if iterate_count <= variables.POPULATION:
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
    TWEETS_TO_EVALUATE       = variables.tweets_sts
    TWEETS_SCORE_TO_EVALUATE = variables.tweets_sts_score

    # Main loop
    #for index, item in enumerate(variables.tweets_semeval):
    for index, item in enumerate(TWEETS_TO_EVALUATE):
        
        # Constraints
        # Constraint 1: massive function - more than massive_functions_max
        if (str(individual).count(variables.massive_function) > variables.massive_functions_max) and variables.massive_functions_constraint:
            print("\n[CONSTRAINT][more than " + str(variables.massive_functions_max) + " massive(s) function(s)][bad individual][fitness zero]\n")
            if variables.log_times:
                print("[cicle ends after " + str(format(time.time() - start, '.3g')) + " seconds]")     
            print("-----------------------------")
            print("\n") 
            breaked = True
            break

        # Constraint 5: root function - functions root_functions (if_then_else hardcoded temporarily)
        if (not str(individual).startswith(variables.root_function) and not str(individual).startswith("if_then_else")) and variables.root_constraint and index == 0:
            print("\n[CONSTRAINT][root node is not " + variables.root_function + "][fitness decreased in " + str(variables.root_decreased_value * 100) + "%]\n")
            if fitness_decreased:
                double_decreased = True            
            fitness_decreased = True
        # End Constraints

        # Check cicle limit
        if variables.cicles_unchanged >= variables.max_unchanged_cicles:
            breaked = True
            break

        # Log each new cicle
        if index == 0:
            if variables.log_all_metrics_each_cicle:
                print("\n[New cicle]: " + str(len(TWEETS_TO_EVALUATE)) + " phrases to evaluate [" + str(variables.positive_tweets) + " positives, " + str(variables.negative_tweets) + " negatives")

        try:
            func_value = float(func(TWEETS_TO_EVALUATE[index]))

            if float(TWEETS_SCORE_TO_EVALUATE[index]) > 0:
                if func_value > 0:
                    correct_evaluations += 1 
                    is_positive   += 1
                    true_positive += 1
                else:
                    false_negative += 1

            elif float(TWEETS_SCORE_TO_EVALUATE[index]) < 0:
                if func_value < 0:
                    correct_evaluations += 1 
                    is_negative   += 1
                    true_negative += 1
                else:
                    false_positive += 1

        except Exception as e: 
            print(e)
            continue

        #logs
        if variables.log_all_messages:
            print("[phrase]: " + TWEETS_TO_EVALUATE[index])
            print("[value]: " + str(TWEETS_SCORE_TO_EVALUATE[index]))
            print("[calculated]:" + func_value)


    if true_positive + false_positive + true_negative + false_negative > 0:
        accuracy = (true_positive + true_negative) / (true_positive + false_positive + true_negative + false_negative)

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
    # End F1

    # Precision, Recall and f1 means
    precision_avg = (precision_positive + precision_negative) / 2
    if precision_avg > variables.best_precision_avg:
        variables.best_precision_avg = precision_avg
        variables.best_precision_avg_function = str(individual)

    recall_avg = (recall_positive + recall_negative) / 2
    if recall_avg > variables.best_recall_avg:
        variables.best_recall_avg = recall_avg
        variables.best_recall_avg_function = str(individual)

    f1_avg = (f1_positive + f1_negative) / 2
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


    if variables.best_fitness < fitnessReturn:
        if variables.best_fitness != 0:
            # save partial best individual (in case we need stop evolution)
            with open(variables.BEST_INDIVIDUAL_2CLASSES, 'w') as f:
                f.write(str(individual))
                f.write("\n\n# Generation -> " + str(generation_count))
                #f.write("\n# Neutral Range -> [" + str(variables.neutral_inferior_range) + ", " + str(variables.neutral_superior_range) + "]")
            variables.best_fitness_history.append(variables.best_fitness)
        variables.best_fitness = fitnessReturn
        variables.fitness_positive = is_positive
        variables.fitness_negative = is_negative
        #variables.fitness_neutral  = is_neutral
        is_positive = 0
        is_negative = 0
        #is_neutral  = 0
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
    if variables.log_parcial_results and not breaked: 
        if variables.log_all_metrics_each_cicle:
            print("[correct evaluations] " + str(correct_evaluations))
            print('{message: <{width}}'.format(message="[accuracy] ", width=18) + " -> " + str(round(accuracy, 3)))
            print('{message: <{width}}'.format(message="[precision] ", width=18) + " -> " + "[positive]: " + '{message: <{width}}'.format(message=str(round(precision_positive, 3)), width=6) + " " + "[negative]: " + '{message: <{width}}'.format(message=str(round(precision_negative, 3)), width=6) + "[avg]: " + '{message: <{width}}'.format(message=str(round(precision_avg, 3)), width=6))
            print('{message: <{width}}'.format(message="[recall] ", width=18) + " -> " + "[positive]: " + '{message: <{width}}'.format(message=str(round(recall_positive, 3)), width=6) + " " + "[negative]: " + '{message: <{width}}'.format(message=str(round(recall_negative, 3)), width=6) + "[avg]: " + '{message: <{width}}'.format(message=str(round(recall_avg, 3)), width=6))
            print('{message: <{width}}'.format(message="[f1] ", width=18) + " -> " + "[positive]: " + '{message: <{width}}'.format(message=str(round(f1_positive, 3)), width=6) + " " + "[negative]: " + '{message: <{width}}'.format(message=str(round(f1_negative, 3)), width=6) + "[avg]: " + '{message: <{width}}'.format(message=str(round(f1_avg, 3)), width=6))
            print('{message: <{width}}'.format(message="[f1 SemEval] ", width=18) + " -> " + str(round(f1_positive_negative_avg, 3)))
        
        print('{message: <{width}}'.format(message="[fitness] ", width=18) + " -> " + str(round(fitnessReturn, 5)) + " ****")
        print('{message: <{width}}'.format(message="[best fitness] ", width=18) + " -> " + str(round(variables.best_fitness, 5)))
        
        print('{message: <{width}}'.format(message="[confusion matrix]", width=18) + " -> " + "[true_positive]: " + str(true_positive) + " " + "[false_positive]: " + str(false_positive) + " " + "[true_negative]: " + str(true_negative) + " " + "[false_negative]: " + str(false_negative) + "\n")

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

    return fitnessReturn,



toolbox.register("evaluate", evalSymbRegTweetsFromSemeval)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genHalfAndHalf, min_=0, max_=6)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.register("mutateEphemeral", gp.mutEphemeral)


toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))


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
    start = time.time()
    global evaluation_acumulated_time
    random.seed()

    pop = toolbox.population(n=variables.POPULATION)
    hof = tools.HallOfFame(4)

    # Parameters
        # population (list of individuals)
        # toolbox (that contains the evolution operators)
        # Mating probability (two individuals)
        # Mutation probability
        # Number of generations
        # Statistics objetc (updated inplace)
        # HallOfFame object that contain the best individuals
        # Whether or not to log the statistics
    pop, log = myEaSimple(pop, toolbox, variables.CROSSOVER, variables.MUTATION, variables.GENERATIONS, stats=False,
                                   halloffame=hof, verbose=False)


    #logs
    print("\n")
    print("## Results ##\n")
    print("[total tweets]: " + str(variables.positive_tweets + variables.negative_tweets) + " [" + str(variables.positive_tweets) + " positives, " + str(variables.negative_tweets) + " negatives\n")
    print("[best fitness (F1 avg (+/-)]: " + str(variables.best_fitness) + " [" + str(variables.fitness_positive + variables.fitness_negative) + " correct evaluations] ["+ str(variables.fitness_positive) + " positives, " + str(variables.fitness_negative) + " negatives\n")
    print("[function]: " + str(hof[0]) + "\n")
    print("[best accuracy]: " + str(variables.best_accuracy) + "\n")
    print("[best precision positive]: " + str(variables.best_precision_positive))
    print("[best precision negative]: " + str(variables.best_precision_negative))
    #print("[best precision neutral]: "  + str(variables.best_precision_neutral))    
    print("[best precision avg]: " + str(variables.best_precision_avg))
    print("[best precision avg function]: " + variables.best_precision_avg_function + "\n")    
    print("[best recall positive]: " + str(variables.best_recall_positive))    
    print("[best recall negative]: " + str(variables.best_recall_negative))
    print("[best recall avg]: " + str(variables.best_recall_avg))
    print("[best recall avg function]: " + variables.best_recall_avg_function + "\n")
    print("[best f1 positive]: " + str(variables.best_f1_positive))    
    print("[best f1 negative]: " + str(variables.best_f1_negative))
    print("[best f1 avg]: " + str(variables.best_f1_avg))
    print("[best f1 avg (+/-)]: " + str(variables.best_f1_positive_negative_avg))
    print("[best f1 avg function]: " + variables.best_f1_avg_function + "\n")       
    #print(json.dumps(variables.all_fitness_history))
    print("\n")
    #print(set(variables.all_fitness_history))
    #logs 

    end = time.time()
    print("[evaluation function consumed " + str(format(evaluation_acumulated_time, '.3g')) + " seconds]")
    print("[main function ended][" + str(format(end - start, '.3g')) + " seconds]\n")
    
    variables.model_results.append(hof[0])

    return pop, log, hof


if __name__ == "__main__":
    #if(len(sys.argv) <= 1):
    #    print("No parameter passed - Using the SemEval 2014 benchmark")
    #else:
    #    print(len(sys.argv))
    print("[starting classifier module]")

    getDictionary("train")
    #loadTrainTweets()
    loadTrainTweets_STS()


    parameters = str(variables.CROSSOVER) + " crossover, " + str(variables.MUTATION) + " mutation, " + str(variables.POPULATION) + " population, " + str(variables.GENERATIONS) + " generation\n\n"

    with open(variables.TRAIN_RESULTS_2CLASSES, 'a') as f:
        f.write("[PARAMS]: " + parameters)
    
    for i in range(variables.TOTAL_MODELS):
        main()
        
        with open(variables.TRAIN_RESULTS_2CLASSES, 'a') as f:
            f.write(str(variables.model_results[len(variables.model_results) - 1]) + "\n")

        mail_content = "Parameters: " + parameters + "\n\n" + str(variables.model_results[len(variables.model_results) - 1]) + "\n"
        mail_content += "\n\nTotal tweets: " + str(variables.positive_tweets + variables.negative_tweets) + " [" + str(variables.positive_tweets) + " positives, " + str(variables.negative_tweets) + " negatives and\n"
        mail_content += "Fitness (F1 pos and neg): " + str(variables.best_fitness) + " [" + str(variables.fitness_positive + variables.fitness_negative) + " correct evaluations] ["+ str(variables.fitness_positive) + " positives, " + str(variables.fitness_negative) + " negatives and\n"

        try:
            send_mail(i+1, variables.TOTAL_MODELS, variables.POPULATION, variables.GENERATIONS, mail_content)
        except Exception as e:
            print("[Warning] No internet connection, the email can't be send!")
            print(e)


        # Restart the variables
        iterate_count = 1
        generation_count = 1
        best_of_generation = 0

        variables.fitness_positive = 0
        variables.fitness_negative = 0
        variables.fitness_neutral  = 0

        variables.cicles_unchanged = 0
        variables.generations_unchanged = 0
        variables.generations_unchanged_reached_msg = False
        
        variables.best_fitness = 0
        variables.best_fitness_history  = []
        variables.best_fitness_per_generation_history = []
        variables.all_fitness_history   = []

        variables.best_accuracy = 0

        variables.best_precision_positive = 0
        variables.best_precision_negative = 0
        variables.best_precision_neutral  = 0
        variables.best_precision_avg      = 0

        variables.best_recall_positive = 0
        variables.best_recall_negative = 0
        variables.best_recall_neutral  = 0
        variables.best_recall_avg      = 0

        variables.best_f1_positive = 0
        variables.best_f1_negative = 0
        variables.best_f1_neutral  = 0
        variables.best_f1_avg      = 0
        variables.best_f1_positive_negative_avg = 0

        variables.best_precision_avg_function = ""
        variables.best_recall_avg_function    = ""
        variables.best_f1_avg_function        = ""

    #print(len(variables.all_fitness_history))
    #print(variables.all_fitness_history)
    
    # remove the 0's values to plot
    #plt.plot(list(filter(lambda a: a != 0, variables.all_fitness_history)))    

    #plt.plot(variables.best_fitness_per_generation_history)
    #plt.ylabel('f1')
    #plt.show()

end = time.time()
print("Script ends after " + str(format(end - start, '.3g')) + " seconds")