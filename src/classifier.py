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
#from functions import *
from gpFunctions import *
from loadFunctions import *
from sendMail import *

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
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

iterate_count, generation_count, best_of_generation = 1, 1, 0


def evalMessages(individual):
    """
        GP evaluation function
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

    variables.neutral_inferior_range, variables.neutral_superior_range = 0, 0

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

    is_positive, is_negative, is_neutral = 0, 0, 0
    true_positive, true_negative, true_neutral, false_positive, false_negative, false_neutral = 0, 0, 0, 0, 0, 0
    precision_positive, precision_negative, precision_neutral, precision_avg                  = 0, 0, 0, 0
    recall_positive, recall_negative, recall_neutral, recall_avg                              = 0, 0, 0, 0
    f1_positive, f1_negative, f1_neutral, f1_avg, f1_positive_negative_avg                    = 0, 0, 0, 0, 0
    accuracy   = 0
    func_value = 0

    # Constraint controls 
    breaked, fitness_decreased, double_decreased = False, False, False

    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)

    MESSAGES_TO_EVALUATE = variables.messages['train']

    # Main loop - iterate through the messages
    for index, item in enumerate(MESSAGES_TO_EVALUATE):
    
        # CONSTRAINTS
        # Constraint 1: massive function - more than massive_functions_max
        if (str(individual).count(variables.massive_function) > variables.massive_functions_max) and variables.massive_functions_constraint:
            print("\n[CONSTRAINT][more than " + str(variables.massive_functions_max) + " massive(s) function(s)][bad individual][fitness zero]\n")
            if variables.log_times:
                print("[cicle ends after " + str(format(time.time() - start, '.3g')) + " seconds]")     
            print("-----------------------------\n")
            breaked = True
            break

        count_neutral_range = 0
        count_neutral_range = str(individual).count("neutralRange")
        
        # Constraint 2: neutralRange - more than one neutralRange
        if count_neutral_range > 1:
            print("\n[CONSTRAINT][more than one neutralRange function][bad individual][fitness zero]\n")
            if variables.log_times:
                print("[cicle ends after " + str(format(time.time() - start, '.3g')) + " seconds]")     
            print("-----------------------------\n")
            breaked = True
            break

        # neutralRange constraints
        if(variables.neutral_range_constraint):      
            # Constraint 4: neutralRange - function does not exist in the model
            if count_neutral_range == 0 and index == 0:
                print("\n[CONSTRAINT][model does not have neutralRange function][fitness decreased in " + str(variables.root_decreased_value * 100) + "%]\n")
                if fitness_decreased:
                    double_decreased = True
                fitness_decreased = True

        # Constraint 5: root function - functions root_functions (if_then_else hardcoded temporarily)
        if (not str(individual).startswith(variables.root_function) and not str(individual).startswith("if_then_else")) and variables.root_constraint and index == 0:
            print("\n[CONSTRAINT][root node is not " + variables.root_function + "][fitness decreased in " + str(variables.root_decreased_value * 100) + "%]\n")
            if fitness_decreased:
                double_decreased = True            
            fitness_decreased = True
        # END CONSTRAINTS

        # Check cicle limit
        if variables.cicles_unchanged >= variables.max_unchanged_cicles:
            breaked = True
            break

        # Log each new cicle
        if index == 0 and variables.log_all_metrics_each_cicle:
            print("\n[New cicle]: " + str(len(MESSAGES_TO_EVALUATE)) + " phrases to evaluate [" + str(variables.POSITIVE_MESSAGES) + " positives, " + str(variables.NEGATIVE_MESSAGES) + " negatives and " + str(variables.NEUTRAL_MESSAGES) + " neutrals]")

        try:
            func_value = float(func(MESSAGES_TO_EVALUATE[index]['message']))

            if MESSAGES_TO_EVALUATE[index]['num_label'] > 0:
                if  func_value > variables.neutral_superior_range:
                    correct_evaluations += 1 
                    is_positive   += 1
                    true_positive += 1
                else:
                    if func_value >= variables.neutral_inferior_range and func_value <= variables.neutral_superior_range:
                        false_neutral += 1
                    elif func_value < variables.neutral_inferior_range:
                        false_negative += 1

            elif MESSAGES_TO_EVALUATE[index]['num_label'] < 0:
                if func_value < variables.neutral_inferior_range:
                    correct_evaluations += 1 
                    is_negative   += 1
                    true_negative += 1
                else:
                    if func_value >= variables.neutral_inferior_range and func_value <= variables.neutral_superior_range:
                        false_neutral += 1
                    elif func_value > variables.neutral_superior_range:
                        false_positive += 1

            elif MESSAGES_TO_EVALUATE[index]['num_label'] == 0:
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

        # LOGS
        if variables.log_all_messages:
            print("[phrase]: "    + MESSAGES_TO_EVALUATE[index]['message'])
            print("[value]: "     + str(MESSAGES_TO_EVALUATE[index]['label']))
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
    if variables.POSITIVE_MESSAGES > 0:
        recall_positive = true_positive / variables.POSITIVE_MESSAGES
        if recall_positive > variables.best_recall_positive:
            variables.best_recall_positive = recall_positive


    if variables.NEGATIVE_MESSAGES > 0:
        recall_negative = true_negative / variables.NEGATIVE_MESSAGES
        if recall_negative > variables.best_recall_negative:
            variables.best_recall_negative = recall_negative

    if variables.NEUTRAL_MESSAGES > 0:
        recall_neutral = true_neutral / variables.NEUTRAL_MESSAGES
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

    # FITNESS
    fitnessReturn = f1_positive_negative_avg
    if fitness_decreased:
        fitnessReturn -= fitnessReturn * variables.root_decreased_value # 80% of the original value
    if double_decreased:
        fitnessReturn -= fitnessReturn * variables.root_decreased_value # Again

    # Saving the best fitness
    if variables.best_fitness < fitnessReturn:
        if variables.best_fitness != 0:
            # save partial best individual (in case we need stop evolution)
            with open('../sandbox/partial_results/' + variables.BEST_INDIVIDUAL, 'w') as f:
                f.write(str(individual))
                f.write("\n\n# Generation -> " + str(generation_count))
                f.write("\n# Neutral Range -> [" + str(variables.neutral_inferior_range) + ", " + str(variables.neutral_superior_range) + "]")
            variables.best_fitness_history.append(variables.best_fitness)
        variables.best_fitness = fitnessReturn
        variables.fitness_positive = is_positive
        variables.fitness_negative = is_negative
        variables.fitness_neutral  = is_neutral
        is_positive, is_negative, is_neutral, variables.cicles_unchanged, variables.generations_unchanged = 0, 0, 0, 0, 0
    else:
        variables.cicles_unchanged += 1
        if new_generation:
            variables.generations_unchanged += 1

    if not new_generation and best_of_generation < fitnessReturn:
        best_of_generation = fitnessReturn

    variables.all_fitness_history.append(fitnessReturn)

    # LOGS   
    if variables.log_parcial_results and not breaked:
        if variables.log_all_metrics_each_cicle:
            print("[correct evaluations] " + str(correct_evaluations))
            print('{message: <{width}}'.format(message="[accuracy] ",   width=18) + " -> " + str(round(accuracy, 3)))
            print('{message: <{width}}'.format(message="[precision] ",  width=18) + " -> " + "[positive]: " + '{message: <{width}}'.format(message=str(round(precision_positive, 3)), width=6) + " " + "[negative]: " + '{message: <{width}}'.format(message=str(round(precision_negative, 3)), width=6) + " " + "[neutral]: " + '{message: <{width}}'.format(message=str(round(precision_neutral, 3)), width=6) + " " + "[avg]: " + '{message: <{width}}'.format(message=str(round(precision_avg, 3)), width=6))
            print('{message: <{width}}'.format(message="[recall] ",     width=18) + " -> " + "[positive]: " + '{message: <{width}}'.format(message=str(round(recall_positive, 3)), width=6) + " " + "[negative]: " + '{message: <{width}}'.format(message=str(round(recall_negative, 3)), width=6) + " " + "[neutral]: " + '{message: <{width}}'.format(message=str(round(recall_neutral, 3)), width=6) + " " + "[avg]: " + '{message: <{width}}'.format(message=str(round(recall_avg, 3)), width=6))
            print('{message: <{width}}'.format(message="[f1] ",         width=18) + " -> " + "[positive]: " + '{message: <{width}}'.format(message=str(round(f1_positive, 3)), width=6) + " " + "[negative]: " + '{message: <{width}}'.format(message=str(round(f1_negative, 3)), width=6) + " " + "[neutral]: " + '{message: <{width}}'.format(message=str(round(f1_neutral, 3)), width=6) + " " + "[avg]: " + '{message: <{width}}'.format(message=str(round(f1_avg, 3)), width=6))
            print('{message: <{width}}'.format(message="[f1 SemEval] ", width=18) + " -> " + str(round(f1_positive_negative_avg, 3)))
        
        print('{message: <{width}}'.format(message="[fitness] "     ,    width=18) + " -> " + str(round(fitnessReturn, 5)) + " ****")
        print('{message: <{width}}'.format(message="[best fitness] ",    width=18) + " -> " + str(round(variables.best_fitness, 5)))
        print('{message: <{width}}'.format(message="[confusion matrix]", width=18) + " -> " + "[true_positive]: " + str(true_positive) + " " + "[false_positive]: " + str(false_positive) + " " + "[true_negative]: " + str(true_negative) + " " + "[false_negative]: " + str(false_negative) + " " + "[true_neutral]: " + str(true_neutral) + " " + "[false_neutral]: " + str(false_neutral) + "\n")

        if variables.log_all_metrics_each_cicle:
            print('{message: <{width}}'.format(message="[cicles unmodified]", width=24) + " -> " + str(variables.cicles_unchanged))
        
        print('{message: <{width}}'.format(message="[generations unmodified]", width=24) + " -> " + str(variables.generations_unchanged))
        print("[function]: " + str(individual))
        
        if variables.log_times:
            print("[cicle ends after " + str(format(time.time() - start, '.3g')) + " seconds]")     
        
        print("-----------------------------\n")
    # LOGS

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
    accuracy = 0
    is_positive, is_negative, is_neutral = 0, 0, 0
    true_positive, true_negative, true_neutral, false_positive, false_negative, false_neutral = 0, 0, 0, 0, 0, 0
    precision_positive, precision_negative, precision_neutral, precision_avg                  = 0, 0, 0, 0
    recall_positive, recall_negative, recall_neutral, recall_avg                              = 0, 0, 0, 0
    f1_positive, f1_negative, f1_neutral, f1_avg, f1_positive_negative_avg                    = 0, 0, 0, 0, 0
    func_value = 0

    # Constraint controls 
    breaked, fitness_decreased, double_decreased = False, False, False

    func = toolbox.compile(expr=individual)

    TWEETS_TO_EVALUATE, TWEETS_SCORE_TO_EVALUATE = variables.tweets_semeval, variables.tweets_semeval_score

    indexes = list(range(len(TWEETS_TO_EVALUATE)))
    #chunks = createIndexChunks(indexes, 10)
    chunks = createRandomIndexChunks(indexes, 10)

    fitness_list = []

    fold_index = 0

    for fold in chunks:
        correct_evaluations, fitnessReturn, accuracy = 0, 0, 0
        is_positive, is_negative, is_neutral         = 0, 0, 0
        true_positive, true_negative, true_neutral, false_positive, false_negative, false_neutral = 0, 0, 0, 0, 0, 0
        precision_positive, precision_negative, precision_neutral, precision_avg                  = 0, 0, 0, 0
        recall_positive, recall_negative, recall_neutral, recall_avg                              = 0, 0, 0, 0
        f1_positive, f1_negative, f1_neutral, f1_avg, f1_positive_negative_avg                    = 0, 0, 0, 0, 0
        first = True
        # Constraint controls 
        breaked, fitness_decreased, double_decreased = False, False, False
        
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

        count_neutral_range = 0
        count_neutral_range = str(individual).count("neutralRange")
        
        # Constraint 2: neutralRange - more than one neutralRange
        if count_neutral_range > 1:
            print("\n[CONSTRAINT][more than one neutralRange function][bad individual][fitness zero]\n")
            if variables.log_times:
                print("[cicle ends after " + str(format(time.time() - start, '.3g')) + " seconds]")     
            print("-----------------------------")
            print("\n") 
            breaked = True
            break

        # neutralRange constraints
        if(variables.neutral_range_constraint):      
            # Constraint 4: neutralRange - function does not exist in the model
            if count_neutral_range == 0:
                if fold_index == 0: #log only on first index
                    print("\n[CONSTRAINT][model does not have neutralRange function][fitness decreased in " + str(variables.root_decreased_value * 100) + "%]\n")
                if fitness_decreased:
                    double_decreased = True
                fitness_decreased = True

        # Constraint 5: root function - functions root_functions (if_then_else hardcoded temporarily)
        #if (not str(individual).startswith(variables.root_function) and not str(individual).startswith("if_then_else")) and variables.root_constraint:# and fold_index == 0:
        if (not str(individual).startswith(variables.root_function)) and variables.root_constraint:# and fold_index == 0:
            if fold_index == 0: #log only on first index
                print("[CONSTRAINT][root node is not " + variables.root_function + "][fitness decreased in " + str(variables.root_decreased_value * 100) + "%]\n")
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
                if variables.log_all_metrics_each_cicle:
                    print("\n[New cicle]: " + str(len(TWEETS_TO_EVALUATE)) + " phrases to evaluate [" + str(variables.POSITIVE_MESSAGES) + " positives, " + str(variables.NEGATIVE_MESSAGES) + " negatives and " + str(variables.NEUTRAL_MESSAGES) + " neutrals]")
                    first = False

            try:
                func_value = float(func(TWEETS_TO_EVALUATE[index]))
                
                if float(TWEETS_SCORE_TO_EVALUATE[index]) > 0:
                    is_positive   += 1
                    if  func_value > variables.neutral_superior_range:
                        correct_evaluations += 1 
                        true_positive += 1
                    else:
                        if func_value >= variables.neutral_inferior_range and func_value <= variables.neutral_superior_range:
                            false_neutral += 1
                        elif func_value < variables.neutral_inferior_range:
                            false_negative += 1

                elif float(TWEETS_SCORE_TO_EVALUATE[index]) < 0:
                    is_negative   += 1
                    if func_value < variables.neutral_inferior_range:
                        correct_evaluations += 1 
                        true_negative += 1
                    else:
                        if func_value >= variables.neutral_inferior_range and func_value <= variables.neutral_superior_range:
                            false_neutral += 1
                        elif func_value > variables.neutral_superior_range:
                            false_positive += 1

                elif float(TWEETS_SCORE_TO_EVALUATE[index]) == 0:
                    is_neutral   += 1
                    if func_value >= variables.neutral_inferior_range and func_value <= variables.neutral_superior_range:
                        correct_evaluations += 1 
                        true_neutral += 1
                    else:
                        if func_value < variables.neutral_inferior_range:
                            false_negative += 1
                        elif func_value > variables.neutral_superior_range:
                            false_positive += 1

            except Exception as e: 
                print(e)
                continue

            # LOGS
            if variables.log_all_messages:
                print("[phrase]: " + TWEETS_TO_EVALUATE[index])
                print("[value]: " + str(TWEETS_SCORE_TO_EVALUATE[index]))
                print("[calculated]:" + func_value)

        if breaked:
            break

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
        if is_positive > 0:
            recall_positive = true_positive / is_positive
            if recall_positive > variables.best_recall_positive:
                variables.best_recall_positive = recall_positive


        if is_negative > 0:
            recall_negative = true_negative / is_negative
            if recall_negative > variables.best_recall_negative:
                variables.best_recall_negative = recall_negative

        if is_neutral > 0:
            recall_neutral = true_neutral / is_neutral
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

        
        fitnessReturn = f1_positive_negative_avg
        orig = 0
        if fitness_decreased:
            orig = fitnessReturn
            fitnessReturn -= fitnessReturn * variables.root_decreased_value # 80% of the original value
            print("  [INDIVIDUAL]  " + str(individual))
            print("     [POS: " + str(is_positive) + "] [NEG: " + str(is_negative) + "] [NEU: " + str(is_neutral) +"]")
            print("       [FITNESS DECREASED] [ORIGINAL: " + str(orig) + "] [DECREASED: " + str(fitnessReturn) + "]")
        orig = 0
        if double_decreased:
            orig = fitnessReturn
            fitnessReturn -= fitnessReturn * variables.root_decreased_value # Again
            print("         [FITNESS DOUBLE DECREASED] [ORIGINAL: " + str(orig) + "] [DECREASED: " + str(fitnessReturn) + "]")

        fitness_list.append(fitnessReturn)
        fold_index += 1

    fitnessReturn = sum(fitness_list)/len(chunks)
    
    print("[fitness list] "     + str(fitness_list))
    print("[fitness sum] "      + str(sum(fitness_list)))
    print("[number of chunks] " + str(len(chunks)))
    print("[avg fitness] "      + str(fitnessReturn) + "\n")

    if variables.best_fitness < fitnessReturn:
        if variables.best_fitness != 0:
            with open('../sandbox/partial_results/' + variables.BEST_INDIVIDUAL, 'w') as f:
                f.write(str(individual))
                f.write("\n\n# Generation -> " + str(generation_count))
                f.write("\n# Neutral Range -> [" + str(variables.neutral_inferior_range) + ", " + str(variables.neutral_superior_range) + "]")
            variables.best_fitness_history.append(variables.best_fitness)
        variables.best_fitness = fitnessReturn
        variables.best_fitness_history_dict[generation_count] = fitnessReturn
        variables.fitness_positive = is_positive
        variables.fitness_negative = is_negative
        variables.fitness_neutral  = is_neutral
        is_positive, is_negative, is_neutral, variables.cicles_unchanged, variables.generations_unchanged = 0, 0, 0, 0, 0
    else:
        variables.cicles_unchanged += 1
        if new_generation:
            variables.generations_unchanged += 1

    if not new_generation and best_of_generation < fitnessReturn:
        best_of_generation = fitnessReturn

    variables.all_fitness_history.append(fitnessReturn)


    # LOGS
    if variables.log_parcial_results and not breaked:# and not variables.calling_by_ag_file: 
        if variables.log_all_metrics_each_cicle:
            print("[correct evaluations] " + str(correct_evaluations))
            print('{message: <{width}}'.format(message="[accuracy] ", width=18)   + " -> " + str(round(accuracy, 3)))
            print('{message: <{width}}'.format(message="[precision] ", width=18)  + " -> " + "[positive]: " + '{message: <{width}}'.format(message=str(round(precision_positive, 3)), width=6) + " " + "[negative]: " + '{message: <{width}}'.format(message=str(round(precision_negative, 3)), width=6) + " " + "[neutral]: " + '{message: <{width}}'.format(message=str(round(precision_neutral, 3)), width=6) + " " + "[avg]: " + '{message: <{width}}'.format(message=str(round(precision_avg, 3)), width=6))
            print('{message: <{width}}'.format(message="[recall] ", width=18)     + " -> " + "[positive]: " + '{message: <{width}}'.format(message=str(round(recall_positive, 3)), width=6) + " " + "[negative]: " + '{message: <{width}}'.format(message=str(round(recall_negative, 3)), width=6) + " " + "[neutral]: " + '{message: <{width}}'.format(message=str(round(recall_neutral, 3)), width=6) + " " + "[avg]: " + '{message: <{width}}'.format(message=str(round(recall_avg, 3)), width=6))
            print('{message: <{width}}'.format(message="[f1] ", width=18)         + " -> " + "[positive]: " + '{message: <{width}}'.format(message=str(round(f1_positive, 3)), width=6) + " " + "[negative]: " + '{message: <{width}}'.format(message=str(round(f1_negative, 3)), width=6) + " " + "[neutral]: " + '{message: <{width}}'.format(message=str(round(f1_neutral, 3)), width=6) + " " + "[avg]: " + '{message: <{width}}'.format(message=str(round(f1_avg, 3)), width=6))
            print('{message: <{width}}'.format(message="[f1 SemEval] ", width=18) + " -> " + str(round(f1_positive_negative_avg, 3)))
        
        print('{message: <{width}}'.format(message="[fitness] ", width=18) + " -> "      + str(round(fitnessReturn, 5)) + " ****")
        print('{message: <{width}}'.format(message="[best fitness] ", width=18) + " -> " + str(round(variables.best_fitness, 5)))
        
        print('{message: <{width}}'.format(message="[confusion matrix]", width=18) + " -> " + "[true_positive]: " + str(true_positive) + " " + "[false_positive]: " + str(false_positive) + " " + "[true_negative]: " + str(true_negative) + " " + "[false_negative]: " + str(false_negative) + " " + "[true_neutral]: " + str(true_neutral) + " " + "[false_neutral]: " + str(false_neutral) + "\n")

        if variables.log_all_metrics_each_cicle:
            print('{message: <{width}}'.format(message="[cicles unmodified]", width=24)  + " -> " + str(variables.cicles_unchanged))
        
        print('{message: <{width}}'.format(message="[generations unmodified]", width=24) + " -> " + str(variables.generations_unchanged))
        print("[function]: " + str(individual))
        
        if variables.log_times:
            print("[cicle ends after " + str(format(time.time() - start, '.3g')) + " seconds]")     
        
        print("-----------------------------\n")
    # LOGS

    evaluation_acumulated_time += time.time() - start
    
    return fitnessReturn,


if variables.train_using_folds:
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


def myVarAnd(population, toolbox, cxpb, mutpb):
    """
        myVarAnd - based on DEAP library (algorithms.py).
        https://github.com/DEAP/deap/blob/master/deap/algorithms.py
    """    
    offspring = [toolbox.clone(ind) for ind in population]

    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                          offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values
    
    for i in range(len(offspring)):
        if random.random() < variables.MUTATE_EPHEMERAL:
            offspring[i], = toolbox.mutateEphemeral(offspring[i], "all")
            del offspring[i].fitness.values

    return offspring


def myEaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):

    """
        myEaSimple - based on DEAP library (algorithms.py).
        https://github.com/DEAP/deap/blob/master/deap/algorithms.py
    """
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
    """
        Main function - call the myEaSimple and print some result logs
    """
    start = time.time()
    global evaluation_acumulated_time
    random.seed()

    pop = toolbox.population(n=variables.POPULATION)
    hof = tools.HallOfFame(variables.HOF)

    pop, log = myEaSimple(pop, toolbox, variables.CROSSOVER, variables.MUTATION, variables.GENERATIONS, stats=False,
                                   halloffame=hof, verbose=False)


    # LOGS
    print("\n## Results ##\n")
    print("[total tweets]: " + str(variables.POSITIVE_MESSAGES + variables.NEGATIVE_MESSAGES + variables.NEUTRAL_MESSAGES) + " [" + str(variables.POSITIVE_MESSAGES) + " positives, " + str(variables.NEGATIVE_MESSAGES) + " negatives and " + str(variables.NEUTRAL_MESSAGES) + " neutrals]\n")
    print("[best fitness (F1 avg (+/-)]: " + str(variables.best_fitness) + " [" + str(variables.fitness_positive + variables.fitness_negative + variables.fitness_neutral) + " correct evaluations] ["+ str(variables.fitness_positive) + " positives, " + str(variables.fitness_negative) + " negatives and " + str(variables.fitness_neutral) + " neutrals]\n")
    print("[function]: " + str(hof[0]) + "\n")
    print("[best accuracy]: " + str(variables.best_accuracy) + "\n")
    print("[best precision positive]: " + str(variables.best_precision_positive))
    print("[best precision negative]: " + str(variables.best_precision_negative))
    print("[best precision neutral]: "  + str(variables.best_precision_neutral))    
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
    print("[best f1 avg function]: " + variables.best_f1_avg_function)
    print("[best fitness history]: " + str(variables.best_fitness_history_dict) + "\n\n")
    # LOGS 

    # PLOT 
    X = range(len(variables.best_fitness_per_generation_history))
    Y = variables.best_fitness_per_generation_history
    plt.plot(X,Y)
    plt.savefig(variables.TRAIN_RESULTS_IMG + "-all.png", bbox_inches='tight') # all models in one image
    plt.savefig(variables.TRAIN_RESULTS_IMG + str(datetime.now())[14:16] + str(datetime.now())[17:19] + ".png", bbox_inches='tight') # one image per model
    # PLOT

    end = time.time()
    print("[evaluation function consumed " + str(format(evaluation_acumulated_time, '.3g')) + " seconds]")
    print("[main function ended][" + str(format(end - start, '.3g')) + " seconds]\n")
    
    variables.model_results.append(hof[0])
    
    if not variables.save_only_best_individual:
        variables.model_results_others.append(hof[1])
        variables.model_results_others.append(hof[2])
        variables.model_results_others.append(hof[3])

    return pop, log, hof


if __name__ == "__main__":
    print("[starting classifier module]")

    loadDictionaries()
    loadTrainMessages(100)

    parameters = str(variables.CROSSOVER) + " crossover, " + str(variables.MUTATION) + " mutation, " + str(variables.POPULATION) + " population, " + str(variables.GENERATIONS) + " generation"

    with open(variables.TRAIN_RESULTS, 'a') as f:
        f.write("[PARAMS]: " + parameters + "\n" + "[DICTIONARIES]: " + str(variables.dic_loaded_total) + "\n\n")
    
    
    # Main loop - will call the main function TOTAL_MODEL times
    for i in range(variables.TOTAL_MODELS):
        main()
        
        with open(variables.TRAIN_RESULTS, 'a') as f:
            f.write(str(variables.model_results[len(variables.model_results) - 1]) + "\n")
            
            if not variables.save_only_best_individual:
                for m in variables.model_results_others:
                    f.write(str(m) + "\n")
                f.write("\n")

        variables.model_results_others = []

        mail_content = "Parameters: " + parameters + "\n\n" + str(variables.model_results[len(variables.model_results) - 1]) + "\n"
        mail_content += "\n\nTotal tweets: " + str(variables.POSITIVE_MESSAGES + variables.NEGATIVE_MESSAGES + variables.NEUTRAL_MESSAGES) + " [" + str(variables.POSITIVE_MESSAGES) + " positives, " + str(variables.NEGATIVE_MESSAGES) + " negatives and " + str(variables.NEUTRAL_MESSAGES) + " neutrals]\n"
        mail_content += "Fitness (F1 pos and neg): " + str(variables.best_fitness) + " [" + str(variables.fitness_positive + variables.fitness_negative + variables.fitness_neutral) + " correct evaluations] ["+ str(variables.fitness_positive) + " positives, " + str(variables.fitness_negative) + " negatives and " + str(variables.fitness_neutral) + " neutrals]\n"
        mail_content += "\nFitness evolution: " + str(variables.best_fitness_history_dict) + "\n"

        try:
            send_mail(i+1, variables.TOTAL_MODELS, variables.POPULATION, variables.GENERATIONS, mail_content)
        except Exception as e:
            print("[Warning] No internet connection, the email can't be send!")
            print(e)


        # Reset the variables
        iterate_count, generation_count = 1
        best_of_generation              = 0
        variables.fitness_positive, variables.fitness_negative, variables.fitness_neutral = 0, 0, 0
        variables.cicles_unchanged, variables.generations_unchanged = 0, 0
        variables.generations_unchanged_reached_msg = False
        variables.best_fitness = 0
        variables.best_fitness_history, variables.best_fitness_per_generation_history, variables.all_fitness_history = [], [], []
        variables.best_accuracy = 0
        variables.best_precision_positive, variables.best_precision_negative, variables.best_precision_neutral, variables.best_precision_avg              = 0, 0, 0, 0
        variables.best_recall_positive, variables.best_recall_negative,variables.best_recall_neutral, variables.best_recall_avg                           = 0, 0, 0, 0
        variables.best_f1_positive, variables.best_f1_negative, variables.best_f1_neutral, variables.best_f1_avg, variables.best_f1_positive_negative_avg = 0, 0, 0, 0, 0
        variables.best_precision_avg_function, variables.best_recall_avg_function, variables.best_f1_avg_function = "", "", ""
        variables.best_fitness_history_dict = {}

end = time.time()
print("Script ends after " + str(format(end - start, '.3g')) + " seconds")