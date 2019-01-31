# Airton Bordin Junior
# airtonbjunior@gmail.com
# Federal University of Goias (UFG)
# Computer Science Master's Degree
# 
# Reference: https://stackoverflow.com/a/34139884

import yagmail
import platform
import variables

def send_mail(ith, total, pop, gen, parameters):
	FROM = 'sclassifier@gmail.com'
	TO = 'sclassifier@gmail.com'
	if(ith < total):
		SUBJECT = 'Model ' + str(ith) + ' of ' + str(total) + ' created! [' + str(pop) + 'pop' + str(gen) + 'gen]' + '[' + platform.system() + ']'
	else:
		SUBJECT = 'Last model created! [' + str(pop) + 'pop' + str(gen) + 'gen]' + '[' + platform.system() + ']'

	mail_content = "Parameters: " + parameters + "\n\n" + str(variables.MODEL['bests'][len(variables.MODEL['bests']) - 1]) + "\n"
	mail_content += "\n\nTotal tweets: " + str(variables.POSITIVE_MESSAGES + variables.NEGATIVE_MESSAGES + variables.NEUTRAL_MESSAGES) + " [" + str(variables.POSITIVE_MESSAGES) + " positives, " + str(variables.NEGATIVE_MESSAGES) + " negatives and " + str(variables.NEUTRAL_MESSAGES) + " neutrals]\n"
	mail_content += "Fitness (F1 pos and neg): " + str(variables.BEST['fitness']) + " [" + str(variables.fitness_positive + variables.fitness_negative + variables.fitness_neutral) + " correct evaluations] ["+ str(variables.fitness_positive) + " positives, " + str(variables.fitness_negative) + " negatives and " + str(variables.fitness_neutral) + " neutrals]\n"
	mail_content += "\nFitness evolution: " + str(variables.HISTORY['fitness']['best']) + "\n"

	yag = yagmail.SMTP(FROM, 'sclassifier123')
	yag.send(TO, SUBJECT, mail_content)