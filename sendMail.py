# Airton Bordin Junior
# airtonbjunior@gmail.com
# Federal University of Goias (UFG)
# Computer Science Master's Degree
# 
# Reference: https://stackoverflow.com/a/34139884

import yagmail
import platform

def send_mail(ith, total, pop, gen, content):
	FROM = 'sclassifier@gmail.com'
	TO = 'sclassifier@gmail.com'
	if(ith < total):
		SUBJECT = 'Model ' + str(ith) + ' of ' + str(total) + ' created! [' + str(pop) + 'pop' + str(gen) + 'gen]' + '[' + platform.system() + ']'
		print("SUBJECT: " + SUBJECT) 
	else:
		SUBJECT = 'Last model created! [' + str(pop) + 'pop' + str(gen) + 'gen]' + '[' + platform.system() + ']'
		print("SUBJECT: " + SUBJECT)
	#TEXT = 'Hey! A new model are created! Check this out!'

	yag = yagmail.SMTP(FROM, 'sclassifier123')
	yag.send(TO, SUBJECT, content)