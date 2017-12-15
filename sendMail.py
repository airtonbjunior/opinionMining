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
	else:
		SUBJECT = 'Last model created! [' + str(pop) + 'pop' + str(gen) + 'gen]' + '[' + platform.system() + ']'

	yag = yagmail.SMTP(FROM, 'sclassifier123')
	yag.send(TO, SUBJECT, content)