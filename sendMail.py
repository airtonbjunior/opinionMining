import yagmail

def send_mail(ith, total, content):
	FROM = 'sclassifier@gmail.com'
	TO = 'sclassifier@gmail.com'
	if(ith < total):
		SUBJECT = 'Model ' + str(ith) + ' of ' + str(total) + ' created!'
		print("SUBJECT: " + SUBJECT) 
	else:
		SUBJECT = 'Last model created!'
		print("SUBJECT: " + SUBJECT)
	#TEXT = 'Hey! A new model are created! Check this out!'

	yag = yagmail.SMTP(FROM, 'sclassifier123')
	yag.send(TO, SUBJECT, content)