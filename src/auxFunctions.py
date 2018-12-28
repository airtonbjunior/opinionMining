import glob
import nltk
import codecs
from datetime import datetime

IMDB_TRAIN_FOLDER_POS   = 'IMDBdataset/train/pos/'
IMDB_TRAIN_FOLDER_NEG   = 'IMDBdataset/train/neg/'
IMDB_TRAIN_ALL_MESSAGES = 'IMDBdataset/train/imdbAllMessages.txt'

STS_TRAIN_MESSAGES  	= 'datasets/STSGoldOriginal.csv'
STS_TRAIN_ALL_MESSAGES  = 'datasets/STS_Gold_All.txt'


SEMEVAL_TRAIN_MESSAGES = 'datasets/train/twitter-train-cleansed-B.txt'
SEMEVAL_TRAIN_MESSAGES_MODIFIED = 'datasets/train/twitter-train-cleansed-B-POSTagged' + str(datetime.now())[11:13] + str(datetime.now())[14:16] + str(datetime.now())[17:19] + '.txt' 

counter = 0

def loadOriginalIMDBAndSave():
	with open(IMDB_TRAIN_ALL_MESSAGES, 'a') as f:
		files_pos = glob.glob(IMDB_TRAIN_FOLDER_POS + "*.txt")
		for file in files_pos:
			counter =+ 1
			with open(file, 'r') as file_content:
				for line in file_content:
					f.write("positive" + '\t' + line.strip())

		files_neg = glob.glob(IMDB_TRAIN_FOLDER_NEG + "*.txt")
		for file in files_neg:
			counter =+ 1
			with open(file, 'r') as file_content:
				for line in file_content:
					f.write("negative" + '\t' + line.strip())

	print(counter + " messages saved!")


def loadOriginalSTSAndSave():
	with open(STS_TRAIN_ALL_MESSAGES, 'w') as f:
		with open(STS_TRAIN_MESSAGES, 'r') as file_content:
			for line in file_content:
				if(line.split(";")[1] == "\"0\""):
					f.write("negative" + '\t' + "Twitter2013" + '\t' + line.split(";")[2].replace('"', '').strip() + "\t[0  0  0]\n")
				elif(line.split(";")[1] == "\"4\""):
					f.write("positive" + '\t' + "Twitter2013" + '\t' + line.split(";")[2].replace('"', '').strip() + "\t[0  0  0]\n")


def loadOriginalSTSTrainAndSave():
	#with open('/home/airton/Desktop/training.1600000.processed.noemoticon.EDITED.csv', 'w') as f:
	with codecs.open('/home/airton/Desktop/training.1600000.processed.noemoticon.EDITED.csv', "w", "latin-1") as f:
		#with open('/home/airton/Desktop/training.1600000.processed.noemoticon.csv', 'r') as file_content:
		with codecs.open('/home/airton/Desktop/training.1600000.processed.noemoticon.csv', "r", "latin-1") as file_content:
			for line in file_content:
				if(line.split("\",\"")[0] == "\"0"):
					f.write("negative" + '\t' + "Twitter2013" + '\t' + line.split("\",\"")[5].replace('"', '').strip() + "\t[0  0  0]\n")
				elif(line.split("\",\"")[0] == "\"4"):
					f.write("positive" + '\t' + "Twitter2013" + '\t' + line.split("\",\"")[5].replace('"', '').strip() + "\t[0  0  0]\n")


def testPosTag():
	with open(SEMEVAL_TRAIN_MESSAGES, 'r') as f:
		for line in f:			
			tweet = nltk.word_tokenize(line.split('\t')[3])
			print(tweet)
			tagged_tweets = nltk.pos_tag(tweet)
			with open(SEMEVAL_TRAIN_MESSAGES_MODIFIED, 'a') as fw:
				for tagged_tweet in tagged_tweets:
					fw.write(str(tagged_tweet))
					fw.write("\n")

			#adjectives = [word for word,pos in tagged_tweet \
			#	if (pos == 'JJ' or pos == 'JJR' or pos == 'JJS')]
			
			#for adjective in adjectives:
			#	print(adjective)
			#	print("\n")
			

if __name__ == "__main__":
	loadOriginalSTSTrainAndSave()
	#testPosTag()