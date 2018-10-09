# Airton Bordin Junior
# airtonbjunior@gmail.com
# Federal University of Goias (UFG)
# Computer Science Master's Degree

import time
import variables

# Train file to textblob classifier (Naive Bayes)
def createTrainFile():
    with open(variables.SEMEVAL_TRAIN_FILE, 'r') as inF:
        
        with open('trainMessages3.json', 'a') as f_w:
            f_w.write("[\n")
            for line in inF:
                tweet_parsed = line.split("\t")
                try:
                    if(tweet_parsed[2] != "neutral"):
                        if(tweet_parsed[2] == "positive"):
                            #if(variables.positive_tweets < variables.MAX_POSITIVES_TWEETS):
                            if(variables.positive_tweets < 1000):
                                f_w.write('{"text": "' + str(tweet_parsed[3].strip().replace('"', '')) + '", "label": "pos"},\n')
                                variables.positive_tweets += 1
                        else:
                            #if(variables.negative_tweets < variables.MAX_NEGATIVES_TWEETS):
                            if(variables.negative_tweets < 1000):
                                f_w.write('{"text": "' + str(tweet_parsed[3].strip().replace('"', '')) + '", "label": "neg"},\n')
                                variables.negative_tweets += 1
                    else:
                        #if(variables.neutral_tweets < variables.MAX_NEUTRAL_TWEETS):
                        if(variables.neutral_tweets < 1000):
                            f_w.write('{"text": "' + str(tweet_parsed[3].strip().replace('"', '')) + '", "label": "neu"},\n')
                            variables.neutral_tweets += 1
                except:
                    print("exception")
                    continue    
            f_w.write("\n]")


def trainNaiveBayesClassifier(file_train):
    from textblob.classifiers import NaiveBayesClassifier
    with open(file_train, 'r') as fp:
        cl = NaiveBayesClassifier(fp, format="json")

        import pickle

        save_classifier = open("naivebayes.classifier","wb") # the pickle file isn't avaible on github because it's too big (>1GB)
        pickle.dump(cl, save_classifier)
        save_classifier.close()


def loadNaiveBayesClassifier(path_classifier="naivebayes.classifier"):
    start = time.time()
    import pickle

    print("[loading classifier]")
    
    classifier_f = open(path_classifier, "rb")
    classifier = pickle.load(classifier_f)
    classifier_f.close()
    
    end = time.time()
    print("  [classifier loaded][" + str(format(end - start, '.3g')) + " seconds]\n")
    
    return classifier


def saveOnlyNaive():
	start = time.time()
	print("[saving Naive values on file]")
	#from functions import *

	all_test_messages = []
	with open('datasets/test/SemEval2014_SVM_Naive_MS_Lreg_S140.txt', 'r') as f:
		for line in f:
			all_test_messages.append(line.split("\t")[2].strip())

	all_results = []
	classifier  = loadNaiveBayesClassifier()

	for msg in all_test_messages:
		msg_class = classifier.classify(msg)
		probs     = classifier.prob_classify(msg)

		all_results.append(str(msg_class) + "\t" + str(probs.prob("pos")) + "\t" + str(probs.prob("neg")) + "\t" + str(probs.prob("neu")) + "\n")

	with open('Naive_test_results.txt', 'a') as f:
		for result in all_results:
			f.write(result)
    

	end = time.time()
	print("  [Naive values saved on file][" + str(format(end - start, '.3g')) + " seconds]\n")


    #with open("datasets/test/SemEval2014_SVM_Naive_MS_Lreg_S140.txt", 'r') as f:
    #    with open("Naive_test_results.txt", 'a') as f_r:
    #        for line in f:
    #            f_r.write(str(line.split("\t")[4].strip()) + "\n")


def includeNaiveBayesValuesOnTestFile():
    start = time.time()
    print("\n[including NaiveBayes values on test file]")

    classifier = loadNaiveBayesClassifier()

    with open('testWithSVM_Naive.txt', 'a') as f_w:
        with open(variables.SEMEVAL_TEST_FILE, 'r') as inF:
            for line in inF:
                t_class = ""
                if (classifier.classify(line.split("\t")[2].strip()) == "pos"):
                    t_class = "positive"
                elif (classifier.classify(line.split("\t")[2].strip()) == "neg"):
                    t_class = "negative"
                elif (classifier.classify(line.split("\t")[2].strip()) == "neu"):
                    t_class = "neutral"

                f_w.write(str(line.split("\t")[0]) + "\t" + line.split("\t")[1] + "\t" + line.split("\t")[2] + "\t" + str(line.split("\t")[3]).strip() + "\t" + t_class + "\n")


    end = time.time()
    print("  [values included][" + str(format(end - start, '.3g')) + " seconds]\n")