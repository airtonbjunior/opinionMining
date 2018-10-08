# Airton Bordin Junior
# airtonbjunior@gmail.com
# Federal University of Goias (UFG)
# Computer Science Master's Degree

import time
import variables
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def trainAndPredict(save_only_LR_separate_file=True):
    start = time.time()
    print("\n[begin train and predict]")

    data, data_labels = [], []
    maximum = 0
    MAX_ANALYSIS = 2500

    print("\n [opening positive tweets]")
    with open("datasets/train/positiveTweets.txt") as f:
        for i in f: 
            maximum += 1
            if maximum <= MAX_ANALYSIS:
                data.append(i) 
                data_labels.append('positive')

    maximum = 0
    print("\n [opening negative tweets]")
    with open("datasets/train/negativeTweets.txt") as f:
        for i in f: 
            maximum += 1
            if maximum <= MAX_ANALYSIS:
                data.append(i)
                data_labels.append('negative')

    maximum = 0
    print("\n [opening neutral tweets]")
    with open("datasets/train/neutralTweets.txt") as f:
        for i in f: 
            maximum += 1
            if maximum <= MAX_ANALYSIS:
                data.append(i)
                data_labels.append('neutral')        


    print("\n [calling CountVectorizer with parameters]")
    vectorizer = CountVectorizer(
        analyzer = 'word',
        stop_words = 'english', 
        ngram_range = (1, 2), 
        lowercase = False,
    )

    #vectorizer = FeatureUnion([
    #    ('cv', CountVectorizer(analyzer = 'word', ngram_range = (1,2), lowercase = False)),
    #    ('av_len', AverageLenVectizer(...))
    #])

    features = vectorizer.fit_transform(
        data
    )

    features_nd = features.toarray()

    print("\n[slicing data]")
    X_train, X_test, y_train, y_test  = train_test_split(
            features_nd, 
            data_labels,
            train_size=0.80, 
            random_state=1234)

    from sklearn.linear_model import LogisticRegression
    log_model = LogisticRegression()
    
    print(" [training model]")
    log_model = log_model.fit(X=X_train, y=y_train)
    print(" [training model completed]")
    
    test_tweets = []
    all_lines_test_file = []

    print(" [loading test tweets]")
    with open(variables.SEMEVAL_TEST_FILE, 'r') as t_f:
        for line in t_f:
            test_tweets.append(line.split("\t")[2].strip())
            all_lines_test_file.append(line)
    print(" [test tweets loaded]")

    print(" [begin test tweets predictions]")
    results = []
    results = log_model.predict(vectorizer.transform(test_tweets))
    print(" [end tweets predictions]")

    if save_only_LR_separate_file:
        print(" [saving only LR results on file]")
        with open('LReg_test_results.txt', 'a') as f_s:
            for index, prediction in enumerate(results):
                f_s.write(str(prediction) + "\n")
        print(" [result saved]")
    else:
        print(" [saving LR results on original test file]")
        with open('test_SVM_MS_LReg.txt', 'a') as f_s:
            for index, prediction in enumerate(results):
                f_s.write(all_lines_test_file[index].strip() + "\t" + str(prediction) + "\n")
        print(" [result saved]")

    end = time.time()
    print("[end train and predict][" + str(format(end - start, '.3g')) + " seconds]\n")
    
    #import pickle
    #save_classifier = open("logRegression.classifier","wb") # the pickle file isn't avaible on github because it's too big (>1GB)
    #pickle.dump(log_model, save_classifier)
    #save_classifier.close()


def saveOnlyLR():
    all_LR_result = []
    with open("test_SVM_MS_LReg.txt", 'r') as f:
        with open("LReg_test_results.txt", 'a') as f_r:
            for line in f:
                f_r.write(str(line.split("\t")[6].strip()) + "\n")


