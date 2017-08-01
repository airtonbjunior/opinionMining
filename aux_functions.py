# Airton Bordin Junior
# airtonbjunior@gmail.com
# Federal University of Goias (UFG)
# Computer Science Master's Degree

# Used only once
def saveTweetsFromIdInFile():
    print("[loading tweets to save in a file]")

    file = open("datasets/twitter-2016train-A-full-tweets2.txt","w") 
    
    tweet_parsed = []

    APP_KEY = 'fBoxg0SJUlIKRN84wOJGGCmgz'
    APP_SECRET = 'yhf4LSdSlfmj25WUzvT8YzWmFXf30SFv2w5Qqa3M6wViWZNpYA'
    twitter = Twython(APP_KEY, APP_SECRET, oauth_version=2)
    ACCESS_TOKEN = twitter.obtain_access_token()

    twitter = Twython(APP_KEY, access_token=ACCESS_TOKEN)

    exceptions = 0

    with open('datasets/twitter-2016train-A-part.txt', 'r') as inF:
        for line in inF:
            tweet_parsed = line.split()
            try:
                tweet = twitter.show_status(id=str(tweet_parsed[0]))
                file.write(tweet_parsed[1].strip() + "#@#" + tweet['text'].strip() + "\n") 
            except Exception as ex:
                template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                message = template.format(type(ex).__name__, ex.args)
                print(message)
                exceptions += 1
                continue

    print("[" + str(exceptions) + " exceptions]")
    print("[tweets saved on file]")

    file.close() 


# Used only to load the file (once)
def saveTestTweetsFromFilesIdLoadedSemeval2014():
    print("[loading tweets from test file Semeval 2014]")

    saveFile = open("datasets/test/SemEval2014-task9-test-B-all-tweets.txt","w") 

    tweet_text = []
    tweet_polarity = []
    tweet_base = []

    tweets_loaded = 0
    polarity_loaded = 0

    with open('d:/SemEval2014-task9-test-B-input.txt', 'r') as inF:
        for line in inF:
            tweet_parsed = line.split("\t")
            tweet_text.append(tweet_parsed[3])
            tweets_loaded += 1
            print(tweet_parsed[3])
    
    with open('d:/SemEval2014-task9-test-B-gold.txt', 'r') as inF2:
        for line2 in inF2:
            tweet_parsed = line2.split("\t")
            tweet_polarity.append(tweet_parsed[2])
            tweet_base.append(tweet_parsed[1])
            polarity_loaded += 1
            print(tweet_parsed[2])
    
    for index, item in enumerate(tweet_text):
        saveFile.write(tweet_polarity[index].strip() + "\t" + tweet_base[index] + "\t" + tweet_text[index].strip() + "\n")

    saveFile.close()


    print("Tweets loaded " + str(tweets_loaded))
    print("Polarity loaded " + str(polarity_loaded))

    print("[tweets loaded]")


# get tweets from id (SEMEVAL database)
def getTweetsFromFileIdLoaded():
    print("[loading tweets from file]")

    global MAX_ANALYSIS_TWEETS

    global tweets_semeval
    global tweets_semeval_score

    global positive_tweets
    global negative_tweets

    tweets_loaded = 0

    with open('datasets/twitter-2016train-A-full-tweets.txt', 'r') as inF:
        for line in inF:
            if tweets_loaded < MAX_ANALYSIS_TWEETS:
                tweet_parsed = line.split("#@#")
                try:
                    # i'm ignoring the neutral tweets
                    if(tweet_parsed[0] != "neutral"):
                        tweets_semeval.append(tweet_parsed[1])
                        if(tweet_parsed[0] == "positive"):
                            positive_tweets += 1
                            tweets_semeval_score.append(1)
                        else:
                            negative_tweets += 1
                            tweets_semeval_score.append(-1)

                        tweets_loaded += 1
                # treat 403 exception mainly
                except:
                    #print("exception")
                    continue
    
    print("[tweets loaded]")


# get tweets from id (SEMEVAL database)
def getTweetsFromIds():
    print("[loading tweets]")

    global tweets_semeval
    global tweets_semeval_score

    global positive_tweets
    global negative_tweets

    tweet_parsed = []

    APP_KEY = 'fBoxg0SJUlIKRN84wOJGGCmgz'
    APP_SECRET = 'yhf4LSdSlfmj25WUzvT8YzWmFXf30SFv2w5Qqa3M6wViWZNpYA'
    twitter = Twython(APP_KEY, APP_SECRET, oauth_version=2)
    ACCESS_TOKEN = twitter.obtain_access_token()

    twitter = Twython(APP_KEY, access_token=ACCESS_TOKEN)

    with open('twitter-2016train-A-part.txt', 'r') as inF:
        for line in inF:
            tweet_parsed = line.split()
            try:
                # i'm ignoring the neutral tweets
                if(tweet_parsed[1] != "neutral"):
                    tweet = twitter.show_status(id=str(tweet_parsed[0]))
                    tweets_semeval.append(tweet['text'])
                    if(tweet_parsed[1] == "positive"):
                        positive_tweets += 1
                        tweets_semeval_score.append(1)
                    else:
                        negative_tweets += 1
                        tweets_semeval_score.append(-1)
            # treat 403 exception mainly
            except:
                #print("exception")
                continue
    
    print("[tweets loaded]")