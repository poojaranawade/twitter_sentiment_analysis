import nltk
# nltk.download()
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
import json
from tweepy.streaming import StreamListener   
from nltk.classify import ClassifierI
from statistics import mode
from tweepy import API, TweepError, OAuthHandler, Cursor,Stream
from nltk.tokenize import word_tokenize
import csv

class VoteClassifier(ClassifierI):
    
    def __init__(self, *classifiers):
        self._classifiers = classifiers
        
    def votes(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return votes
    
    def classify(self, features):
        Votes = self.votes(features)
        return mode(Votes)
    
    def confidence(self, features):
        Votes = self.votes(features)
        choice_votes = Votes.count(mode(Votes))
        conf = choice_votes / len(Votes)
        return conf
 
short_pos = open("positive.txt","r").read()
short_neg = open("negative.txt","r").read()

all_words = []
documents = []

allowed_word_types = ["J"]

for p in short_pos.split('\n'):
    documents.append( (p, "pos") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

    
for p in short_neg.split('\n'):
    documents.append( (p, "neg") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())
            
#save_documents = open("documents.pickle","wb")
#pickle.dump(documents, save_documents)
#save_documents.close()

documents_f = open("documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]

#save_word_features = open("word_features5k.pickle","wb")
#pickle.dump(word_features, save_word_features)
#save_word_features.close()

word_features5k_f = open("word_features5k.pickle", "rb")
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]

random.shuffle(featuresets)

training_set = featuresets[:10000]

open_file = open("originalnaivebayes5k.pickle", "rb")
classifier = pickle.load(open_file)
open_file.close()

open_file = open("MNB_classifier5k.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open("LogisticRegression_classifier5k.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()

open_file = open("LinearSVC_classifier5k.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()

open_file = open("BernoulliNB_classifier5k.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()
    
open_file = open("SGDC_classifier5k.pickle", "rb")
SGDC_classifier = pickle.load(open_file)
open_file.close()

voted_classifier = VoteClassifier(classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)

def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)


consumer_key = "Enter your consumer key"
consumer_secret = "Enter your consumer secret"
access_token = "Enter your access token"
access_secret = "Enter your access secret"

def get_twitter_auth():
    """Setup Twitter authentication.
    Return: tweepy.OAuthHandler object
    """
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    return auth


def get_twitter_client():
    """Setup Twitter API client.
    Return: tweepy.API object
    """
    auth = get_twitter_auth()
    api = API(auth, wait_on_rate_limit=True)
    return api


if __name__ == '__main__':
    api = get_twitter_client()
    i = 1
    page = 1
    searched_tweets = []
    last_id = -1
    query = input("Enter query to be searched: ")
    max_tweets = 1000
    csvFile = open('result.csv', 'w')

    csvWriter = csv.writer(csvFile)

    while len(searched_tweets) < max_tweets:
        count = max_tweets - len(searched_tweets)
        try:
            new_tweets = api.search(q=query, lang='en', result_type='recent', count=count, max_id=str(last_id - 1))
            if not new_tweets:
                break
            searched_tweets.extend(new_tweets)
            last_id = new_tweets[-1].id
        except TweepError as e:
            break
    posNumber, negNumber = 0, 0
    for tweet in searched_tweets:
        sentiment_value, confidence = sentiment(tweet.text)
        
        if confidence*100 >= 80:
            if sentiment_value == "pos":
                csvWriter.writerow([tweet.created_at, 1])
                posNumber += 1
            elif sentiment_value == "neg":
                csvWriter.writerow([tweet.created_at, 0])
                negNumber += 1
    print("Positive Tweet Percentage: ", posNumber * 100 / (posNumber + negNumber))
    print("Negative Tweet Percentage: ", negNumber * 100 / (posNumber + negNumber))
    csvFile.close()
    
    import pandas as pd
    import matplotlib.pyplot as plt
    
    dataFrame = pd.read_csv('result.csv', header=None)
    plt.plot(dataFrame[0], dataFrame[1])
