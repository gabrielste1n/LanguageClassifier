import nltk
import pickle
import string
import numpy as np
import pandas as pd
nltk.download('punkt')
nltk.download('stopwords')
from itertools import chain
from pandas import DataFrame
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier as nbc

# LOAD THE DATA SET

data = pd.read_csv("lang_data.csv", sep=',',header=0).to_dict('list')

# LABEL THE DATA

labelled_data = []

for count in range(len(data['text'])):
    text = str(data['text'][count]).strip()
    language = str(data['language'][count]).strip()
    if text and language:
        labelled_data.append((text, language))

# IMPLEMENT DIMENSION REDUCTION

# REMOVE 10 MOST FREQUENT WORDS

freq = pd.Series(' '.join(str(data['text'])).split()).value_counts()[:10]
freq = list(freq.index)

# REMOVE 10 LEAST FREQUENT WORDS

least_freq = pd.Series(' '.join(str(data['text'])).split()).value_counts()[-10:]
least_freq = list(least_freq.index)

# REMOVE ENGLISH AND DUTCH STOP WORDS (AFRIKAANS NOT YET SUPPORTED)

dimension_reduction = stopwords.words('english') + stopwords.words('dutch') + list(string.punctuation) + freq + least_freq

# CREATE FEATURE SET

vocabulary = set(word.lower() for passage in labelled_data for word in list(filter(lambda word: word not in dimension_reduction, word_tokenize(str(passage[0])))))


def text_features(text):
    words = set(word_tokenize(str(text)))
    features = {}
    for word in vocabulary:
        features[word] = (word in words)
    return features


feature_set = [(text_features(text), language) for (text, language) in labelled_data]

open_classifier = open("LanguageClassificationModel.pickle", "rb")
classifier = pickle.load(open_classifier)
open_classifier.close()

user_input = input("Enter text phrase to classify:\n")
while user_input != "quit":
    print(classifier.classify(text_features(user_input)))
    user_input = input("Enter Text Phrase To Classify:\n")

