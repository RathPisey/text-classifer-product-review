import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

import numpy as np
import re

# initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# read positive and negative words from text files
with open('positive-words.txt', 'r') as f:
    positive_words = f.read().splitlines()

with open('negative-words.txt', 'r') as f:
    negative_words = f.read().splitlines()

# list of pronouns
pronouns = ['I', 'me', 'my', 'you', 'your']

# function to extract features
def extract_features(review, label):
    review = review[0]
    # tokenize the review text
    tokens = word_tokenize(review)
    # remove punctuation
    review = re.sub(r'[^\w\s]', '', review)

    # remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]

    # lemmatize tokens
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # count of positive words
    pos_count = 0
    for word in positive_words:
        pos_count += lemmatized_tokens.count(word)

    # count of negative words
    neg_count = 0
    for word in negative_words:
        neg_count += lemmatized_tokens.count(word)

    # presence of 'no'
    no_count = 1 if 'no' in lemmatized_tokens else 0

    # count of pronouns
    pron_count = 0
    for word in pronouns:
        pron_count += lemmatized_tokens.count(word)

    # presence of '!'
    excl_count = 1 if '!' in review else 0

    # log(length of review)
    log_length = np.log(len(lemmatized_tokens))

    return [pos_count, neg_count, no_count, pron_count, excl_count, log_length]
