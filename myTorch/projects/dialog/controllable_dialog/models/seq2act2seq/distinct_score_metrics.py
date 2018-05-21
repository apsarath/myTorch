import nltk
from nltk.util import ngrams
from collections import Counter
import logging

def distinct_scores(samples_file, k):
    words = []
    with open(samples_file, "r") as f:
        for line in f:
            words += line.split(" ")

    total_num_words = len(words)
    text = nltk.Text(words)
    bigrams = Counter(ngrams(text, 2))
    unigrams = Counter(ngrams(text, 1))
    distinct1 = len(unigrams)/total_num_words
    distinct2 = len(bigrams)/total_num_words
    logging.info("Distinct 1 : {}".format(distinct1)
    logging.info("Distinct 2 : {}".format(distinct2)
    return distinct1, distinct2
