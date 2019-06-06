import numpy as np
import pandas as pd
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from gensim.models import Word2Vec, KeyedVectors


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(list(word2vec.values())[0])

    def fit(self, X, y):
        return self

    def transform(self, X):  # mean
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(list(word2vec.values())[0])

    def fit(self, X, y=None):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])

    def fit_transform(X):
        self.fit(X)
        return self.transform(X)


def main():
    with open("jigsaw-toxic-comment-classification-challenge/train.csv") as f:
        content = f.readlines()

    data = pd.read_csv("jigsaw-toxic-comment-classification-challenge/train.csv")
    print('Load data.')

    comments = []
    classes = []
    for i in data.index:
        x = data.iloc[i]
        comments.append(x[1])
        classes.append(''.join(map(str, (x[2], x[4], x[5], x[6], x[7]))))

    print('Data ready.')



    m_w2v = KeyedVectors.load_word2vec_format('/Users/willskywalker/Documents/Workplace/GoogleNews-vectors-negative300.bin.gz', binary=True)
    w2v = dict(zip(m_w2v.wv.index2word, m_w2v.wv.vectors))
    vectorizer = TfidfEmbeddingVectorizer(w2v)
    vectors = vectorizer.fit_transform(comments)
    print('Word2Vec model loaded.')

    #Define train and test sets (10% test)
    x_train = vectors[:143615]
    x_test = vectors[143615:]
    y_train = lab[:143615]
    y_test = lab[143615:]

    #Train and test classifier
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    print ("Score:\n", score)


if __name__ == '__main__':
    main()
