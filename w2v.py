import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import re
from collections import defaultdict
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error
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

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def main():
    # with open("train_processed.csv") as f:
    #     content = f.readlines()

    data = pd.read_csv("./kindabalanced.csv")
    print('Load data.')

    comments = []
    lab = []
    for i in data.index:
        x = data.iloc[i]
        comments.append(x[2])
        # lab.append(''.join(map(str, map(int, (x[2], x[3], x[4], x[6], x[7])))))
        lab.append('%03d' % int(x[9]))

    print('Data ready.')
    print('Labels: ', str(len(set(lab))))
    print(set(lab))
    length = int(len(comments) * 0.9)


    #model_cbow = Word2Vec(comments, size=200, workers=14, sg=0, window=5)
    #model_cbow.save('./model_cbow')
    model_cbow = Word2Vec.load('./model_cbow')

    #m_w2v = KeyedVectors.load_word2vec_format('/home/s3612406/GoogleNews-vectors-negative300.bin.gz', binary=True)
    #m_w2v2 = KeyedVectors.load_word2vec_format('/home/s3612406/enwiki_20180420_300d.txt.bz2')
    #w2v = dict(zip(model_cbow.wv.index2word, model_cbow.wv.vectors))
    w2v = dict(zip(model_cbow.wv.index2word, model_cbow.wv.vectors))
    vectorizer = TfidfEmbeddingVectorizer(w2v)
    vectors = vectorizer.fit_transform(comments)
    print('Word2Vec model loaded.')

    #Define train and test sets (10% test)
    x_train, x_test, y_train, y_test = train_test_split(vectors, lab, test_size=0.1, random_state=0)

    lb = preprocessing.LabelBinarizer()
    lb.fit(y_train)

    #Train and test classifier
    clf = svm.SVC(probability=True)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    y_proba = clf.predict_proba(x_test)
    y_vec = lb.transform(y_test)
    print('MAE:', mean_absolute_error(y_vec, y_proba))
    # output = {'id': [],
    #           'insult': [],
    #           'obscene': [],
    #           'severe_toxic': [],
    #           'threat': [],
    #           'toxic': [],
    #           'toxicity': [],
    #           'insult_pred': [],
    #           'obscene_pred': [],
    #           'severe_toxic_pred': [],
    #           'threat_pred': [],
    #           'toxic_pred': []}
    # for i in data.index:
    #     x = data.iloc[i]
    #     y = y_proba[i]
    #     for col in x.columns:
    #         output[col].append(x[col])
    #     output['insult_pred'] = y[1]
    #     output['obscene_pred'] = y[2]
    #     output['severe_toxic_pred'] = y[3]
    #     output['threat_pred'] = y[4]
    #     output['toxic_pred'] = y[5]

    # out = pd.DataFrame(data=output)
    # out.to_csv('results/w2v.csv')

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    lab_order = ['000','001','010','011','100','101','110','111']
    c_mat = confusion_matrix(y_test, y_pred, labels=lab_order)
    print (c_mat)
    sn.set()
    fig, ax = plt.subplots(figsize=(15,10))
    ax = sn.heatmap(c_mat, annot=True, vmax=np.mean(c_mat)+60,vmin=-20,fmt='g')
    ax.set_title('Word2vec confusion matrix', fontsize=20, fontdict={})
    plt.yticks(np.arange(8)+0.5,lab_order,va="center",fontsize="20")
    plt.xticks(np.arange(8)+0.5,lab_order,va="center",fontsize="20")
    plt.ylabel("Gold Standard",fontsize="20")
    plt.xlabel("Predictions",fontsize="20")
    plt.savefig('fig2.png')


if __name__ == '__main__':
    main()
