from __future__ import print_function

print(__doc__)

__author__ = 'mramire8'
__copyright__ = "Copyright 2014, ML Lab"
__version__ = "0.1"
__status__ = "Development"

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

import sys
import os

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("../aal_misc/samples/neutral"))

import argparse
from sklearn import linear_model
from sklearn.naive_bayes import  MultinomialNB
from datautil.textutils import StemTokenizer
from datautil.load_data import *
import numpy as np
from time import time
import pickle
import sklearn.metrics as metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from sklearn.cross_validation import StratifiedKFold

apfk = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawTextHelpFormatter)
apfk.add_argument('--train',
                metavar='TRAIN',
                default="20news",
                help='training data (libSVM format)')

apfk.add_argument('--seed',
                metavar='SEED',
                type=int,
                default=1234567,
                help='random seed')

apfk.add_argument('--reg',
                metavar='REG',
                type=float,
                default=None,
                help='regularization parameter LR-L1')


args = apfk.parse_args()
rand = np.random.RandomState(args.seed)

print(args)
print()

# vct = CountVectorizer(encoding='latin1', min_df=5, max_df=1.0, binary=True, ngram_range=(1, 3),
#                       token_pattern=r'\b\w+\b', tokenizer=StemTokenizer())

vct = TfidfVectorizer(encoding='ISO-8859-1', min_df=1, max_df=1.0, binary=False, ngram_range=(1, 1),
                      token_pattern='\\b\\w+\\b')  #, tokenizer=StemTokenizer())


if (__name__ == '__main__'):
    t0 = time()
    np.set_printoptions(precision=4)

    # easy to hard. see "Less is More" paper: http://axon.cs.byu.edu/~martinez/classes/678/Presentations/Clawson.pdf
    categories = [['alt.atheism', 'talk.religion.misc'],
                  ['comp.graphics', 'comp.windows.x'],
                  ['comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware'],
                  ['rec.sport.baseball', 'sci.crypt']]
    categories = [['student','faculty']]
    reg = args.reg
    print(reg)

    fixk_saved = "{0}10.p".format(args.train)
    try:
        print("Loading existing file... %s " % args.train)
        fixk_file = open(fixk_saved, "rb")
        data = pickle.load(fixk_file)
        fixk_file.close()
        # vectorizer = open("{0}vectorizer.p".format(args.train), "rb")
        # vct = pickle.load(vectorizer)
        # vectorizer.close()
    except (IOError, ValueError):
        print ("Loading from scratch...")
        data = load_dataset(args.train, 10, categories[0], vct, 100, percent=.5, raw=True)
        fixk_file = open(fixk_saved, "wb")
        pickle.dump(data, fixk_file)
        fixk_file.close()
        vectorizer = open("{0}vectorizer.p".format(args.train), "wb")
        pickle.dump(vct, vectorizer)
        vectorizer.close()

    data.train.bow = vct.fit_transform(data.train.data)
    data.test.bow = vct.transform(data.test.data)

    print("Data %s" % args.train)
    print("Data size %s" % len(data.train.data))
    print("Number of features: %s" % len(vct.get_feature_names()))


    clf = linear_model.LogisticRegression(penalty='l1', C=4)
    clf.fit(data.test.bow, data.test.target)

    predictions = clf.predict_proba(data.train.bow)
    predict = clf.predict(data.train.bow)
    print ("Accuracy:", metrics.accuracy_score(data.train.target, predict))

    try:
        coef = clf.coef_[0]
        print ("Non-zero features: ", np.where(coef > 0)[0].shape)
        ind = np.argsort(coef)
        fn = np.array(vct.get_feature_names())
        print (fn[ind[:10]])
        print (fn[ind[-10:]])
    except Exception():
        pass

    print ("Training student")
    sizes = range(50, 1000, 100)
    sizes = [20000]
    # Scoring Function
    # Classification
    # - accuracy
    # - average_precision
    # - f1a
    # - precision
    # - recall
    # - roc_auc
    # Clustering
    # - adjusted_rand_score
    # Regression
    # - mean_absolute_error
    # - mean_squared_error
    # - r2

    # To apply an classifier on this data, we need to flatten the image, to
    # turn the data in a (samples, feature) matrix:
    X = data.train.bow
    y = data.train.target

    from experiment.experiment_utils import split_data_sentences
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    labels, sent_train = split_data_sentences(data.train, sent_detector, vct, limit=2)
    #
    # X = sent_train
    y = np.array(labels)
    X = vct.transform(sent_train)

    labels, sent_train = split_data_sentences(data.test, sent_detector,vct, limit=2)

    data.test.data = sent_train
    data.test.target = np.array(labels)
    data.test.bow = vct.transform(data.test.data)

    n_samples = data.train.bow.shape[0]
    print("Size of training data:", n_samples)
    print("Features:", data.train.bow.shape[1])

    # Set the parameters by cross-validation
    tuned_parameters = [{'C': [2e-3, 2e-2, 2e-1, 2e0, 2e1, 2e2, 2e3]}]
    tuned_parameters = [{'C':  [pow(10,x) for x in range(-3,4)]}]
    # tuned_parameters = [{'C': [1e3, 2e3, 3e3]}]


    # scores = ['accuracy','precision', 'recall']
    measures = ['accuracy']
    for s in sizes:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        kcv = StratifiedKFold(y=y_train[:s], n_folds=5, shuffle=True, random_state=546321)
        for measure in measures:
            print("# Tuning hyper-parameters for %s" % measure)
            print("Size: ", s)
            print(len(y_train[:s]))
            clf_new = linear_model.LogisticRegression(penalty='l1')
            clf = GridSearchCV(clf_new, tuned_parameters, cv=5, scoring=measure, n_jobs=20)
            clf.fit(X_train, y_train)

            print("Best parameters set found on development set:")
            print(clf.best_estimator_)
            print()
            print("Grid scores on development set:")
            print()
            for params, mean_score, scores in clf.grid_scores_:
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean_score, scores.std() / 2, params))
            print()

            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            y_true, y_pred = data.test.target, clf.predict(data.test.bow)
            print(classification_report(y_true, y_pred))
            print()
