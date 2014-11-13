from __future__ import print_function
# print(__doc__)
__author__ = 'mramire8'
__copyright__ = "Copyright 2013, ML Lab"
__version__ = "0.1"
__status__ = "Development"

import sys
import os

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("../aal_misc/samples/neutral"))

import argparse
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
from learner.adaptive_lr import LogisticRegressionAdaptive
from datautil.textutils import StemTokenizer
from experiment.experiment_utils import split_data_sentences
from datautil.load_data import *

import numpy as np
from time import time
import pickle
import sklearn.metrics as metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
import matplotlib as mpl
import brewer2mpl
import nltk
from sklearn.cross_validation import StratifiedKFold, cross_val_score, KFold, ShuffleSplit

apfk = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawTextHelpFormatter)
apfk.add_argument('--train',
                metavar='TRAIN',
                default="arxiv",
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

apfk.add_argument('--fixk',
                metavar='FIXK',
                type=int,
                default=50,
                help='neutral training data text')



args = apfk.parse_args()
rand = np.random.RandomState(args.seed)

# print(args)
# print()
import itertools
# vct = CountVectorizer(encoding='latin1', min_df=5, max_df=1.0, binary=True, ngram_range=(1, 1),
#                       token_pattern=r'\b\w+\b')#, tokenizer=StemTokenizer())
# #
vct = TfidfVectorizer(encoding='latin1', min_df=5, max_df=1.0, binary=False, ngram_range=(1, 1),
                      token_pattern='\\b\\w+\\b')  #, tokenizer=StemTokenizer())


def uncertainty(x):
    return x[np.argmin(x)]


if (__name__ == '__main__'):

    ## what to print and show
    print_non_zero = False
    print_cv_accu = True
    show_learning_curve = True

    ## configuration settings

    test_case = "sent-sent"   # sent-sent, sent-doc, doc-doc, doc-sent
    lim = 2                # none: original method, 1:one character, 0:no limit
    classifier = 'lradapt'

    if classifier == "mnb":
        C_values = [1]
    elif classifier == "lr":
        C_values = [2e-3, 2e-2, 2e-1, 2e0, 2e1, 2e2, 2e3]
        # C_values = [1, 2, 3, 4, 5]
        C_values = [1, 10]
    else:
        C_values = [1]

    sizes = range(50, 1000, 100)
    # sizes = range(50, 2000, 100)
    sizes = range(50, 2000, 100)

    t0 = time()
    np.set_printoptions(precision=4)
    # rand = np.random.mtrand.RandomState(8675309)

    # easy to hard. see "Less is More" paper: http://axon.cs.byu.edu/~martinez/classes/678/Presentations/Clawson.pdf
    categories = [['alt.atheism', 'talk.religion.misc'],
                  ['comp.graphics', 'comp.windows.x'],
                  ['comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware'],
                  ['rec.sport.baseball', 'sci.crypt']]
    categories = [['student','faculty']]

    categories = [['comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware']]

    categories = [['cs.AI','cs.LG'],
                  ['physics.comp-ph','physics.data-an']]

    reg = args.reg
    print (reg)

    print ("*" * 40)
    print ("Starting - Get data")
    print ("*" * 40)

    fixk_saved = "{0}10.p".format(args.train)
    try:
        print ("Loading existing file... %s " % args.train)
        fixk_file = open(fixk_saved, "rb")
        data = pickle.load(fixk_file)
        fixk_file.close()
        # vectorizer = open("{0}vectorizer.p".format(args.train), "rb")
        # vct = pickle.load(vectorizer)
        # vectorizer.close()
    except (IOError, ValueError):
        print ("Loading from scratch...")
        #name, fixk, categories, vct, min_size, raw=False, percent=.5
        data = load_dataset(args.train, 10, categories[1], vct, 100, percent=.5, raw=True)
        fixk_file = open(fixk_saved, "wb")
        pickle.dump(data, fixk_file)
        fixk_file.close()
        # vectorizer = open("{0}vectorizer.p".format(args.train), "wb")
        # pickle.dump(vct, vectorizer)
        # vectorizer.close()

    print("Dataset name: %s" % args.train)
    print("Categories: %s" % data.train.target_names)

    data.train.bow = vct.fit_transform(data.train.data)
    data.test.bow = vct.transform(data.test.data)

    print("Original data size: ", data.train.bow.shape)
    print("Original distribution: ", 1.* data.train.target.sum()/data.train.bow.shape[0])
    ## if I want sentences in the data instead of documents
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    print("Type of data:", test_case)
    if test_case == "sent-sent" or test_case == "sent_doc":
        labels, sent_train = split_data_sentences(data.train, sent_detector,vct, limit=lim)
        data.train.data = sent_train
        data.train.target = np.array(labels)
        data.train.bow = vct.transform(data.train.data)
    if test_case == "sent-sent" or test_case == "doc-sent":
        labels, sent_train = split_data_sentences(data.test, sent_detector, vct, limit=lim)
        data.test.data = sent_train
        data.test.target = np.array(labels)
        data.test.bow = vct.transform(data.test.data)

    print("Data size %s" % len(data.train.data))
    print("Number of features: %s" % len(vct.get_feature_names()))
    print ("Values in vector (min-max):", data.train.bow.min(), data.train.bow.max())
    print ("Label Distribution on 1:", data.train.target.sum(), 1.* data.train.target.sum() / len(data.train.target))

    # expert = linear_model.LogisticRegression(penalty='l1', C=0.3)

    ## Show number of non-zero features
    print("Vectorizer: %s" % vct)
    if print_non_zero:
        print ("*" * 40)
        print ("Non-zero weights in classifier")
        print ("*" * 40)
        for c in C_values:
            print ("C=",c)
            if classifier == "mnb":
                clf = MultinomialNB(alpha=c)
            elif classifier == 'lr':
                clf = linear_model.LogisticRegression(penalty='l1', C=c)
            else:
                clf = LogisticRegressionAdaptive(penalty='l1', C=c)
            clf.fit(data.train.bow, data.train.target)
            print("Classifier name:", clf.__class__.__name__, "C=", clf.C)
            print("Train data:", len(data.train.target))
            # predictions = clf.predict_proba(data.test.bow)
            predict = clf.predict(data.test.bow)
            print ("Accuracy:", metrics.accuracy_score(data.test.target, predict),)
            try:
                coef = clf.coef_[0]
                print("Non-zero features: ", len([x for x in coef if x > 0.0]))
                # ind = np.argsort(coef)
                # fn = np.array(vct.get_feature_names())
                # print fn[ind[:10]]
                # print fn[ind[-10:]]
            except Exception:
                pass
    if print_cv_accu:
        print("*" * 40)
        print("CV-Accuracy of the model")
        print ("*" * 40)

        for c in C_values:
            print ("C=",c)
            if classifier == "mnb":
                clf = MultinomialNB(alpha=c)
            elif classifier == 'lr':
                clf = linear_model.LogisticRegression(penalty='l1', C=c)
            else:
                clf = LogisticRegressionAdaptive(penalty='l1', C=c)
            # print clf
            print("Classifier name:", clf.__class__.__name__, "C=", clf.C)
            cv_scores = cross_val_score(clf, data.train.bow, data.train.target, cv=5, n_jobs=20)
            print("5-f CV Accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))
            print("Train data:", len(data.train.target))
            print("Testing data:", len(data.test.target))
            # predictions = clf.predict_proba(data.test.bow)
            clf.fit(data.train.bow, data.train.target)
            predict = clf.predict(data.test.bow)
            print ("Accuracy:", metrics.accuracy_score(data.test.target, predict),)

    # Scoring Function
    # Classification
    # - accuracy
    # - average_precision
    # - f1
    # - precision
    # - recall
    # - roc_auc
    # Clustering
    # - adjusted_rand_score
    # Regression
    # - mean_absolute_error
    # - mean_squared_error
    # - r2
    # mpl.style.use('bmh')
    if show_learning_curve:

        print("*" * 40)
        print("Learning Curve")
        print("*" * 40)

        show_train = False
        plt.grid(color='.75', which='major', axis='y', linestyle='--', linewidth=1)
        plt.title("Data:" + args.train + " on " + test_case)
        plt.xlabel("Number of Labels")
        plt.ylabel('accuracy'.title())
        col = brewer2mpl.get_map('Set1', 'qualitative', 7).mpl_colors
        colors_n = itertools.cycle(col)

        random_state = np.random.RandomState(5612)
        # kcv = StratifiedKFold(y=data.train.target, n_folds=5, random_state=random_state,shuffle=True)


        indices = np.arange(data.train.target.shape[0])
        random_state.shuffle(indices)

        data.train.target = data.train.target[indices]
        # Use an object array to shuffle: avoids memory copy
        data_lst = np.array(data.train.data, dtype=object)
        data_lst = data_lst[indices]
        data.train.data = data_lst.tolist()
        data.train.bow = data.train.bow[indices]

        kcv = KFold(len(data.train.target), n_folds=5, random_state=random_state,shuffle=True)

        # print ("Vectorizer:", vct)

        for c in C_values:
            current_color = colors_n.next()
            if classifier == "mnb":
                clf = MultinomialNB(alpha=c)
            elif classifier == 'lr':
                clf = linear_model.LogisticRegression(penalty='l1', C=c)
            else:
                clf = LogisticRegressionAdaptive(penalty='l1', C=c)
            scoring_fn = 'accuracy'
            print("Classifier name:", clf.__class__.__name__, "C=", clf.C)
            print("CV data:", data.train.bow.shape)
            train_sizes, train_scores, test_scores = learning_curve(
                clf, data.train.bow.tocsr(), data.train.target, train_sizes=sizes, cv=5, scoring=scoring_fn, n_jobs=20)

            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = 1.0 * np.std(test_scores, axis=1) / np.sqrt(5.0)

            plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1, color=current_color)
            plt.plot(train_sizes, test_scores_mean, 'o-', mfc='white', linewidth=2, mew=2, markersize=10, mec=current_color, color=current_color,
                     # label="Cross-validation score")
                     label="C={}".format(clf.C))

            if show_train:
                plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                                 train_scores_mean + train_scores_std, alpha=0.1,
                                 color="r")
                plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                         label="Training score")
            print ("-"*40)
            print ("\nCOST\tMEAN\tSTDEV")
            print ("\n".join(["{0}\t{1:.3f}\t{2:.4f}".format(c,m,s) for c,m,s in zip(train_sizes, test_scores_mean, test_scores_std)]))
        plt.legend(loc="best")
        # plt.savefig('lr-{0}.png'.format(vct.__class__.__name__), bbox_inches="tight", dpi=200, transparent=True)
        plt.savefig('lradapt-sent-sent.png', bbox_inches="tight", dpi=200, transparent=True)
        plt.show()
    print("\nElapsed time: %.4f" % (time() - t0))
