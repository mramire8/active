__author__ = 'mramire8'
__copyright__ = "Copyright 2013, ML Lab"
__version__ = "0.2"
__status__ = "Development"

import sys
import os

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("../"))
sys.path.append(os.path.abspath("../experiment/"))

from experiment.experiment_utils import split_data_sentences, parse_parameters_mat, clean_html, set_cost_model
from datautil.load_data import load_from_file, split_data
from datautil.textutils import StemTokenizer
from learner.adaptive_lr import LogisticRegressionAdaptive
import argparse
import numpy as np
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.datasets.base import Bunch
import time
import nltk
import matplotlib.pyplot as plt
from collections import Counter

#############  COMMAND LINE PARAMETERS ##################
ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawTextHelpFormatter)
ap.add_argument('--train',
                metavar='TRAIN',
                default="imdb",
                help='training data (libSVM format)')

ap.add_argument('--seed',
                metavar='SEED',
                type=int,
                default=876543210,
                help='Max number of iterations')

ap.add_argument('--classifier',
                metavar='CLASSIFIER',
                type=str,
                default="lr",
                choices=['lr', 'lradapt'],
                help='underlying classifier')

args = ap.parse_args()
rand = np.random.mtrand.RandomState(args.seed)
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

print args
print

def get_data(clf, train, cats, fixk, min_size, vct, raw, limit=2):
    import copy
    min_size = 10

    args.fixk = None

    data, vct2 = load_from_file(train, cats, fixk, min_size, vct, raw=raw)

    print("Data %s" % args.train)
    print("Data size %s" % len(data.train.data))


    ### SENTENCE TRANSFORMATION
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

    ## delete <br> to "." to recognize as end of sentence
    data.train.data = clean_html(data.train.data)
    data.test.data = clean_html(data.test.data)

    print("Train:{}, Test:{}, {}".format(len(data.train.data), len(data.test.data), data.test.target.shape[0]))
    ## Get the features of the sentence dataset

    ## create splits of data: pool, test, oracle, sentences
    expert_data = Bunch()
    train_test_data = Bunch()

    expert_data.sentence, train_test_data.pool = split_data(data.train)
    expert_data.oracle, train_test_data.test = split_data(data.test)

    data.train.data = train_test_data.pool.train.data
    data.train.target = train_test_data.pool.train.target

    data.test.data = train_test_data.test.train.data
    data.test.target = train_test_data.test.train.target

    ## convert document to matrix
    data.train.bow = vct.fit_transform(data.train.data)
    data.test.bow = vct.transform(data.test.data)

    #### EXPERT CLASSIFIER: ORACLE
    print("Training Oracle expert")

    labels, sent_train = split_data_sentences(expert_data.oracle.train, sent_detector, vct, limit=limit)
    print len(sent_train)
    expert_data.oracle.train.data = sent_train
    expert_data.oracle.train.target = np.array(labels)
    expert_data.oracle.train.bow = vct.transform(expert_data.oracle.train.data)
    print expert_data.oracle.train.bow.shape
    # exp_clf = linear_model.LogisticRegression(penalty='l1', C=args.expert_penalty)
    exp_clf = copy.copy(clf)
    exp_clf.fit(expert_data.oracle.train.bow, expert_data.oracle.train.target)

    #### EXPERT CLASSIFIER: SENTENCES
    print("Training sentence expert")
    labels, sent_train = split_data_sentences(expert_data.sentence.train, sent_detector, vct, limit=limit)

    expert_data.sentence.train.data = sent_train
    expert_data.sentence.train.target = np.array(labels)
    expert_data.sentence.train.bow = vct.transform(expert_data.sentence.train.data)

    sent_clf = None
    # if args.cheating:
    sent_clf = copy.copy(clf)
    # sent_clf = linear_model.LogisticRegression(penalty='l1', C=args.expert_penalty)
    sent_clf.fit(expert_data.sentence.train.bow, expert_data.sentence.train.target)

    return exp_clf, data, vct, sent_clf, expert_data


def calibrate_scores(n_scores, bounds=(.5,1)):
    delta = 1.* (bounds[1] - bounds[0]) / (n_scores -1)
    calibrated = (np.ones(n_scores)*bounds[1]) - (np.array(range(n_scores))*delta)
    return calibrated


def model_score_distribution(clf, test, ground_truth, title):
    predictions = clf.predict_proba(test)
    predicted_labels = clf.predict(test)
    scores = predictions.max(axis=1)
    score_histogram(predictions[:, 0], ground_truth, scores, title)
    labels = range(len(np.unique(ground_truth)))
    score_confusion_matrix(predicted_labels, ground_truth, labels)


def score_confusion_matrix(predicted, true_labels, labels):
    cm = confusion_matrix(true_labels, predicted, labels=labels)
    print "Predicted -->"
    print "\t" + "\t".join(str(l) for l in np.unique(true_labels))
    for l in np.unique(true_labels):
        print "{}\t{}".format(l,"\t".join(["{}".format(r) for r in cm[l]]))


def score_histogram(predicted, ground_truth, scores, title):
    import matplotlib as mpl
    # mpl.style.use('fivethirtyeight')
    mpl.style.use('ggplot')
    # mpl.style.use('bmh')
    total =len(ground_truth)

    plt.subplot(121)
    c0 = predicted[ground_truth==0]
    c1 = predicted[ground_truth==1]
    n, bins, patches = plt.hist([c0, c1], stacked=True, bins=10, align='mid',label=['y=0', 'y=1'])
    plt.title(title + 'Distribution $P_{L}(y=0|x)$ $y=0$ (mean=%.2f, N=%d)' % (np.mean(predicted), total))
    plt.xlabel("Score $P_{L}(\hat{y}|x)$")
    plt.ylabel("Frequency")
    plt.legend()

    plt.subplot(122)
    n, bins, patches = plt.hist([scores[ground_truth==0], scores[ground_truth==1]], stacked=True, bins=10, label=['y=0', 'y=1'], align='mid')
    plt.title(title + 'Distribution with respect to $y=1$ (mean=%.2f, N=%d)' % (np.mean(predicted), total))
    plt.xlabel("Score $P_{L}(\hat{y}|x)$")
    plt.ylabel("Frequency")

    plt.show()

def normalized(scores):
    c = Counter(scores)
    t = sum(c.values())
    w = [1.* x / t for x in c.values()]
    return c.keys(), w

def plot_histogram_normed(scores, target, title, range_x=np.arange(0,1.01,.1)):
    fig = plt.figure()
    c0 = scores[target==0]
    c1 = scores[target==1]
    print scores.shape
    x0, w0 = normalized(c0)
    x1, w1 = normalized(c1)
    n, bins, patches = plt.hist([x0, x1] , weights=[w0,w1], stacked=True, bins=10, align='mid',label=['y=0', 'y=1'])
    plt.title(title + ' Distribution $P_{L}(y=0|x)$ $y=0$ (mean=%.2f, N=%d)' % (np.mean(scores), len(target)), fontsize=14)
    plt.xlabel("Score $P_{L}(\hat{y}|x)$")
    plt.ylabel("Frequency")
    plt.xticks(range_x)
    plt.legend()


def get_score_distribution(clf, test, title):
    ground_truth = test.target
    model_score_distribution(clf, test.bow, ground_truth, title)

def get_all_score_distribution(clf, train, test, sizes):
    print "TESTING SIZE OF TRAINING"
    for s in sizes:
        print "-"*40
        print "Training size: ", s
        title = "Training size {}".format(s)
        clf.fit(train.bow[:s], train.target[:s])
        get_score_distribution(clf, test, title)

def main():

    vct = TfidfVectorizer(encoding='latin1', min_df=5, max_df=1.0, binary=False, ngram_range=(1, 1),
                      token_pattern='\\b\\w+\\b', tokenizer=StemTokenizer())

    print("Start loading ...")

    ########## NEWS GROUPS ###############
    # easy to hard. see "Less is More" paper: http://axon.cs.byu.edu/~martinez/classes/678/Presentations/Clawson.pdf
    categories = [['alt.atheism', 'talk.religion.misc'],
                  ['comp.graphics', 'comp.windows.x'],
                  ['comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware'],
                  ['rec.sport.baseball', 'sci.crypt']]

    if args.classifier == 'lr':
        clf = linear_model.LogisticRegression(penalty='l1', C=1)
    else:
        clf = LogisticRegressionAdaptive(penalty='l1', C=1)

    exp_clf, data, vct, sent_clf, expert_data = get_data(clf, args.train, [categories[0]], None, None, vct, raw=True)  # expert: classifier, data contains train and test

    print "\nExpert: %s " % exp_clf

    print ("Sentences scoring")
    t0 = time.time()
    ### experiment starts

    print "Expert: :", exp_clf
    print "Sentence:", sent_clf
    print "Student:", clf

    test = expert_data.sentence.train
    print test.keys()
    # get_score_distribution(exp_clf, test, "Oracle ")

    get_all_score_distribution(clf, data.train, test, range(50, 2051, 200))
    print "Elapsed time %s seconds" % (time.time()-t0)

if __name__ == '__main__':
    main()


