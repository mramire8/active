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

from datautil.textutils import StemTokenizer
from datautil.load_data import *

import numpy as np
from time import time
import pickle

apfk = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawTextHelpFormatter)
apfk.add_argument('--train',
                metavar='TRAIN',
                default="aviation",
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

print args
print

vct = CountVectorizer(encoding='latin1', min_df=5, max_df=1.0, binary=True, ngram_range=(1, 3),
                      token_pattern=r'\b\w+\b', tokenizer=StemTokenizer())
#
# vct = CountVectorizer(encoding='latin1', min_df=5, max_df=1.0, binary=True, ngram_range=(1, 3),
#                       token_pattern=r'\b\w+\b')

def uncertainty(x):
    return x[np.argmin(x)]


if (__name__ == '__main__'):
    t0 = time()
    np.set_printoptions(precision=4)
    # rand = np.random.mtrand.RandomState(8675309)

    # easy to hard. see "Less is More" paper: http://axon.cs.byu.edu/~martinez/classes/678/Presentations/Clawson.pdf
    categories = [['alt.atheism', 'talk.religion.misc'],
                  ['comp.graphics', 'comp.windows.x'],
                  ['comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware'],
                  ['rec.sport.baseball', 'sci.crypt']]

    reg = args.reg
    print reg

    # data = load_dataset(args.train, 10, categories[0], vct, 100, raw=False,  percent=.5)  ## load all dataset

    fixk_saved = "{0}10.p".format(args.train)
    try:
        print "Loading existing file... %s " % args.train
        fixk_file = open(fixk_saved, "rb")
        data = pickle.load(fixk_file)
        fixk_file.close()
        vectorizer = open("{0}vectorizer.p".format(args.train), "rb")
        vct = pickle.load(vectorizer)
        vectorizer.close()
    except (IOError, ValueError):
        print "Loading from scratch..."
        data = load_dataset(args.train, 10, categories[0], vct, 100, percent=.5)
        fixk_file = open(fixk_saved, "wb")
        pickle.dump(data, fixk_file)
        fixk_file.close()
        vectorizer = open("{0}vectorizer.p".format(args.train), "wb")
        pickle.dump(vct, vectorizer)
        vectorizer.close()


    # train on test and test on train (test on k words of the train)

    #swap train and test
    # tmp = dataset.train
    # dataset.train = dataset.test
    # dataset.test = tmp

    # test_data = Bunch()
    # test_data.data = data.train.bow.tocsr()   # full words, for training
    # test_data.fixk = data.train.bowk.tocsr()  # k words BOW for querying
    # test_data.target = data.train.target
    # # pool.predicted = []
    # test_data.kwords = np.array(data.train.kwords)  # k words
    # test_data.remaining = set(range(test_data.data.shape[0]))  # indices of the pool

    # train_x = data.test.data
    # train_x = vct.fit_transform(train_x)
    print("Data %s" % args.train)
    print("Data size %s" % len(data.train.data))
    print
    # kvalues = [10, 100]
    # threshold = np.arange(0.2, 0.46, 0.2)
    # for reg in np.arange(0.1, 0.3, 0.1):

    kvalues = [10, 25, 50, 75, 100]
    threshold = np.arange(0.3, 0.46, 0.05)

    count_results = []

    expert = linear_model.LogisticRegression(penalty='l1', C=0.3)
    clf = linear_model.LogisticRegression(penalty='l1', C=1.0)

    results = []

    print "Training the expert"
    expert_x = data.test.bow.tocsr()
    expert_y = data.test.target
    expert.fit(expert_x, expert_y)

    print "Training student"
    bootstrap = rand.permutation(len(data.train.data))[:50]

    train_x = data.train.bow.tocsr()[bootstrap]
    train_y = data.train.target[bootstrap]

    clf.fit(train_x, train_y)
    print "*"*60
    for x in bootstrap:
        print data.train.data[x]
    print "*"*60
    predictions = clf.predict_proba(train_x)
    uncertainty = predictions.min(axis=1)
    exp_pred = expert.predict_proba(train_x)

    sorted_ind = np.argsort(uncertainty)[::-1]

    for p,e in zip(predictions[sorted_ind],exp_pred[sorted_ind]):
        print ("{0} \t {1} \t {2}".format(p[0],p[1], np.min(p)))

    print("Elapsed time: %.4f" % (time() - t0))
