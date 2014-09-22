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
import sklearn.metrics as metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt


apfk = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawTextHelpFormatter)
apfk.add_argument('--train',
                metavar='TRAIN',
                default="imdb",
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
import itertools
# vct = CountVectorizer(encoding='latin1', min_df=5, max_df=1.0, binary=True, ngram_range=(1, 3),
#                       token_pattern=r'\b\w+\b', tokenizer=StemTokenizer())

vct = TfidfVectorizer(encoding='ISO-8859-1', min_df=1, max_df=1.0, binary=False, ngram_range=(1, 2),
                      token_pattern='\\b\\w+\\b')  #, tokenizer=StemTokenizer())

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
    categories = [['student','faculty']]
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
                         #name, fixk, categories, vct, min_size, raw=False, percent=.5
        data = load_dataset(args.train, 10, categories[0], vct, 100, percent=.5)
        fixk_file = open(fixk_saved, "wb")
        pickle.dump(data, fixk_file)
        fixk_file.close()
        vectorizer = open("{0}vectorizer.p".format(args.train), "wb")
        pickle.dump(vct, vectorizer)
        vectorizer.close()

    # data, vct = load_from_file(args.train, categories, args.fixk, 100, vct)
    print("Data %s" % args.train)
    print("Data size %s" % len(data.train.data))
    print("Number of features: %s" % len(vct.get_feature_names()))


    # expert = linear_model.LogisticRegression(penalty='l1', C=0.3)
    clf = linear_model.LogisticRegression(penalty='l1', C=1)

    clf.fit(data.test.bow, data.test.target)

    predictions = clf.predict_proba(data.train.bow)
    predict = clf.predict(data.train.bow)
    print "Accuracy:", metrics.accuracy_score(data.train.target, predict)

    unc = np.min(predictions, axis=1)
    idn=unc > 0.3
    # print len(idn)
    # print
    # print float(np.sum(1-data.train.target))
    # print np.sum(idn[data.train.target == 0])/float(np.sum(1-data.train.target))
    # print
    # print float(np.sum(data.train.target))
    # print np.sum(idn[data.train.target ==vct.get_feature_names() 1])/float(np.sum(data.train.target))
    # print
    # print np.sum(idn)

    coef = clf.coef_[0]
    ind = np.argsort(coef)
    fn = np.array(vct.get_feature_names())
    print fn[ind[:10]]
    print fn[ind[-10:]]
    markers = itertools.cycle(['o','s','v'])
    # print data.train.data[0]
    print "Training student"

    sizes = [50, 100, 150, 200, 250,300, 500, 1000]
    y = []
    x = []
    print "SIZE \t ACCURACY"
    for s in sizes:
        bootstrap = rand.permutation(len(data.train.data))
        indices = bootstrap[:s]

        train_x = data.train.bow.tocsr()[indices[:s]]
        train_y = data.train.target[indices[:s]]

        clf.fit(train_x, train_y)

        predictions = clf.predict(data.test.bow.tocsr())
        scores = metrics.accuracy_score(data.test.target,predictions)
        ## print clf.__class__.__name__
        print "{0} \t {1}".format(s, scores)
        x.append(s)
        y.append(scores)
    plt.figure()
    plt.title("Data:"+args.train+" Accuracy - Doc train - Doc test")
    plt.plot(x, y, label='Accuracy', linewidth=2, color='red', marker=markers.next())
    plt.show()
    print("Elapsed time: %.4f" % (time() - t0))
