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


args = apfk.parse_args()
rand = np.random.mtrand.RandomState(args.seed)

print args
print

vct = CountVectorizer(encoding='latin1', min_df=5, max_df=1.0, binary=True, ngram_range=(1, 3),
                      token_pattern=r'\b\w+\b', tokenizer=StemTokenizer())

# vct = CountVectorizer(encoding='latin1', min_df=1, max_df=1.0, binary=True, ngram_range=(1, 1),
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

    dataset = load_dataset(args.train, None, categories[0], vct, 100, raw=True,  percent=.5)

    # train on test and test on train (test on k words of the train)


    print("Data %s" % args.train)
    print("Data size %s" % len(dataset.train.data))
    print

    kvalues = [10, 25, 50, 75, 100]
    threshold = np.arange(.3, .46, 0.05)

    count_results = []

    for fixk in kvalues:
    # for reg in [.2]:

        fixk_saved = "{0}{1}.p".format(args.train, fixk)
        try:
            fixk_file = open(fixk_saved, "rb")
            data = pickle.load(fixk_file)
        except IOError:
            data = process_data(dataset, fixk, 100, vct, silent=True)
            fixk_file = open(fixk_saved, "wb")
            pickle.dump(data, fixk_file)

        train_x = data.test.bow
        train_y = data.test.target

        test_x = data.train.bowk
        test_y = data.train.target

        results = []
        print "*"*60
        print
        print("K= %s" % fixk)
        # penalty = np.arange(0.02, 0.1, 0.02)
        # penalty = np.append(penalty,np.arange(.1, 1.1, .2))
        penalty = np.arange(.001, 0.021, 0.003)
        for reg in penalty:
            clf = linear_model.LogisticRegression(penalty='l1', C=reg)
            print("penalty: %s" % reg)
            clf.fit(train_x, data.test.target)

            prob_y = clf.predict_proba(test_x)
            pred_y = clf.classes_[np.argmax(prob_y, axis=1)]
            unc = prob_y.min(axis=1)

            results.append(np.array([np.min(x) for x in prob_y]).mean())

            iteration = [fixk, reg, len(unc), [sum(unc > t) for t in threshold]]
            print "Counts:\t %s" % iteration

            count_results.append(iteration)


        print(results)
        print "\nREG\tK\tCOND.ERR"
        for reg,ce in zip(penalty, results):
            print "{0}\t{1}\t{2:.4f}".format(reg, fixk, ce)

    print
    print ("Thresholds: %s" % threshold)
    print("*"*20 + " SUMMARY " + "*"*20)
    for f,r,l, count in count_results:
        print "{0}\t{1}\t{2}\t{3}\t".format(f, r, l, count)

    print("Elapsed time: %.4f" % (time() - t0))
