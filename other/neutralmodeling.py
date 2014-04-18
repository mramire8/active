__author__ = 'mramire8'
__copyright__ = "Copyright 2013, ML Lab"
__version__ = "0.1"
__status__ = "Development"

import sys
import os

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("../aal_misc/samples/neutral"))

import argparse
import numpy as np
from sklearn import metrics
from sklearn import linear_model

# from neutral import *
from datautil.textutils import StemTokenizer
from datautil.load_data import *
import auc_util

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawTextHelpFormatter)
ap.add_argument('--train',
                metavar='TRAIN',
                default="aviation",
                help='training data (libSVM format)')

ap.add_argument('--seed',
                metavar='SEED',
                type=int,
                default=1234567,
                help='random seed')

ap.add_argument('--reg',
                metavar='REG',
                type=float,
                default=.1,
                help='random seed')

ap.add_argument('--neutral',
                metavar='NEUTRAL',
                # default="C:/Users/mramire8/Documents/code/python/accassumption/data/movie-neutraldata.txt",
                # default="../../aal_misc/accassumption/data/movie-neutraldata.txt",
                default="../../aal_misc/accassumption/data/aviation_neutral.txt",
                help='neutral training data text')

args = ap.parse_args()
rand = np.random.mtrand.RandomState(args.seed)

print args
print

########## NEWS GROUPS ###############
def get_dictionary(ratio, docfreq, term_head, all_terms):
    RATIO = term_head.index('RATIO')
    DF = term_head.index('DOCFREQ')
    FEATURE = term_head.index('FEATURE')
    ALPHA1 = term_head.index('ALPHA1')

    # dc_pos = [t[FEATURE] for t in all_terms if float(t[RATIO]) >= ratio and int(t[DF]) >= docfreq and float(t[ALPHA1]) < 0]
    # dc_neg = [t[FEATURE] for t in all_terms if float(t[RATIO]) >= ratio and int(t[DF]) >= docfreq and float(t[ALPHA1]) >= 0]
    return [t[FEATURE] for t in all_terms]
    # return dc_pos, dc_neg


def print_features(coef, names):
    """ Print sorted list of non-zero features/weights. """
    ### coef = clf.coef_[0]
    ### names = vec.get_feature_names()
    print("Number of Features: %s" % len(names))
    print "\n".join('%s\t%.2f' % (names[j], coef[j]) for j in np.argsort(coef)[::-1] if coef[j] != 0)
    print "Done"


def load_neutral(docs, feat_names, vct):
    """
     Returns a dataset of the docs based on a computation of features using the relevance dictionary
     @rtype : array, array
     @param  docs: dataset of documents
     @param  relevance: relevance dictionary word->value
     @return: train_x feature vector of target values \n
             train_y target values
     """

    SEEN_WORDS = feat_names.index("seenwords")
    CLASS = feat_names.index("NEUTRALITY-CLASS")
    TEXT = feat_names.index("text")
    VOTES = feat_names.index("votes")

    train_y = []
    final = []
    text = []
    target_names = ['neutral', 'nonneutral']
    for i_doc in docs:
        if int(i_doc[VOTES]) > 1:
            seen_words = int(i_doc[SEEN_WORDS])
            d = i_doc[TEXT].lower().strip().split(" ")
            seen_doc = d[0:seen_words]
            seen_doc = " ".join(seen_doc)
            text.append(seen_doc)
            y = target_names.index(str(i_doc[CLASS].lower().strip()))
            train_y.append(y)
    final = vct.transform(text)
    return final, np.array(train_y), target_names


# vct = CountVectorizer(encoding='latin1', min_df=1, max_df=1.0, binary=True, ngram_range=(1, 3),
#                       token_pattern=r'\b\w+\b', tokenizer=StemTokenizer(), vocabulary=dict_all)
vct = CountVectorizer(encoding='latin1', min_df=1, max_df=1.0, binary=True, ngram_range=(1, 3),
                      token_pattern=r'\b\w+\b')#, tokenizer=StemTokenizer())

if (__name__ == '__main__'):
    rand = np.random.mtrand.RandomState(8675309)

    # easy to hard. see "Less is More" paper: http://axon.cs.byu.edu/~martinez/classes/678/Presentations/Clawson.pdf
    categories = [['alt.atheism', 'talk.religion.misc'],
                  ['comp.graphics', 'comp.windows.x'],
                  ['comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware'],
                  ['rec.sport.baseball', 'sci.crypt']]
    data = load_dataset(args.train, None, categories[0], vct, None, raw=True)

    train_x = data.test.data
    train_x = vct.fit_transform(train_x)

    print("Data %s" % args.train)

    print("Data size %s" % len(data.train.data))

    # Smaller C means fewer features selected.
    docs, headers = load_documents(args.neutral)

    neu_x, neu_y, target_names = load_neutral(docs, headers, vct)
    print("Neutral data:%s" % (neu_x.shape[0]))
    print
    print "*" * 60
    auc_results = []
    penalties = np.arange(.01, .1, .01)
    penalties = np.append(penalties, np.arange(.1,.9,.025))
    for reg in penalties:
    # for reg in [args.reg]:

        clf = linear_model.LogisticRegression(penalty='l1', C=reg)
        clf.fit(train_x, data.test.target)
        # print(clf)

        prob_y = clf.predict_proba(neu_x)
        pred_y = clf.classes_[np.argmax(prob_y, axis=1)]
        unc = prob_y[np.argmin(prob_y, axis=1), 1]
        unc = np.array([np.min(x) for x in prob_y])
        ordered_x_ind = np.argsort(neu_y)

        ord_x = neu_x[ordered_x_ind]
        ord_y = neu_y[ordered_x_ind]

        unc_ord = np.argsort(unc)[::-1]  # order with highest uncertainty on top
        print "Penalty:\t {0:.3f} \t AUC: {1:.4f}".format(reg, auc_util.auc_uncertainty(neu_y, prob_y))
        auc_results.append("Penalty:\t {0:.3f} \t AUC: {1:.4f}".format(reg, auc_util.auc_uncertainty(neu_y, prob_y)))
        # n = 0
        # counts = [n for x in pred_y if x == 0]

        np.set_printoptions(precision=4)
        if False:
            print("*"*20 + " PROB. ESTIMATIONS " + "*"*20)
            print "ID\tTrueLabel\tProb_0\t1-maxprob"
            i = 0
            for x, y in zip(neu_y, prob_y):
                print "%d\t%s \t %s \t %.4f" % (i, x, y, 1 - y[np.argmax(y)])
                i += 1

        if True:
            print("*"*20 + " UNC ORDERED PREDICTIONS " + "*"*20)
            print("ID\tTrueNeutral\tUncertainty")
            for a, x, y in zip(unc_ord, neu_y[unc_ord], unc[unc_ord]):
                print ("%s\t%s\t%.4f" % (a, x, y))

    print
    print("*"*20 + " AUC NEUTRAL SUMMARY " + "*"*20)
    for x in auc_results:
        print x