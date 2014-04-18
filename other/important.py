__author__ = 'mramire8'
import sys
import os

sys.path.append(os.path.abspath("."))

import argparse
from sklearn.naive_bayes import MultinomialNB

from datautil.textutils import StemTokenizer
from datautil.load_data import *
import ast

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawTextHelpFormatter)
ap.add_argument('--train',
                metavar='TRAIN',
                default="imdb",
                help='training data (libSVM format)')


ap.add_argument('--seed',
                metavar='SEED',
                type=int,
                default=1234567,
                help='random seed')

ap.add_argument('--ngrams',
                metavar='NGRAMS',
                type=str,
                default="(2,2)",
                help='random seed')

ap.add_argument('--stemm',
                action='store_true',
                help='stemming the data or not')

args = ap.parse_args()
rand = np.random.mtrand.RandomState(args.seed)

print args
print


def main():
    print "n-grams: %s" % args.ngrams
    # print "Stemming: %" % args.stem

    ngrams = ast.literal_eval(args.ngrams)
    vct = CountVectorizer(encoding='ISO-8859-1', min_df=5, max_df=1.0, binary=False, ngram_range=ngrams,
                          token_pattern='\\b\\w+\\b')
    if args.stemm:
        vct = CountVectorizer(encoding='ISO-8859-1', min_df=5, max_df=1.0, binary=False, ngram_range=ngrams,
                              token_pattern='\\b\\w+\\b', tokenizer=StemTokenizer())

    print vct

    # vct_analizer = vct.build_analyzer()

    print("Start loading ...")
    t0 = time()

    # data fields: data, bow, file_names, target_names, target

    ########## NEWS GROUPS ###############
    # easy to hard. see "Less is More" paper: http://axon.cs.byu.edu/~martinez/classes/678/Presentations/Clawson.pdf
    categories = [['alt.atheism', 'talk.religion.misc'],
                  ['comp.graphics', 'comp.windows.x'],
                  ['comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware'],
                  ['rec.sport.baseball', 'sci.crypt']]


    data = load_dataset(args.train, None, categories[0], vct, None)

    print("Data %s" % args.train)
    print("Data size %s" % len(data.train.data))


    #feature_counts = np.ones(x_train.shape[0]) * x_train
    #feature_frequencies = feature_counts / np.sum(feature_counts)
    #alpha = feature_frequencies

    train_x = data.train.bow.tocsr()
    train_y = data.train.target
    feature_names = np.asarray(vct.get_feature_names())
    doc_frequency = np.diff(train_x.tocsc().indptr)

    clf1 = MultinomialNB(alpha=1)
    clf10 = MultinomialNB(alpha=10)
    clf100 = MultinomialNB(alpha=100)

    clf1.fit(train_x, train_y)
    clf10.fit(train_x, train_y)
    clf100.fit(train_x, train_y)

    alpha1_feature_log_ratios = clf1.feature_log_prob_[0] - clf1.feature_log_prob_[1]
    alpha10_feature_log_ratios = clf10.feature_log_prob_[0] - clf10.feature_log_prob_[1]
    alpha100_feature_log_ratios = clf100.feature_log_prob_[0] - clf100.feature_log_prob_[1]
    # print("Class 0 : %s" % target_names[0])

    print "i\tFEATURE\tDOCFREQ\tALPHA1\tALPHA10\tALPHA100"
    for i in xrange(len(feature_names)):
        print "%d\t%s\t%d\t%0.3f\t%0.3f\t%0.3f" % (i, feature_names[i].encode('utf-8'), doc_frequency[i], alpha1_feature_log_ratios[i],
                                                   alpha10_feature_log_ratios[i], alpha100_feature_log_ratios[i])

    print "Elapsed time: %s" % (time() - t0)


if __name__ == '__main__':
    main()

