__author__ = 'mramire8'
__copyright__ = "Copyright 2013, ML Lab"
__version__ = "0.1"
__status__ = "Development"

import sys
import os

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("../"))
sys.path.append(os.path.abspath("../experiment/"))


from experiment.experiment_utils import parse_parameters_mat, clean_html, set_cost_model
import argparse
import numpy as np
from sklearn.datasets.base import Bunch
from datautil.load_data import load_from_file
from sklearn import linear_model
import time

from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import random
import nltk
from scipy.sparse import vstack


#############  COMMAND LINE PARAMETERS ##################
ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawTextHelpFormatter)
ap.add_argument('--train',
                metavar='TRAIN',
                default="imdb",
                help='training data (libSVM format)')


ap.add_argument('--expert-penalty',
                metavar='EXPERT_PENALTY',
                type=float,
                default=1.0,
                help='Expert penalty value for the classifier simulation')


ap.add_argument('--cost-function',
                metavar='COST_FUNCTION',
                type=str,
                default="uniform",
                help='cost function of the x-axis [uniform|log|linear|direct]')

ap.add_argument('--cost-model',
                metavar='COST_MODEL',
                type=str,
                default="[[10.0,5.7], [25.0,8.2], [50.1,10.9], [75,15.9], [100,16.7], [125,17.8], [150,22.7], [175,19.9], [200,17.4]]",
                help='cost function parameters of the cost function')

ap.add_argument('--fixk',
                metavar='FIXK',
                type=int,
                default=10,
                help='fixed k number of words')


ap.add_argument('--seed',
                metavar='SEED',
                type=int,
                default=876543210,
                help='Max number of iterations')


ap.add_argument('--student',
                metavar='STUDENT',
                # type=float,
                default="bestk",
                help='Anytime student type: [lambda|anyunc|anyzero]')

ap.add_argument('--classifier',
                metavar='CLASSIFIER',
                type=str,
                default="lr",
                help='underlying classifier')

args = ap.parse_args()
rand = np.random.mtrand.RandomState(args.seed)
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

print args
print


def print_features(coef, names):
    """ Print sorted list of non-zero features/weights. """
    print "\n".join('%s/%.2f' % (names[j], coef[j]) for j in np.argsort(coef)[::-1] if coef[j] != 0)

def sentences_average(pool, vct):
   ## COMPUTE: AVERAGE SENTENCES IN DOCUMENTS
    tk = vct.build_tokenizer()
    allwords = 0.
    sum_sent = 0.
    average_words = 0
    min_sent = 10000
    max_sent = 0
    for docid, label in zip(pool.remaining, pool.target):

        doc = pool.text[docid].replace("<br>", ". ")
        doc = doc.replace("<br />", ". ")
        isent = sent_detector.tokenize(doc)
        sum_sent += len(isent)
        min_sent = min(min_sent, len(isent))
        max_sent = max(max_sent, len(isent))
        for s in sent_detector.tokenize(doc):
            average_words += len(tk(s))
            allwords += 1

    print("Average sentences fragments %s" % (sum_sent / len(pool.target)))
    print("Min sentences fragments %s" % min_sent)
    print("Max sentences fragments %s" % max_sent)
    print("Total sentences fragments %s" % sum_sent)
    print("Average size of sentence %s" % (average_words / allwords))


def get_data(train, cats, vct, raw):

    min_size = None

    args.fixk = None
    fixk=None

    data, vct2 = load_from_file(train, cats, fixk, min_size, vct, raw=raw)

    print("Data %s" % args.train)
    print("Data size %s" % len(data.train.data))

    parameters = parse_parameters_mat(args.cost_model)

    print "Cost Parameters %s" % parameters

    cost_model = set_cost_model(args.cost_function, parameters=parameters)
    print "\nCost Model: %s" % cost_model.__class__.__name__

    ### SENTENCE TRANSFORMATION
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

    ## delete <br> to "." to recognize as end of sentence
    data.train.data = clean_html(data.train.data)
    data.test.data = clean_html(data.test.data)

    print("Train:{}, Test:{}, {}".format(len(data.train.data), len(data.test.data), data.test.target.shape[0]))

    ## convert document to matrix
    data.train.bow = vct.fit_transform(data.train.data)
    data.test.bow = vct.transform(data.test.data)
    return data


####################### MAIN ####################
from scipy.sparse import diags


def sentence2values(doc_text, sent_detector, score_model, vcn):
        np.set_printoptions(precision=4)
        sents = sent_detector.tokenize(doc_text)
        sents_feat = vcn.transform(sents)
        coef = score_model.coef_[0]
        dc = diags(coef, 0)

        mm = sents_feat * dc  # sentences feature vectors \times diagonal of coeficients. sentences by features
        return mm, sents, sents_feat


def score_top_feat(pool, sent_detector, score_model, vcn):
    test_sent = []

    list_pool = list(pool.remaining)
    # indices = rand.permutation(len(pool.remaining))
    # remaining = [list_pool[index] for index in indices]
    target_sent = []
    for i in list_pool:
        mm, _, sent_bow = sentence2values(pool.text[i], sent_detector, score_model, vcn)
        max_vals = np.argmax(mm.max(axis=1))

        if isinstance(test_sent, list):
            test_sent = sent_bow[max_vals]
        else:
            test_sent = vstack([test_sent, sent_bow[max_vals]], format='csr')
        target_sent.append(pool.target[i])
    return test_sent, target_sent


def main():
    t0 = time.time()

    # vct = CountVectorizer(encoding='ISO-8859-1', min_df=5, max_df=1.0, binary=True, ngram_range=(1, 3),
    #                       token_pattern='\\b\\w+\\b')#, tokenizer=StemTokenizer())

    vct = TfidfVectorizer(encoding='latin1', min_df=5, max_df=1.0, binary=False, ngram_range=(1, 1),
                      token_pattern='\\b\\w+\\b')  #, tokenizer=StemTokenizer())

    print("Start loading ...")

    ########## NEWS GROUPS ###############
    # easy to hard. see "Less is More" paper: http://axon.cs.byu.edu/~martinez/classes/678/Presentations/Clawson.pdf
    categories = [['alt.atheism', 'talk.religion.misc'],
                  ['comp.graphics', 'comp.windows.x'],
                  ['comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware'],
                  ['rec.sport.baseball', 'sci.crypt']]

    min_size = max(10, args.fixk)

    if args.fixk < 0:
        args.fixk = None

    clf = linear_model.LogisticRegression(penalty='l1', C=args.expert_penalty)

    data = get_data(args.train, [categories[0]], vct, True)  # expert: classifier, data contains train and test

    from collections import Counter
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import brewer2mpl
    import itertools

    colors_n = itertools.cycle(brewer2mpl.get_map('Paired', 'qualitative', 8).mpl_colors)
    print "Sentence distribution"
    ### experiment starts
    X, y, sent_per_doc = split_data_sentences(data, sent_detector, vct, limit=2)

    sent_per_doc = np.array(sent_per_doc)
    sents = Counter(sent_per_doc)

    mpl.style.use('bmh')
    n, bins, patches = plt.hist(sents.keys(), weights=sents.values(),  bins=50)#, color=colors_n.next())
    plt.title("Document Number of Sentence Distribution {0} (mean={1:.2f}) N={2}".format(args.train, sent_per_doc.mean(),
                                                                                         len(sent_per_doc)),
              fontsize=14, fontfamily='Aria')
    plt.xlabel("Number of sentences")
    plt.ylabel("Frequency")
    plt.show()
    for b in bins:
        print b

    print("Elapsed time %.3f" % (time.time() - t0))

def split_data_sentences(data, sent_detector, vct, limit=0):
    sent_train = []
    labels = []
    tokenizer = vct.build_tokenizer()
    size_per_doc = []
    print ("Spliting into sentences... Limit:", limit)
    ## Convert the documents into sentences: train
    for t, sentences in zip(data.train.target, sent_detector.batch_tokenize(data.train.data)):

        if limit is None:
            sents = [s for s in sentences if len(tokenizer(s)) > 1]
        elif limit > 0:
            sents = [s for s in sentences if len(s.strip()) > limit]
            size_per_doc.append(len(sentences))
        elif limit == 0:
            sents = [s for s in sentences]
        sent_train.extend(sents)  # at the sentences separately as individual documents
        labels.extend([t] * len(sents))  # Give the label of the document to all its sentences

    return labels, sent_train, size_per_doc


if __name__ == '__main__':
    main()


