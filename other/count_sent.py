__author__ = 'mramire8'
__copyright__ = "Copyright 2013, ML Lab"
__version__ = "0.1"
__status__ = "Development"

import sys
import os

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("../"))
sys.path.append(os.path.abspath("../experiment/"))


from experiment.experiment_utils import split_data_first1sentences, parse_parameters_mat, clean_html, set_cost_model
import argparse
import numpy as np
from sklearn.datasets.base import Bunch
from datautil.load_data import load_from_file, split_data
from sklearn import linear_model
import time

from collections import defaultdict
from strategy import structured
from expert import baseexpert
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import random
import nltk
from scipy.sparse import vstack
from sklearn import metrics
from learner.adaptive_lr import LogisticRegressionAdaptive

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


def get_data(clf, train, cats, fixk, min_size, vct, raw):
    import copy

    min_size = 10

    args.fixk = None

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
    print("\nTraining Oracle expert\n"+"-"*40)
    print("Expert training data in docs: %s" % len(expert_data.oracle.train.data))
    labels, sent_train, _, _, f1 = split_data_first1sentences(expert_data.oracle.train, sent_detector, vct, limit=0)
    print("Expert training data in sentences: %s, %s" % (len(labels), len(sent_train)))
    total = len(labels)
    print("Sentence distribution with not cleaning (y=1): %.4f" % (1.*np.sum(labels)/len(labels)))
    print("Number of first sentences: %s" % len(f1))
    garbage1 = [s for s in f1 if len(s) <= 2]

    print "Garbage in the first sentence: %s" % len(garbage1)

    expert_data.oracle.train.data = sent_train
    expert_data.oracle.train.target = np.array(labels)
    expert_data.oracle.train.bow = vct.transform(expert_data.oracle.train.data)

    # exp_clf = linear_model.LogisticRegression(penalty='l1', C=args.expert_penalty)
    exp_clf = copy.copy(clf)
    exp_clf.fit(expert_data.oracle.train.bow, expert_data.oracle.train.target)


    print("\nRemoving 1 character\n"+"-"*40)
    labels, sent_train, dumped1, dumped_lbl,_ = split_data_first1sentences(expert_data.oracle.train, sent_detector, vct, limit=1)
    print("Expert training data in sentences: %s, %s" % (len(labels), len(sent_train)))
    print("Removed sentences: %s" % (total - len(labels)))
    pred = exp_clf.predict(vct.transform(dumped1))
    print("Dumped sentence distribution with limit=1 (y=1): %.4f" % (1.*np.sum(dumped_lbl)/len(dumped_lbl)))
    print "Distribution of dumped by the oracle (y=1): \t%.4f" % (1.*pred.sum()/len(pred))
    print("\nRemoving 2 Characters\n"+"-"*40)
    labels, sent_train, dumped2, dumped_lbl,_ = split_data_first1sentences(expert_data.oracle.train, sent_detector, vct, limit=2)
    print("Expert training data in sentences: %s, %s" % (len(labels), len(sent_train)))
    print("Removed sentences: %s" % (total - len(labels)))
    print("Dumped sentence distribution with limit=1 (y=1): %.4f" % (1.*np.sum(dumped_lbl)/len(dumped_lbl)))
    pred = exp_clf.predict(vct.transform(dumped2))
    print "Distribution of dumped by the oracle (y=1): \t%.4f" % (1.*pred.sum()/len(pred))



    #### EXPERT CLASSIFIER: SENTENCES
    sent_clf = None
    if False:
        print("Training sentence expert")
        labels, sent_train = split_data_first1sentences(expert_data.sentence.train, sent_detector, vct, limit=0)

        expert_data.sentence.train.data = sent_train
        expert_data.sentence.train.target = np.array(labels)
        expert_data.sentence.train.bow = vct.transform(expert_data.sentence.train.data)

        # if args.cheating:
        sent_clf = copy.copy(clf)
        # sent_clf = linear_model.LogisticRegression(penalty='l1', C=args.expert_penalty)
        sent_clf.fit(expert_data.sentence.train.bow, expert_data.sentence.train.target)

    return exp_clf, data, vct, cost_model, sent_clf


####################### MAIN ####################
def get_sentences_by_method(pool, student, test_sent):
    test_sent = []

    list_pool = list(pool.remaining)
    # indices = rand.permutation(len(pool.remaining))
    # remaining = [list_pool[index] for index in indices]
    target_sent = []
    text_sent = []
    for i in list_pool:
        _, sent_bow, sent_txt = student.x_utility(pool.data[i], pool.text[i])
        if isinstance(test_sent, list):
            test_sent = sent_bow
        else:
            test_sent = vstack([test_sent, sent_bow], format='csr')
        text_sent.append(sent_txt)
        target_sent.append(pool.target[i])
    return test_sent, target_sent, text_sent

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

    exp_clf, data, vct, cost_model, sent_clf = get_data(clf, args.train, [categories[0]], args.fixk, min_size, vct, raw=True)  # expert: classifier, data contains train and test

    print "\nExpert: %s " % exp_clf

    print ("Sentences scoring")
    ### experiment starts

    print("Elapsed time %.3f" % (time.time() - t0))


def print_document(text_sent, offset, method_name='', top=500, **kwargs):
    #text_sent, truth=pool.target, prediction=predict, org_doc=pool.text):
    print "*"*60
    n = len(text_sent)
    labels = list(kwargs.keys())
    labels.append('text_sent')
    print method_name+"\t", "\t".join(labels)
    print "*"*60
    range_docs = range(offset, min(n, offset+top))
    for i in range_docs:
        if kwargs is not None:
            for w in kwargs.values():
                print w[i], "\t",
        print text_sent[i].encode('latin1').replace("\n"," ")



if __name__ == '__main__':
    main()


