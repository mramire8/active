__author__ = 'mramire8'
__copyright__ = "Copyright 2014, ML Lab"
__version__ = "0.1"
__status__ = "Research"

import sys
import os

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("../"))
sys.path.append(os.path.abspath("../experiment/"))

from experiment_utils import *
import argparse
import numpy as np
from sklearn.datasets.base import Bunch
from datautil.load_data import load_from_file
from sklearn import linear_model
import time
from sklearn import metrics
from collections import defaultdict
from strategy import structured
from expert import baseexpert
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import random
import nltk
import re
from scipy.sparse import diags
#############  COMMAND LINE PARAMETERS ##################
ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawTextHelpFormatter)
ap.add_argument('--train',
                metavar='TRAIN',
                default="imdb",
                help='training data (libSVM format)')

ap.add_argument('--neutral-threshold',
                metavar='NEUTRAL',
                type=float,
                default=.4,
                help='neutrality threshold of uncertainty')

ap.add_argument('--expert-penalty',
                metavar='EXPERT_PENALTY',
                type=float,
                default=0.1,
                help='Expert penalty value for the classifier simulation')

ap.add_argument('--trials',
                metavar='TRIALS',
                type=int,
                default=5,
                help='number of trials')

ap.add_argument('--folds',
                metavar='FOLDS',
                type=int,
                default=1,
                help='number of folds')

ap.add_argument('--budget',
                metavar='BUDGET',
                type=int,
                default=200,
                help='budget')

ap.add_argument('--step-size',
                metavar='STEP_SIZE',
                type=int,
                default=10,
                help='instances to acquire at every iteration')

ap.add_argument('--bootstrap',
                metavar='BOOTSTRAP',
                type=int,
                default=50,
                help='size of the initial labeled dataset')

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

ap.add_argument('--maxiter',
                metavar='MAXITER',
                type=int,
                default=200,
                help='Max number of iterations')

ap.add_argument('--seed',
                metavar='SEED',
                type=int,
                default=876543210,
                help='Max number of iterations')


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


####################### MAIN ####################
def clean_html(data):
    sent_train = []
    print ("Cleaning text ... ")
    for text in data:
        doc = text.replace("<br>", ". ")
        doc = doc.replace("<br />", ". ")
        doc = re.sub(r"\.", ". ", doc)
        # doc = re.sub(r"x*\.x*", ". ", doc)
        sent_train.extend([doc])

    return sent_train


def split_data_sentences(data, sent_detector, vct=CountVectorizer()):
    sent_train = []
    labels = []
    tokenizer = vct.build_tokenizer()

    print ("Spliting into sentences...")
    ## Convert the documents into sentences: train
    for t, sentences in zip(data.target, sent_detector.batch_tokenize(data.data)):
        sents = [s for s in sentences if len(tokenizer(s)) > 1]
        sent_train.extend(sents)  # at the sentences separately as individual documents
        labels.extend([t] * len(sents))  # Give the label of the document to all its sentences
    return labels, sent_train


def main():
    accuracies = defaultdict(lambda: [])

    aucs = defaultdict(lambda: [])

    x_axis = defaultdict(lambda: [])

    vct = CountVectorizer(encoding='ISO-8859-1', min_df=1, max_df=1.0, binary=True, ngram_range=(1, 1),
                          token_pattern='\\b\\w+\\b')#, tokenizer=StemTokenizer())

    vct = TfidfVectorizer(encoding='ISO-8859-1', min_df=1, max_df=1.0, binary=False, ngram_range=(1, 1),
                          token_pattern='\\b\\w+\\b')#, tokenizer=StemTokenizer())


    vct_analizer = vct.build_tokenizer()

    print("Start loading ...")
    # data fields: data, bow, file_names, target_names, target

    ########## NEWS GROUPS ###############
    # easy to hard. see "Less is More" paper: http://axon.cs.byu.edu/~martinez/classes/678/Presentations/Clawson.pdf
    categories = [['alt.atheism', 'talk.religion.misc'],
                  ['comp.graphics', 'comp.windows.x'],
                  ['comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware'],
                  ['rec.sport.baseball', 'sci.crypt']]

    min_size = 10 # max(10, args.fixk)

    # if args.fixk < 0:
    args.fixk = None

    data, vct = load_from_file(args.train, [categories[3]], args.fixk, min_size, vct, raw=True)

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

    labels, sent_train = split_data_sentences(data.train, sent_detector)

    data.train.data = sent_train
    data.train.target = np.array(labels)

    labels, sent_train = split_data_sentences(data.test, sent_detector)
    data.test.data = sent_train
    data.test.target = np.array(labels)

    print("Train:{}, Test:{}, {}".format(len(data.train.data), len(data.test.data), data.test.target.shape[0]))
    ## Get the features of the sentence dataset
    data.train.bow = vct.fit_transform(data.train.data)
    data.test.bow = vct.transform(data.test.data)


    #### EXPERT CLASSIFIER

    exp_clf = linear_model.LogisticRegression(penalty='l1', C=args.expert_penalty)
    exp_clf.fit(data.test.bow, data.test.target)
    expert = baseexpert.NeutralityExpert(exp_clf, threshold=args.neutral_threshold,
                                         cost_function=cost_model.cost_function)
    print "\nExpert: %s " % expert


    #### STUDENT CLASSIFIER
    clf = linear_model.LogisticRegression(penalty="l1", C=1)
    # clf = set_classifier(args.classifier)

    student = structured.AALStructured(model=clf, accuracy_model=None, budget=args.budget, seed=args.seed, vcn=vct,
                                       subpool=250, cost_model=cost_model)
    student.set_score_model(exp_clf)



    print "\nStudent Classifier: %s" % clf


    #### ACTIVE LEARNING SETTINGS
    step_size = args.step_size
    bootstrap_size = args.bootstrap
    evaluation_points = 200

    print ("Sentence Classification")
    t0 = time.time()
    tac = []
    tau = []

    # predition = exp_clf.predict(data.train.bow)

    predictions = exp_clf.predict_proba(data.train.bow)
    unc = np.min(predictions, axis=1)

    coef = exp_clf.coef_[0]
    dc = diags(coef, 0)
    # ind = np.argsort(coef)
    fn = np.array(vct.get_feature_names())
    # print fn[ind[:10]]
    # print fn[ind[-10:]]
    print "Features:%s " % len(fn)
    auc = metrics.roc_auc_score(data.train.target, predictions[:, 1])

    pred_y = exp_clf.classes_[np.argmax(predictions, axis=1)]

    accu = metrics.accuracy_score(data.train.target, pred_y)

    most_post = np.argsort(predictions[:, 0])

    print()
    print "\n".join('%s/%.2f' % (fn[j], coef[j]) for j in np.argsort(coef)[::-1] if coef[j] != 0)
    print"*"*80
    print("AUC:{}, Accu:{}".format(auc, accu))
    print ("Size of predictions {} - {}".format(most_post.shape[0], predictions.shape[0]))



    print "*"*50
    print "Positive"
    for d in most_post[:10]:
        print d,
        print predictions[d],
        print data.train.target[d],
        print data.train.data[d]

        mm = data.train.bow[d] * dc  # sentences feature vectors \times diagonal of coeficients. sentences by features
        print "\n".join("%.3f / %s" % (v, n) for v, f, n in zip(mm.A[0], data.train.bow[d].A[0,:], fn) if f > 0)
        print "-"*20

    print "*"*50
    print "Negative"
    for d in most_post[-10:]:
        print d,
        print predictions[d],
        print data.train.target[d],
        print data.train.data[d]
        mm = data.train.bow[d] * dc  # sentences feature vectors \times diagonal of coeficients. sentences by features
        # print mm[mm > 0]

        print "\n".join("%.3f / %s" % (v, n) for v, f, n in zip(mm.A[0], data.train.bow[d].A[0,:], fn) if f > 0)
        print "-"*20

    print "*"*50
    print "Middle"
    m = len(most_post) / 2
    for d in most_post[m-50:m+50]:
        print d,
        print predictions[d],
        print data.train.target[d],
        print data.train.data[d]
        mm = data.train.bow[d] * dc  # sentences feature vectors \times diagonal of coeficients. sentences by features
        # print mm[mm > 0]

        print "\n".join("%.3f / %s" % (v, n) for v, f, n in zip(mm.A[0], data.train.bow[d].A[0,:], fn) if f > 0)
        print "-"*20



    print("Elapsed time %.3f" % (time.time() - t0))


def neutral_label(label):
    if label is None:
        return 0
    else:
        return 1


def format_query(query_labels):
    string = ""
    for l, q in query_labels:
        string = string + "{0}".format(l)
        for qi in q:
            string = string + "\t{0:.2f} ".format(qi)
        string = string + "\n"
    return string

def main2():
    # load paramters
    # load data
    # preprocess data
    # set student
    # set expert
    # start loop
    pass


if __name__ == '__main__':
    main()


