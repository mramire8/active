__author__ = 'mramire8'
__author__ = 'maru'
__copyright__ = "Copyright 2013, ML Lab"
__version__ = "0.1"
__status__ = "Development"

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

from collections import defaultdict
from strategy import structured
from expert import baseexpert
from sklearn.feature_extraction.text import CountVectorizer
import random
import nltk
#############  COMMAND LINE PARAMETERS ##################
ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawTextHelpFormatter)
ap.add_argument('--train',
                metavar='TRAIN',
                default="pan",
                help='training data (libSVM format)')

ap.add_argument('--neutral-threshold',
                metavar='NEUTRAL',
                type=float,
                default=.4,
                help='neutrality threshold of uncertainty')

ap.add_argument('--expert-penalty',
                metavar='EXPERT_PENALTY',
                type=float,
                default=0.3,
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

ap.add_argument('--fixk',
                metavar='FIXK',
                type=int,
                default=10,
                help='fixed k number of words')

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

ap.add_argument('--lambda-value',
                metavar='LAMBDA_VALUE',
                type=float,
                default=1.0,
                help='tradeoff paramters for the objective function ')

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


####################### MAIN ####################
def main():
    accuracies = defaultdict(lambda: [])

    aucs = defaultdict(lambda: [])

    x_axis = defaultdict(lambda: [])

    vct = CountVectorizer(encoding='ISO-8859-1', min_df=5, max_df=1.0, binary=True, ngram_range=(3, 3),
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

    min_size = max(10, args.fixk)

    if args.fixk < 0:
        args.fixk = None

    data, vct = load_from_file(args.train, [categories[3]], args.fixk, min_size, vct, raw=True)

    print("Data %s" % args.train)
    print("Data size %s" % len(data.train.data))

    parameters = parse_parameters_mat(args.cost_model)

    print "Cost Parameters %s" % parameters

    cost_model = set_cost_model(args.cost_function, parameters=parameters)
    print "\nCost Model: %s" % cost_model.__class__.__name__

    #### STUDENT CLASSIFIER
    clf = linear_model.LogisticRegression(penalty="l1", C=1)
    # clf = set_classifier(args.classifier)
    print "\nStudent Classifier: %s" % clf

    #### EXPERT CLASSIFIER

    exp_clf = linear_model.LogisticRegression(penalty='l1', C=args.expert_penalty)
    exp_clf.fit(data.test.bow, data.test.target)
    expert = baseexpert.NeutralityExpert(exp_clf, threshold=args.neutral_threshold,
                                         cost_function=cost_model.cost_function)
    print "\nExpert: %s " % expert

    #### ACTIVE LEARNING SETTINGS
    step_size = args.step_size
    bootstrap_size = args.bootstrap
    evaluation_points = 200

    print ("Sentences scoring")
    t0 = time.time()
    tac = []
    tau = []
    ### experiment starts

    student = structured.AALStructured(model=clf, accuracy_model=None, budget=args.budget, seed=args.seed, vcn=vct,
                                       subpool=250, cost_model=cost_model)
    student.set_score_model(exp_clf)

    fixk = structured.AALStructuredFixk(model=clf, accuracy_model=None, budget=args.budget, seed=args.seed, vcn=vct,
                                        subpool=250, cost_model=cost_model)
    fixk.set_score_model(exp_clf)

    coef = exp_clf.coef_[0]
    feats = vct.get_feature_names()
    print "*" * 60
    # print_features(coef, feats)
    print "*" * 60

    pool = Bunch()
    pool.data = data.train.bow.tocsr()   # full words, for training
    pool.text = data.train.data
    pool.target = data.train.target
    pool.predicted = []
    pool.remaining = set(range(pool.data.shape[0]))  # indices of the pool

    sum_sent = 0
    average_words = 0.

    print sentences_average(pool, vct)

    if False:
        random.seed(args.seed)
        rnd_set = random.sample(zip(pool.remaining, pool.target), 100)

        print "*" * 10
        print "random %s documents" % len(rnd_set)
        i = 0

        targets = []
        queries = []

        doctext = pool.text
        # ## examples
        # doctext = ['this is a great movie',
        #     'I hate the acting',
        #     'this goes straight to dvd',
        #     'I wasted my time',
        #     'just horrible',
        #     'this is a classic',
        #     'this is the perfect movie for the weekend',
        #     'I love this movie']
        # lbl_example = [1,0,0,0,0,1,1,1]
        # rnd_set = zip(range(8),lbl_example)
        ## end example

        print "ID\tLABEL\tTOTSENTS\tTKSCORE\tMAXKSCORE\tFK\tLK\tTOPK"
        for docid, label in rnd_set:
            doc = doctext[docid].replace("<br>", ". ")
            doc = doc.replace("<br />", ". ")
            doc = doc.encode('latin-1')
            n = len(sent_detector.tokenize(doc))

            fksents = fixk.getk(doc)                 # first k sentences
            lksents = fixk.getlastk(doc)             # last k sentences
            # sents, sentscores = student.getk(doc)    # top k sum sentences
            sents2, sentscores2 = student.getkmax(doc) # top k max sentences

            # sents2, sentscores2 = student.getk3(doc)  # top k max - min
            # queries.append(sents[0][1])
            # targets.extend([label])

            # print "{0}\t{1}\t{7}\t{2}\t{3}\tfk:{4}\tb1:{5}\tb2:{6}\tall:{8}".format(i, label, sentscores[0], sentscores2[0], fksents[0][1],
            #     lksents[0][1], sents[0][1], n, doc)
            print "{0}\t{1}\t{7}\t{2}\t{3}\tfk:{4}\tb1:{5}\tb2:{6}\tall:{8}".format(i, label, sentscores2[0], sentscores2[0], fksents[0][1],
                lksents[0][1], sents2[0][1], n, doc)
            i += 1


        # print("If the expert sees sentences...")
        # queries_ft = vct.transform(queries)
        # print queries_ft.shape
        # expert_labels = expert.label_instances(queries_ft, np.array(targets))

        # print("Percentage of Neutrals: %s" % (expert_labels.count(None) / float(len(rnd_set))))
        # print("Percentage of 0: %s" % (expert_labels.count(0) / float(len(rnd_set))))
        # print("Percentage of 1: %s" % (expert_labels.count(1) / float(len(rnd_set))))

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


if __name__ == '__main__':
    main()


