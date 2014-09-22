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


from experiment.experiment_utils import split_data_sentences, parse_parameters_mat, clean_html, set_cost_model
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

#############  COMMAND LINE PARAMETERS ##################
ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawTextHelpFormatter)
ap.add_argument('--train',
                metavar='TRAIN',
                default="20news",
                help='training data (libSVM format)')

ap.add_argument('--neutral-threshold',
                metavar='NEUTRAL',
                type=float,
                default=.4,
                help='neutrality threshold of uncertainty')

ap.add_argument('--expert-penalty',
                metavar='EXPERT_PENALTY',
                type=float,
                default=1,
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

def get_data(train, cats, fixk, min_size, vct, raw):
    min_size = 10

    args.fixk = None

    data, vct = load_from_file(train, cats, fixk, min_size, vct, raw=raw)

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

    labels, sent_train = split_data_sentences(expert_data.oracle.train, sent_detector, vct)

    expert_data.oracle.train.data = sent_train
    expert_data.oracle.train.target = np.array(labels)
    expert_data.oracle.train.bow = vct.transform(expert_data.oracle.train.data)

    exp_clf = linear_model.LogisticRegression(penalty='l1', C=args.expert_penalty)
    exp_clf.fit(expert_data.oracle.train.bow, expert_data.oracle.train.target)

    #### EXPERT CLASSIFIER: SENTENCES
    print("Training sentence expert")
    labels, sent_train = split_data_sentences(expert_data.sentence.train, sent_detector, vct)

    expert_data.sentence.train.data = sent_train
    expert_data.sentence.train.target = np.array(labels)
    expert_data.sentence.train.bow = vct.transform(expert_data.sentence.train.data)

    sent_clf = None
    # if args.cheating:
    sent_clf = linear_model.LogisticRegression(penalty='l1', C=args.expert_penalty)
    sent_clf.fit(expert_data.sentence.train.bow, expert_data.sentence.train.target)

    return exp_clf, data, vct, cost_model, sent_clf

####################### MAIN ####################
def get_sentences_by_method(pool, student, test_sent):
    test_sent = []

    list_pool = list(pool.remaining)
    # indices = rand.permutation(len(pool.remaining))
    # remaining = [list_pool[index] for index in indices]
    target_sent = []
    for i in list_pool:
        _, sent_bow, sent_txt = student.x_utility(pool.data[i], pool.text[i])
        if isinstance(test_sent, list):
            test_sent = sent_bow
        else:
            test_sent = vstack([test_sent, sent_bow], format='csr')
        target_sent.append(pool.target[i])
    return test_sent, target_sent

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
    accuracies = defaultdict(lambda: [])

    aucs = defaultdict(lambda: [])

    x_axis = defaultdict(lambda: [])

    # vct = CountVectorizer(encoding='ISO-8859-1', min_df=5, max_df=1.0, binary=True, ngram_range=(1, 3),
    #                       token_pattern='\\b\\w+\\b')#, tokenizer=StemTokenizer())
    vct = TfidfVectorizer(encoding='ISO-8859-1', min_df=1, max_df=1.0, binary=False, ngram_range=(1, 2),
                          token_pattern='\\b\\w+\\b')  #, tokenizer=StemTokenizer())
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

    # data, vct = load_from_file(args.train, [categories[3]], args.fixk, min_size, vct, raw=True)
    #
    # print("Data %s" % args.train)
    # print("Data size %s" % len(data.train.data))
    #
    # parameters = parse_parameters_mat(args.cost_model)
    #
    # print "Cost Parameters %s" % parameters
    #
    # cost_model = set_cost_model(args.cost_function, parameters=parameters)
    # print "\nCost Model: %s" % cost_model.__class__.__name__
    #
    # #### STUDENT CLASSIFIER
    # clf = linear_model.LogisticRegression(penalty="l1", C=1)
    # # clf = set_classifier(args.classifier)
    # print "\nStudent Classifier: %s" % clf
    #
    # #### EXPERT CLASSIFIER
    #
    # exp_clf = linear_model.LogisticRegression(penalty='l1', C=args.expert_penalty)
    # exp_clf.fit(data.test.bow, data.test.target)
    # expert = baseexpert.NeutralityExpert(exp_clf, threshold=args.neutral_threshold,
    #                                      cost_function=cost_model.cost_function)

    exp_clf, data, vct, cost_model, sent_clf = get_data(args.train, [categories[0]], args.fixk, min_size, vct, raw=True)  # expert: classifier, data contains train and test
    print "\nExpert: %s " % exp_clf

    #### ACTIVE LEARNING SETTINGS
    step_size = args.step_size
    bootstrap_size = args.bootstrap
    evaluation_points = 200

    print ("Sentences scoring")
    t0 = time.time()
    ### experiment starts
    clf = linear_model.LogisticRegression(penalty='l1', C=args.expert_penalty)

    # student = structured.AALStructured(model=clf, accuracy_model=None, budget=args.budget, seed=args.seed, vcn=vct,
    #                                    subpool=250, cost_model=cost_model)
    # student.set_score_model(exp_clf)

    student = structured.AALStructuredReading(model=clf, accuracy_model=None, budget=args.budget, seed=args.seed, vcn=vct,
                                              subpool=250, cost_model=cost_model)
    student.set_score_model(exp_clf)  # expert model
    student.set_sentence_model(sent_clf)  # expert sentence model


    # fixk = structured.AALStructuredFixk(model=clf, accuracy_model=None, budget=args.budget, seed=args.seed, vcn=vct,
    #                                     subpool=250, cost_model=cost_model)
    # fixk.set_score_model(exp_clf)

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

    # print sentences_average(pool, vct)
    import itertools

    fns = [student.score_fk, student.score_max, student.score_rnd, student.score_max_feat, student.score_max_sim]

    if True:
        ## create data for testing method
        # select the first sentence always
        print args.train
        print "Testing size: %s" % len(pool.target)
        print "Class distribution: %s" % (1. * pool.target.sum() / len(pool.target))

        student.fn_utility = student.utility_one
        for fn in fns:
            ## firstk
            test_sent = []
            student.score = fn
            test_sent, target_sent = get_sentences_by_method(pool, student, test_sent)
            predict = exp_clf.predict(test_sent)

            accu = metrics.accuracy_score(pool.target, predict)
            print "Accu %s \t%s" % (student.score.__name__, accu)
        # print "Targets %s" % np.unique(target_sent)
        # print "Targets %s" % np.unique(pool.target)
        # print

        test_sent, target_sent = score_top_feat(pool, sent_detector, exp_clf, vct)
        predict = exp_clf.predict(test_sent)

        accu = metrics.accuracy_score(pool.target, predict)
        print "Accu %s \t%s" % (student.score.__name__, accu)


    if False:
        random.seed(args.seed)
        rnd_set = random.sample(zip(pool.remaining, pool.target), 100)

        print "*" * 10
        print "random %s documents" % len(rnd_set)
        i = 0

        targets = []
        queries = []

        doctext = pool.text

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


