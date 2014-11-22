__author__ = 'mramire8'
__copyright__ = "Copyright 2013, ML Lab"
__version__ = "0.2"
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
from learner.adaptive_lr import LogisticRegressionAdaptive
import matplotlib.pyplot as plt

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
                default=1.0,
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
    from collections import Counter
    tk = vct.build_tokenizer()
    allwords = 0.
    sum_sent=[]
    average_words=[]
    min_sent = 10000
    max_sent = 0

    for docid, label in zip(pool.remaining, pool.target):
    
        doc = pool.text[docid].replace("<br>", ". ")
        doc = doc.replace("<br />", ". ")
        isent = sent_detector.tokenize(doc)
        sents = [s for s in isent if len(s.strip()) > 2]
        sum_sent.append(len(sents))
        for s in sents:
            average_words.append(len(tk(s)))
    counts = Counter(sum_sent)
    words = Counter(average_words)

    print("Average sentences fragments %s" % (np.mean(counts.values())))
    print("Min sentences fragments %s" % min(counts.keys()))
    print("Max sentences fragments %s" % max(counts.keys()))
    print("Total sentences fragments %s" % sum(counts.values()))
    print("Average size of sentence %s" % (np.mean(words.keys())))
    print("Most common: ", counts.most_common(3))
    plt.xlabel("# Sentences")
    plt.ylabel("Frequency")

    n, bins, patches = plt.hist(counts.keys(),weights=counts.values(), bins=10, alpha=0.75)
    plt.title('Distribution (mean=%.2f, N=%d)' % (np.mean(counts.values()), len(pool.target)))

    plt.show()



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

    labels, sent_train = split_data_sentences(expert_data.oracle.train, sent_detector, vct, limit=2)
    print len(sent_train)
    expert_data.oracle.train.data = sent_train
    expert_data.oracle.train.target = np.array(labels)
    expert_data.oracle.train.bow = vct.transform(expert_data.oracle.train.data)
    print expert_data.oracle.train.bow.shape
    # exp_clf = linear_model.LogisticRegression(penalty='l1', C=args.expert_penalty)
    exp_clf = copy.copy(clf)
    exp_clf.fit(expert_data.oracle.train.bow, expert_data.oracle.train.target)

    #### EXPERT CLASSIFIER: SENTENCES
    print("Training sentence expert")
    labels, sent_train = split_data_sentences(expert_data.sentence.train, sent_detector, vct, limit=2)

    expert_data.sentence.train.data = sent_train
    expert_data.sentence.train.target = np.array(labels)
    expert_data.sentence.train.bow = vct.transform(expert_data.sentence.train.data)

    sent_clf = None
    # if args.cheating:
    sent_clf = copy.copy(clf)
    # sent_clf = linear_model.LogisticRegression(penalty='l1', C=args.expert_penalty)
    sent_clf.fit(expert_data.sentence.train.bow, expert_data.sentence.train.target)

    return exp_clf, data, vct, cost_model, sent_clf,  expert_data


####################### MAIN ####################
def get_sentences_by_method(pool, student, test_sent):
    test_sent = []

    list_pool = list(pool.remaining)
    # indices = rand.permutation(len(pool.remaining))
    # remaining = [list_pool[index] for index in indices]
    target_sent = []
    text_sent = []
    all_scores = []
    order_sent=[]
    for i in list_pool:
        scores, sent_bow, sent_txt, order = student.x_utility(pool.data[i], pool.text[i])
        if isinstance(test_sent, list):
            test_sent = sent_bow
        else:
            test_sent = vstack([test_sent, sent_bow], format='csr')
        text_sent.append(sent_txt)
        target_sent.append(pool.target[i])
        order_sent.append(order)
        all_scores.append(scores)
    return test_sent, target_sent, text_sent, all_scores, order_sent


def calibrate_scores(n_scores, bounds=(.5,1)):
    delta = 1.* (bounds[1] - bounds[0]) / (n_scores -1)
    calibrated = (np.ones(n_scores)*bounds[1]) - (np.array(range(n_scores))*delta)
    return calibrated


def reshape_scores(scores, sent_mat):
    sr = []
    i = 0
    for row in sent_mat:
        sc = []
        for col in row:
            sc.append(scores[i])
            i = i+1
        sr.append(sc)
    return np.array(sr)


def get_sentences_by_method_cal(pool, student, test_sent):
    test_sent = []

    list_pool = list(pool.remaining)
    # indices = rand.permutation(len(pool.remaining))
    # remaining = [list_pool[index] for index in indices]
    target_sent = []
    text_sent = []
    all_scores = []
    docs = []
    for i in list_pool:
        utilities, sent_bow, sent_txt = student.x_utility_cal(pool.data[i], pool.text[i])  # utulity for every sentences in document
        all_scores.extend(utilities) ## every score
        docs.append(sent_bow)  ## sentences for each document
        text_sent.append(sent_txt)  ## text sentences for each document
        target_sent.append(pool.target[i])   # target of every document, ground truth

    ## Calibrate scores
    n = len(all_scores)
    all_scores = np.array(all_scores)
    break_point = 2
    order = all_scores.argsort()[::-1] ## descending order
    ## generate scores equivalent to max prob
    a = calibrate_scores(n/break_point, bounds=(.5,1))
    a = np.append(a, calibrate_scores(n - n/break_point, bounds=(1,.5)))

    ## new scores assigned to original sentence order
    new_scores = np.zeros(n)
    new_scores[order] = a
    cal_scores = reshape_scores(new_scores, docs)

    selected_sent = [np.argmax(row) for row in cal_scores] ## get the sentence of the highest score per document
    selected = [docs[i][k] for i, k in enumerate(selected_sent)]  ## get the bow
    selected_score = [np.max(row) for i, row in enumerate(cal_scores)]  ## get the bow
    test_sent = list_to_sparse(selected)

    return test_sent, np.array(selected_score), selected_sent


def list_to_sparse(selected):
    test_sent = []
    for s in selected:
        if isinstance(test_sent, list):
            test_sent = s
        else:
            test_sent = vstack([test_sent, s], format='csr')
    return test_sent


def get_sentences_by_method_cal_scale(pool, student, test_sent, class_sensitive=True):
    from sklearn import preprocessing 
    test_sent = []

    list_pool = list(pool.remaining)
    # indices = rand.permutation(len(pool.remaining))
    # remaining = [list_pool[index] for index in indices]
    target_sent = []
    text_sent = []
    all_scores = []
    all_p0 = []
    docs = []
    for i in list_pool:
        utilities, sent_bow, sent_txt = student.x_utility_cal(pool.data[i], pool.text[i])  # utulity for every sentences in document
        all_scores.extend(utilities) ## every score
        docs.append(sent_bow[::-1])  ## sentences for each document
        text_sent.append(sent_txt[::-1])  ## text sentences for each document
        target_sent.append(pool.target[i])   # target of every document, ground truth
        all_p0.extend([student.sent_model.predict_proba(s)[0][0] for s in sent_bow])
    ## Calibrate scores

    n = len(all_scores)
    if n != len(all_p0):
        raise Exception("Oops there is something wrong! We don't have the same size")

    all_p0 = np.array(all_p0)

    order = all_p0.argsort()[::-1] ## descending order
    ## generate scores equivalent to max prob
    ordered_p0 = all_p0[order]
    # class_sensitive = True
    if class_sensitive:
        c0_scores = preprocessing.scale(ordered_p0[ordered_p0 > .5])
        c1_scores = -1. * preprocessing.scale(ordered_p0[ordered_p0 <= .5])
        a = np.concatenate((c0_scores, c1_scores))
    else:
        a = preprocessing.scale(ordered_p0)

    new_scores = np.zeros(n)
    new_scores[order] = a
    cal_scores = reshape_scores(new_scores, docs)
    p0 = reshape_scores(all_p0, docs)

    selected_sent = [np.argmax(row) for row in cal_scores] ## get the sentence of the highest score per document
    selected = [docs[i][k] for i, k in enumerate(selected_sent)]  ## get the bow
    selected_score = [np.max(row) for i, row in enumerate(cal_scores)]  ## get the bow
    selected_cl = [p0[i][k] for i, k in enumerate(selected_sent)]
    test_sent = list_to_sparse(selected)

    return test_sent, np.array(selected_score), selected_cl, selected_sent

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


def score_distribution(calibrated, pool, sent_data, student, sizes, cheating=False, show=False):

    print "Testing size: %s" % len(pool.target)
    print "Class distribution: %s" % (1. * pool.target.sum() / len(pool.target))
    # # Sentence dataset
    train_sent = sent_data.oracle.train
    import copy
    ## don't care utility of document
    student.fn_utility = student.utility_one
    ## only testing distribution with max score of the student sentence model
    fns = [student.score_max]
    # fns = [student.score_rnd]
    results = defaultdict(lambda: [])
    for size in sizes:

        # train underlying sentence classifier of the student
        if not cheating:
            clf_test = copy.copy(student.sent_model)
            clf_test.fit(train_sent.bow[:size], train_sent.target[:size])
            student.set_sentence_model(clf_test)

        clf_test = student.sent_model

        for fn in fns:
            test_sent = []
            student.score = fn

            ## for every document pick a sentence
            if calibrated == 'zscore':
                test_sent, scores, ori_scores, sel_sent = get_sentences_by_method_cal_scale(pool, student, test_sent)
                if show:
                    plot_histogram(sel_sent, "Zcores", show=True)
            elif calibrated == 'uniform':
                test_sent, scores, sel_sent = get_sentences_by_method_cal(pool, student, test_sent)
                if show:
                    plot_histogram(sel_sent, "Uniform", show=True)
            else:
                test_sent, _, _, scores, sel_sent = get_sentences_by_method(pool, student, test_sent)
                if show:
                    plot_histogram(sel_sent, calibrated, show=True)
                # pred_prob = clf_test.predict_proba(test_sent)
                # scores = pred_prob[:,0]

            predict = clf_test.predict(test_sent)
            mname = fn.__name__
            print "-" * 40
            test_name = "caliball-{}-size-{}".format(mname, size)
            print test_name
            if show:
                plot_histogram([scores[pool.target == 0], scores[pool.target == 1]], test_name, show=False)
            score_confusion_matrix(pool.target, predict, [0, 1])
            accu = metrics.accuracy_score(pool.target, predict)
            print "Accu %s \t%s" % (test_name, accu)
            results[size].append(sel_sent)
    return results


def other_distribution(exp_clf, fns, pool, sent_clf, student, vct):
    # # create data for testing method
    # select the first sentence always
    print args.train
    print "Testing size: %s" % len(pool.target)
    print "Class distribution: %s" % (1. * pool.target.sum() / len(pool.target))
    student.fn_utility = student.utility_one
    # clf_test = clf
    # clf_test.fit(pool.data, pool.target)
    # student.set_sentence_model(clf_test)
    clf_test = sent_clf
    offset = 0
    for fn in fns:
        ## firstk
        test_sent = []
        student.score = fn
        test_sent, target_sent, text_sent = get_sentences_by_method(pool, student, test_sent)
        predict = clf_test.predict(test_sent)
        pred_prob = clf_test.predict_proba(test_sent)
        mname = fn.__name__
        plot_histogram(pred_prob[:, 0], mname)
        # print "METHOD: %s" % fn.__name__

        if False:
            print_document(text_sent, offset, method_name=mname, top=500, truth=pool.target,
                           prediction=predict)  #, org_doc=pool.text)
        offset += 500
        # accu = metrics.accuracy_score(pool.target, predict)
        print mname
        score_confusion_matrix(pool.target, predict, [0, 1])
        #print "Accu %s \t%s" % (student.score.__name__, accu)
    if False:  ## show top feature method
        test_sent, target_sent = score_top_feat(pool, sent_detector, exp_clf, vct)
        predict = clf_test.predict(test_sent)

        accu = metrics.accuracy_score(pool.target, predict)
        print "Accu %s \t%s" % (score_top_feat.__name__, accu)


def main():
    test_methods = False
    test_distribution = True
    sent_average = False
    sizes = range(1000, 20000, 5000)

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

    # clf = linear_model.LogisticRegression(penalty='l1', C=args.expert_penalty)
    clf = LogisticRegressionAdaptive(penalty='l1', C=1)

    exp_clf, data, vct, cost_model, sent_clf, sent_data = get_data(clf, args.train, [categories[0]], args.fixk, min_size, vct, raw=True)  # expert: classifier, data contains train and test
    print "\nExpert: %s " % exp_clf

    print ("Sentences scoring")
    t0 = time.time()
    ### experiment starts

    student = structured.AALStructuredReading(model=clf, accuracy_model=None, budget=args.budget, seed=args.seed, vcn=vct,
                                              subpool=250, cost_model=cost_model)
    student.set_score_model(exp_clf)  # expert model
    student.set_sentence_model(sent_clf)  # expert sentence model
    student.limit = 2
    print "Expert: :", exp_clf
    print "Sentence:", sent_clf

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
    pool.remaining = range(pool.data.shape[0])  # indices of the pool

    if sent_average:
        print sentences_average(pool, vct)

    fns = [student.score_fk, student.score_max, student.score_rnd, student.score_max_feat, student.score_max_sim]
    fns = [student.score_max]

    if test_methods:
        other_distribution(exp_clf, fns, pool, sent_clf, student, vct)  ## get prob. distribution without calibration

    calibrated = 'random'
    results = []
    if test_distribution:
        from collections import Counter
        ## create data for testing method
        # select the first sentence always
        # for i in range(5):
        for cal in ['uniform', 'zscore']:
            results.append(score_distribution(calibrated, pool, sent_data, student, [1], cheating=True))
        avg = Counter()
        for r in results:
            c = Counter(r[1][0])
            for k,v in c.iteritems():
                avg[k] += v/10.

        plt.hist(avg.keys(), weights=avg.values(), bins=100, align='mid', alpha=.65)
        plt.title("Random Distributions t=10", fontsize=12)
        plt.xlabel("Sentence Location")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()


    print "Elapsed time %.3f" % (time.time() - t0)


def score_confusion_matrix(true_labels, predicted, labels):
    cm = metrics.confusion_matrix(true_labels, predicted, labels=labels)
    print "Predicted -->"
    print "\t" + "\t".join(str(l) for l in np.unique(true_labels))
    for l in np.unique(true_labels):
        print "{}\t{}".format(l,"\t".join(["{}".format(r) for r in cm[l]]))
    print "\n{}\t{}".format(cm[0][0]+cm[1][0],cm[0][1]+cm[1][1],)


def plot_histogram(values, title, show=False):

    n, bins, patches = plt.hist(values, stacked=True, bins=100, align='mid',label=['y=0', 'y=1'], alpha=.65)

    # plt.title(title + ' Distribution $P_{L}(y=0|x)$ $y=0$ (mean=%.2f, N=%d)' % (np.mean(values), len(values)), fontsize=12)
    plt.xlabel("$P_{L}(\hat{y}=0|x)$")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(title+".png", bbox_inches="tight", dpi=200, transparent=True)
    if show:
        plt.show()
    else:
        plt.clf()


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
        # print "-"*60

if __name__ == '__main__':
    main()


