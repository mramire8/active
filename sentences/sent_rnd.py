__author__ = 'mramire8'
__copyright__ = "Copyright 2014, ML Lab"
__version__ = "0.1"
__status__ = "Research"

import sys
import os

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("../"))
sys.path.append(os.path.abspath("../experiment/"))

from experiment import experiment_utils
import argparse
import numpy as np
from sklearn.datasets.base import Bunch
from datautil.load_data import load_from_file, split_data
from sklearn import linear_model
import time
from sklearn import metrics
from strategy import randomsampling, structured
from collections import defaultdict
from expert import baseexpert
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# import random
import nltk
from scipy.sparse import vstack
# import re
# from scipy.sparse import diags
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

ap.add_argument('--expert',
                metavar='EXPERT_TYPE',
                type=str,
                default='pred',
                help='Type of expert [neutral|true|pred]')

ap.add_argument('--student',
                metavar='STUDENT_TYPE',
                type=str,
                default='rnd_firstkmax_tfe',
                help='Type of 7 [sr|rnd|fixkSR|sr_seq|firsk_seq|rnd_max | rnd_firstk| firstkmax_tfe | firstkmax_seq_tfe]')

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
                default=150,
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
                default=100,
                help='Max number of iterations')

ap.add_argument('--seed',
                metavar='SEED',
                type=int,
                default=876543210,
                help='Max number of iterations')

ap.add_argument('--cheating',
                action="store_true",
                help='experiment cheating version - study purposes')

args = ap.parse_args()
rand = np.random.mtrand.RandomState(args.seed)
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

print args
print


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


def split_data_sentences(data, sent_detector, vct=CountVectorizer()):
    sent_train = []
    labels = []
    tokenizer = vct.build_tokenizer()

    print ("Splitting into sentences...")
    ## Convert the documents into sentences: train
    for t, sentences in zip(data.target, sent_detector.batch_tokenize(data.data)):
        sents = [s for s in sentences if len(tokenizer(s)) > 1]
        sent_train.extend(sents)  # at the sentences separately as individual documents
        labels.extend([t] * len(sents))  # Give the label of the document to all its sentences
    return labels, sent_train


def get_student(clf, cost_model, sent_clf, t, vct):
    cheating = args.cheating

    if args.student in "fixkSR":
        # student = structured.AALStructuredFixk(model=clf, accuracy_model=None, budget=args.budget, seed=t, vcn=vct,
        #                                         subpool=250, cost_model=cost_model)
        # student.set_score_model(expert)
        student = structured.AALStructuredReadingFirstK(model=clf, accuracy_model=None, budget=args.budget, seed=t,
                                                        vcn=vct,
                                                        subpool=250, cost_model=cost_model)
    elif args.student in "fixkSRMax":
        student = structured.AALStructuredReadingFirstKMax(model=clf, accuracy_model=None, budget=args.budget, seed=args.seed,
                                                     vcn=vct,
                                                     subpool=250, cost_model=cost_model)
    elif args.student in "sr":
        student = structured.AALStructuredReadingMax(model=clf, accuracy_model=None, budget=args.budget, seed=args.seed,
                                                     vcn=vct,
                                                     subpool=250, cost_model=cost_model)

    elif args.student in "sr_tfe":
        student = structured.AALTFEStructuredReading(model=clf, accuracy_model=None, budget=args.budget, seed=args.seed,
                                                     vcn=vct,
                                                     subpool=250, cost_model=cost_model)

    elif args.student in "firstk_tfe":
        student = structured.AALTFEStructuredReadingFK(model=clf, accuracy_model=None, budget=args.budget, seed=args.seed,
                                                     vcn=vct,
                                                     subpool=250, cost_model=cost_model)
    elif args.student in "firstkmax_tfe":
        student = structured.AALTFEStructuredReadingFK(model=clf, accuracy_model=None, budget=args.budget, seed=args.seed,
                                                     vcn=vct,
                                                     subpool=250, cost_model=cost_model)
        student.set_sent_score(student.score_fk_max)

    elif args.student in "rnd_sr_tfe":
        student = structured.AALTFERandomThenSR(model=clf, accuracy_model=None, budget=args.budget,
                                                     seed=args.seed, vcn=vct, subpool=250, cost_model=cost_model)
        student.set_sent_score(student.score_max)
        student.fn_utility = student.utility_rnd

    elif args.student in "rnd_firstkmax_tfe":
        student = structured.AALTFERandomThenSR(model=clf, accuracy_model=None, budget=args.budget,
                                                     seed=args.seed, vcn=vct, subpool=250, cost_model=cost_model)
        student.set_sent_score(student.score_fk_max)
        student.fn_utility = student.utility_rnd

    elif args.student in "rnd":
        # student = structured.AALStructuredReadingRandomK(model=clf, accuracy_model=None, budget=args.budget, seed=args.seed, vcn=vct,
        #                                    subpool=250, cost_model=cost_model)
        # student.set_score_model(clf)  # student classifier
        # student.set_sentence_model(sent_clf)  # cheating part, use and expert in sentences
        raise ValueError("RND sentences selection is not available yet.")

    elif args.student in "sr_seq":
        student = structured.AALUtilityThenSR_Max(model=clf, accuracy_model=None, budget=args.budget, seed=args.seed,
                                                  vcn=vct, subpool=250, cost_model=cost_model)
    elif args.student in "sr_seq_tfe":
        student = structured.AALTFEUtilityThenSR_Max(model=clf, accuracy_model=None, budget=args.budget, seed=args.seed,
                                                     vcn=vct, subpool=250, cost_model=cost_model)

    elif args.student in "rnd_max":
        student = structured.AALRandomReadingMax(model=clf, accuracy_model=None, budget=args.budget, seed=args.seed,
                                                 vcn=vct, subpool=250, cost_model=cost_model)


    elif args.student in "rnd_firstk":
        student = structured.AALRandomReadingFK(model=clf, accuracy_model=None, budget=args.budget, seed=args.seed,
                                                vcn=vct, subpool=250, cost_model=cost_model)


    elif args.student in "firstk_seq":
        student = structured.AALUtilityThenSR_Firstk(model=clf, accuracy_model=None, budget=args.budget,
                                                     seed=args.seed, vcn=vct, subpool=250, cost_model=cost_model)

    elif args.student in "firstkmax_seq":
        student = structured.AALUtilityThenSR_Firstk(model=clf, accuracy_model=None, budget=args.budget,
                                                     seed=args.seed, vcn=vct, subpool=250, cost_model=cost_model)
        student.set_sent_score(student.score_fk_max)

    elif args.student in "firstkmax_seq_tfe":
        student = structured.AALTFEUtilityThenSR_Max(model=clf, accuracy_model=None, budget=args.budget,
                                                     seed=args.seed, vcn=vct, subpool=250, cost_model=cost_model)
        student.set_sent_score(student.score_fk_max)

    else:
        raise ValueError("Oops! We do not know that anytime strategy. Try again.")

    student.set_score_model(clf)  # student classifier
    student.set_sentence_model(sent_clf)  # cheating part, use and expert in sentences
    student.set_cheating(cheating)

    return student


def update_sentence(neutral_data, neu_x, neu_y, labels, query_index, pool, vct):
    """`
    Add to P_S all the sentences in the documents with the label of the document
    :param neutral_data:
    :param neu_x:
    :param neu_y:
    :param labels:
    :param query_index:
    :param pool:
    :param vct:
    :return:
    """
    if not args.cheating:  #prepapre the data to update P_S

        qlbl = []
        ## for every query find the text and sentences
        for lbl, index in zip(labels, query_index):
            subinstances = sent_detector.tokenize(pool.text[index])
            doc_sentences = vct.transform(subinstances)

            # if student.sent_model is None:  # in the first iteration, add all
            #     confidence = [True] * len(subinstances)
            # else:
            #     confidence = [student.sent_model.predict_proba(s).max() > args.neutral_threshold
            #                   for s in doc_sentences]
            # print len(confidence)

            for xik in doc_sentences: #,  confidence:

                # if confident:
                    if isinstance(neutral_data, list):
                        neutral_data = xik
                    else:
                        neutral_data = vstack([neutral_data, xik], format='csr')
                    qlbl.append(lbl)
        neu_y = np.append(neu_y, qlbl)
        neu_x = neutral_data
    else:  # for compatibility with cheating experiments
        return np.array([]),np.array([]),np.array([])
    return neu_x, neu_y, neutral_data


def update_sentence_query(neutral_data, neu_x, neu_y, query, labels):
    '''
    Add only the annotated sentence to P_s
    :param neutral_data:
    :param neu_x:
    :param neu_y:
    :param query:
    :param labels:
    :return:
    '''
    if not args.cheating:
        if isinstance(neutral_data, list):
            neutral_data = query
        else:
            for q in query:
                neutral_data = vstack([neutral_data, q], format='csr')
        neu_y = np.append(neu_y, labels)
        neu_x = neutral_data
    else:
        return np.array([]),np.array([]),np.array([])
    return neu_x, neu_y, neutral_data


def main():
    accuracies = defaultdict(lambda: [])

    ora_accu = defaultdict(lambda: [])
    oracle_accuracies =[]

    aucs = defaultdict(lambda: [])

    x_axis = defaultdict(lambda: [])

    vct = TfidfVectorizer(encoding='ISO-8859-1', min_df=1, max_df=1.0, binary=False, ngram_range=(1, 2),
                          token_pattern='\\b\\w+\\b')  #, tokenizer=StemTokenizer())

    print("Start loading ...")
    # data fields: data, bow, file_names, target_names, target

    ########## NEWS GROUPS ###############
    # easy to hard. see "Less is More" paper: http://axon.cs.byu.edu/~martinez/classes/678/Presentations/Clawson.pdf
    categories = [['alt.atheism', 'talk.religion.misc'],
                  ['comp.graphics', 'comp.windows.x'],
                  ['comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware'],
                  ['rec.sport.baseball', 'sci.crypt']]

    min_size = 10

    args.fixk = None

    data, vct = load_from_file(args.train, [categories[3]], args.fixk, min_size, vct, raw=True)

    print("Data %s" % args.train)
    print("Data size %s" % len(data.train.data))

    parameters = experiment_utils.parse_parameters_mat(args.cost_model)

    print "Cost Parameters %s" % parameters

    cost_model = experiment_utils.set_cost_model(args.cost_function, parameters=parameters)
    print "\nCost Model: %s" % cost_model.__class__.__name__

    ### SENTENCE TRANSFORMATION
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

    ## delete <br> to "." to recognize as end of sentence
    data.train.data = experiment_utils.clean_html(data.train.data)
    data.test.data = experiment_utils.clean_html(data.test.data)

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

    labels, sent_train = split_data_sentences(expert_data.oracle.train, sent_detector)

    expert_data.oracle.train.data = sent_train
    expert_data.oracle.train.target = np.array(labels)
    expert_data.oracle.train.bow = vct.transform(expert_data.oracle.train.data)

    exp_clf = linear_model.LogisticRegression(penalty='l1', C=args.expert_penalty)
    exp_clf.fit(expert_data.oracle.train.bow, expert_data.oracle.train.target)

    if "neutral" in args.expert:
        expert = baseexpert.NeutralityExpert(exp_clf, threshold=args.neutral_threshold,
                                             cost_function=cost_model.cost_function)
    elif "true" in args.expert:
        expert = baseexpert.TrueOracleExpert(cost_function=cost_model.cost_function)
    elif "pred" in args.expert:
        expert = baseexpert.PredictingExpert(exp_clf,  #threshold=args.neutral_threshold,
                                             cost_function=cost_model.cost_function)
    else:
        raise Exception("We need an expert!")

    print "\nExpert: %s " % expert

    #### EXPERT CLASSIFIER: SENTENCES
    print("Training sentence expert")
    labels, sent_train = split_data_sentences(expert_data.sentence.train, sent_detector)

    expert_data.sentence.train.data = sent_train
    expert_data.sentence.train.target = np.array(labels)
    expert_data.sentence.train.bow = vct.transform(expert_data.sentence.train.data)

    sent_clf = None
    if args.cheating:
        sent_clf = linear_model.LogisticRegression(penalty='l1', C=args.expert_penalty)
        sent_clf.fit(expert_data.sentence.train.bow, expert_data.sentence.train.target)

    #### STUDENT CLASSIFIER
    clf = linear_model.LogisticRegression(penalty="l1", C=1)
    # clf = set_classifier(args.classifier)


    print "\nStudent Classifier: %s" % clf
    print "\nSentence Classifier: %s" % sent_clf
    print "\nExpert Oracle Classifier: %s" % exp_clf

    #### ACTIVE LEARNING SETTINGS
    step_size = args.step_size
    bootstrap_size = args.bootstrap
    evaluation_points = 200

    print("\nExperiment: step={0}, BT={1}, plot points={2}, fixk:{3}, minsize:{4}".format(step_size, bootstrap_size,
                                                                                          evaluation_points, args.fixk,
                                                                                          min_size))
    print ("Anytime active learning experiment - use objective function to pick data")
    t0 = time.time()
    tac = []
    tau = []
    ### experiment starts
    for t in range(args.trials):
        trial_accu = []

        trial_aucs = []

        print "*" * 60
        print "Trial: %s" % t

        student = get_student(clf, cost_model, sent_clf, t, vct)

        print "\nStudent: %s " % student

        train_indices = []
        neutral_data = []  # save the xik vectors
        train_x = []
        train_y = []
        neu_x = []  # data to train the classifier
        neu_y = np.array([])

        pool = Bunch()
        pool.data = data.train.bow.tocsr()  # full words, for training
        pool.text = data.train.data
        pool.target = data.train.target
        pool.predicted = []
        pool.remaining = list(rand.permutation(pool.data.shape[0]))  # indices of the pool

        bootstrapped = False
        current_cost = 0
        iteration = 0
        query_index = None
        query_size = None
        oracle_answers = 0
        while 0 < student.budget and len(pool.remaining) > step_size and iteration <= args.maxiter:
            util = []

            if not bootstrapped:
                # k = int(step_size / 2)
                # data_points = defaultdict(lambda: [])
                #
                # for i in pool.remaining:
                #     data_points[pool.target[i]].append(i)
                #
                # query_index = []
                # for label in data_points.keys():
                #     candidates = data_points[label]
                #     indices = rand.permutation(len(candidates))
                #     query_index.extend([candidates[index] for index in indices[:k]])
                #
                query_index = pool.remaining[:bootstrap_size]
                bootstrapped = True
                query = pool.data[query_index]

                print
            else:

                chosen = student.pick_next(pool=pool, step_size=step_size)

                query_index = [x for x, y in chosen]  # document id of chosen instances
                query = [y for x, y in chosen]  # sentence of the document

                query_size = [1] * len(query_index)

            ground_truth = pool.target[query_index]

            if iteration == 0:  ## bootstrap uses ground truth
                labels = ground_truth
                spent = [0] * len(ground_truth)  ## bootstrap cost is ignored
            else:
                # print "ask labels"
                labels = expert.label_instances(query, ground_truth)
                spent = expert.estimate_instances(query_size)

            ### accumulate the cost of the query
            query_cost = np.array(spent).sum()
            current_cost += query_cost

            useful_answers = np.array([[x, y] for x, y in zip(query_index, labels) if y is not None])

            neutral_answers = np.array([[x, z] for x, y, z in zip(query_index, labels, query_size) if y is None]) \
                if iteration != 0 else np.array([])

            ## add data recent acquired to train
            if useful_answers.shape[0] != 0:
                train_indices.extend(useful_answers[:, 0])
                print train_indices
                # add labels to training
                train_x = pool.data[train_indices]  # # train with all the words

                # update labels with the expert labels
                train_y.extend(useful_answers[:, 1])

            neu_x, neu_y, neutral_data = update_sentence(neutral_data, neu_x, neu_y, labels, query_index, pool, vct)

            if neu_y.shape[0] != neu_x.shape[0]:
                raise Exception("Training data corrupted!")
            if train_x.shape[0] != len(train_y):
                raise Exception("Training data corrupted!")

            # remove labels from pool
            for qi in query_index:
                pool.remaining.remove(qi)

            # retrain the model
            current_model = student.train_all(train_x, train_y, neu_x, neu_y)

            # evaluate and save results
            y_probas = current_model.predict_proba(data.test.bow)

            auc = metrics.roc_auc_score(data.test.target, y_probas[:, 1])

            pred_y = current_model.classes_[np.argmax(y_probas, axis=1)]

            correct_labels = (np.array(ground_truth) == np.array(labels).reshape(len(labels))).sum()

            accu = metrics.accuracy_score(data.test.target, pred_y)

            print ("TS:{0}\tAccu:{1:.3f}\tAUC:{2:.3f}\tCost:{3:.2f}\tCumm:{4:.2f}\tSpent:{5}\tneu:{6}\t{7}\tND:{8}\tTD:{9}\t ora_accu:{10}".format(
                len(train_indices),
                accu,
                auc, query_cost,
                current_cost,
                spent,
                len(neutral_answers), neu_y.shape[0], neu_y.sum(), np.array(train_y).sum(), correct_labels))

            ## the results should be based on the cost of the labeling
            if iteration > 0:  # bootstrap iteration

                student.budget -= query_cost  ## Bootstrap doesn't count
                # oracle accuracy (from queries)
                oracle_answers += correct_labels
                x_axis_range = current_cost
                x_axis[x_axis_range].append(current_cost)
                ## save results
                accuracies[x_axis_range].append(accu)
                aucs[x_axis_range].append(auc)
                ora_accu[x_axis_range].append(1. * correct_labels/len(ground_truth))
                # partial trial results
                trial_accu.append([x_axis_range, accu])
                trial_aucs.append([x_axis_range, auc])
                # oracle_accuracies[x_axis_range].append(oracle_answers)
            iteration += 1
            # end of budget loop

        tac.append(trial_accu)
        tau.append(trial_aucs)
        oracle_accuracies.append(1.*oracle_answers / (len(train_indices)-bootstrap_size))
        print "Trial: {}, Oracle right answers: {}, Iteration: {}, Labels:{}, ACCU-OR:{}".format(t, oracle_answers,
                 iteration, len(train_indices)-bootstrap_size,1.*oracle_answers / (len(train_indices)-bootstrap_size))
        #end trial loop
    if args.cost_function not in "uniform":
        accuracies = experiment_utils.extrapolate_trials(tac, cost_25=parameters[1][1], step_size=args.step_size)
        aucs = experiment_utils.extrapolate_trials(tau, cost_25=parameters[1][1], step_size=args.step_size)
    print "\nAverage oracle accuracy: ", np.array(oracle_accuracies).mean()
    print("Elapsed time %.3f" % (time.time() - t0))
    cheating = "CHEATING" if args.cheating else "NOCHEAT"
    experiment_utils.print_extrapolated_results(accuracies, aucs, file_name=args.train+"-"+cheating+"-"+args.student)
    experiment_utils.oracle_accuracy(ora_accu, file_name=args.train+"-"+cheating+"-"+args.student)


def format_query(query_labels):
    string = ""
    for l, q in query_labels:
        string = string + "{0}".format(l)
        for qi in q:
            string = string + "\t{0:.2f} ".format(qi)
        string = string + "\n"
    return string


def main2():
    # # load paramters: student, expert, cost, sampling, ...
    # set_options()
    #
    # # load data and preprocess
    # pre_process_data(set_datasets())
    #
    # # start loop
    pass

## MAIN FUNCTION
if __name__ == '__main__':
    main()


