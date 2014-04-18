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
# import re
# from scipy.sparse import diags
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

ap.add_argument('--expert',
                metavar='EXPERT_TYPE',
                type=str,
                default='pred',
                help='Type of expert [neutral|true|pred]')

ap.add_argument('--student',
                metavar='STUDENT_TYPE',
                type=str,
                default='sr',
                help='Type of 7 [sr|rnd|fixkSR|]')

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
        expert = baseexpert.PredictingExpert(exp_clf, #threshold=args.neutral_threshold,
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

        if args.student in "fixkSR":
            # student = structured.AALStructuredFixk(model=clf, accuracy_model=None, budget=args.budget, seed=t, vcn=vct,
            #                                         subpool=250, cost_model=cost_model)
            # student.set_score_model(expert)
            student = structured.AALStructuredReadingFirstK(model=clf, accuracy_model=None, budget=args.budget, seed=t, vcn=vct,
                                                    subpool=250, cost_model=cost_model)
            student.set_score_model(clf)  # student classifier
            student.set_sentence_model(sent_clf)  # cheating part, use and expert in sentences
            # TODO: change the following line to update the model in the none cheating version
            student.set_cheating(True)

        elif args.student in "sr":
            student = structured.AALStructuredReadingMax(model=clf, accuracy_model=None, budget=args.budget, seed=args.seed, vcn=vct,
                                               subpool=250, cost_model=cost_model)
            student.set_score_model(clf)  # student classifier
            student.set_sentence_model(sent_clf)  # cheating part, use and expert in sentences
            # TODO: change the following line to update the model in the none cheating version
            student.set_cheating(True)


        elif args.student in "bestk":
            student = structured.AALStructured(model=clf, accuracy_model=None, budget=args.budget, seed=t, vcn=vct,
                                                    subpool=250, cost_model=cost_model)
            student.set_score_model(exp_clf)

        elif args.student in "rnd":
            student = structured.AALStructuredReadingRandomK(model=clf, accuracy_model=None, budget=args.budget, seed=args.seed, vcn=vct,
                                               subpool=250, cost_model=cost_model)
            student.set_score_model(clf)  # student classifier
            student.set_sentence_model(sent_clf)  # cheating part, use and expert in sentences
            # TODO: change the following line to update the model in the none cheating version
            student.set_cheating(True)

        else:
            raise ValueError("Oops! We do not know that anytime strategy. Try again.")

        print "\nStudent: %s " % student

        train_indices = []
        neutral_data = []     # save the xik vectors
        train_x = []
        train_y = []
        neu_x = []            # data to train the classifier
        neu_y = np.array([])

        pool = Bunch()
        pool.data = data.train.bow.tocsr()   # full words, for training
        pool.text = data.train.data
        pool.target = data.train.target
        pool.predicted = []
        pool.remaining = set(range(pool.data.shape[0]))  # indices of the pool

        bootstrapped = False
        current_cost = 0
        iteration = 0
        query_index = None
        query_size = None

        while 0 < student.budget and len(pool.remaining) > step_size and iteration <= args.maxiter:
            util = []

            if not bootstrapped:
                ## random from each bootstrap
                bt = randomsampling.BootstrapFromEach(t * 10)

                query_index = bt.bootstrap(pool=pool, k=bootstrap_size)
                bootstrapped = True
                query = pool.data[query_index]
                print "Bootstrap: %s " % bt.__class__.__name__
                print
            else:
                # print "pick instance"

                chosen = student.pick_next(pool=pool, step_size=step_size)

                query_index = [x for x, y in chosen]  # document id of chosen instances
                query = [y for x, y in chosen]  # sentence of the document

                query_size = [1] * len(query_index)

            ground_truth = pool.target[query_index]
            print ground_truth
            print "-"*40
            if iteration == 0: ## bootstrap uses ground truth
                labels = ground_truth
                spent = [0] * len(ground_truth) ## bootstrap cost is ignored
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

            print labels
            print "*"*40

            ## add data recent acquired to train
            if useful_answers.shape[0] != 0:
                # print "get training"
                # train_indices.extend(query_index)
                train_indices.extend(useful_answers[:, 0])

                # add labels to training
                train_x = pool.data[train_indices]  # # train with all the words

                # update labels with the expert labels
                #train_y = pool.target[train_indices]
                train_y.extend(useful_answers[:, 1])

            if neutral_answers.shape[0] != 0:
                # current query neutrals
                qlbl = []

                for xik, lbl in zip(query, labels):
                    # neutral_data.append(xik)
                    if isinstance(neutral_data, list):
                        neutral_data = xik
                    else:
                        neutral_data = vstack([neutral_data, xik], format='csr')
                    qlbl.append(neutral_label(lbl))

                ## append the labels of the current query
                neu_y = np.append(neu_y, qlbl)
                neu_x = neutral_data
                #end usefulanswers


            if train_x.shape[0] != len(train_y):
                raise Exception("Training data corrupted!")

            # remove labels from pool
            pool.remaining.difference_update(query_index)

            # retrain the model
            # current_model = student.train(train_x, train_y)
            # print "train models"
            current_model = student.train_all(train_x, train_y, neu_x, neu_y)
            # print "evaluate"
            # evaluate and save results
            y_probas = current_model.predict_proba(data.test.bow)

            auc = metrics.roc_auc_score(data.test.target, y_probas[:, 1])

            pred_y = current_model.classes_[np.argmax(y_probas, axis=1)]

            accu = metrics.accuracy_score(data.test.target, pred_y)

            print ("TS:{0}\tAccu:{1:.3f}\tAUC:{2:.3f}\tCost:{3:.2f}\tCumm:{4:.2f}\tSpent:{5}\tneu:{6}\t{7}".format(
                len(train_indices),
                accu,
                auc, query_cost,
                current_cost,
                spent,
                len(neutral_answers), neu_y.shape[0]))

            ## the results should be based on the cost of the labeling
            if iteration > 0:   # bootstrap iteration

                student.budget -= query_cost ## Bootstrap doesn't count

                x_axis_range = current_cost
                x_axis[x_axis_range].append(current_cost)
                ## save results
                accuracies[x_axis_range].append(accu)
                aucs[x_axis_range].append(auc)
                # partial trial results
                trial_accu.append([x_axis_range, accu])
                trial_aucs.append([x_axis_range, auc])

            iteration += 1
            # end of budget loop

        tac.append(trial_accu)
        tau.append(trial_aucs)
        #end trial loop
    if args.cost_function not in "uniform":
        accuracies = experiment_utils.extrapolate_trials(tac, cost_25=parameters[1][1], step_size=args.step_size)
        aucs = experiment_utils.extrapolate_trials(tau, cost_25=parameters[1][1], step_size=args.step_size)

    print("Elapsed time %.3f" % (time.time() - t0))
    experiment_utils.print_extrapolated_results(accuracies, aucs, file_name=args.student)


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


