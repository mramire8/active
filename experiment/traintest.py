__author__ = 'mramire8'
__copyright__ = "Copyright 2013, ML Lab"
__version__ = "0.1"
__status__ = "Development"

import sys
import os

sys.path.append(os.path.abspath("."))

import argparse
from sklearn import metrics
from sklearn.datasets.base import Bunch
from collections import defaultdict
from sklearn.naive_bayes import MultinomialNB

from datautil.textutils import StemTokenizer
from strategy import randomsampling
from datautil.load_data import *
from expert import baseexpert

import time
from experiment_utils import *

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawTextHelpFormatter)
ap.add_argument('--train',
                metavar='TRAIN',
                #default="C:/Users/mramire8/Documents/Research/Oracle confidence and Interruption/newExperiment/evidencefit/data/evidence-data-nolabel.txt",
                default="20news",
                help='training data (libSVM format)')

ap.add_argument('--test',
                metavar='TEST',
                default="C:/Users/mramire8/Documents/Research/Oracle confidence and Interruption/dataset/aclImdb/from-userstudy/usefulwords/docs-time.txt",
                help='document of the user study')

ap.add_argument('--seed',
                metavar='SEED',
                type=int,
                default=1234567,
                help='random seed')

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
                default=20000,
                help='budget')

ap.add_argument('--step-size',
                metavar='STEP_SIZE',
                type=int,
                default=1,
                help='instances to acquire at every iteration')

ap.add_argument('--bootstrap',
                metavar='BOOTSTRAP',
                type=int,
                default=2,
                help='size of the initial labeled dataset')

ap.add_argument('--cost-model',
                metavar='COST_MODEL',
                type=str,
                default="0.0642,8.4204",
                help='coefficients of the model separated by commas [x^n x^n-1....x  x^0]')

ap.add_argument('--cost-function',
                metavar='COST_FUNCTION',
                type=str,
                default="uniform",
                help='cost function of the x-axis [uniform|log|linear]')

ap.add_argument('--accu-model',
                metavar='ACCU_MODEL',
                type=str,
                default="0.5,0",
                help='coefficients of the model separated by commas [x^n x^n-1....x  x^0]')

ap.add_argument('--accu-function',
                metavar='ACCU_FUNCTION',
                type=str,
                default="fixed",
                help='accuracy function known by the student  [fixed|linear|log]')

ap.add_argument('--expert',
                metavar='EXPERT',
                type=str,
                default="fixed",
                help='Expert type [fixed|log|linear|true]')

ap.add_argument('--fixk',
                metavar='FIXK',
                type=int,
                default=None,
                help='fixed k number of words')
ap.add_argument('--maxiter',
                metavar='MAXITER',
                type=int,
                default=20,
                help='Max number of iterations')

args = ap.parse_args()
rand = np.random.mtrand.RandomState(args.seed)

print args
print



def main():
    accuracies = defaultdict(lambda: [])

    aucs = defaultdict(lambda: [])

    x_axis = defaultdict(lambda: [])

    vct = CountVectorizer(encoding='ISO-8859-1', min_df=5, max_df=1.0, binary=True, ngram_range=(1, 1),
                          token_pattern='\\b\\w+\\b', tokenizer=StemTokenizer())
    vct_analizer = vct.build_tokenizer()
    print("Start loading ...")
    # data fields: data, bow, file_names, target_names, target

    ########## NEWS GROUPS ###############
    # easy to hard. see "Less is More" paper: http://axon.cs.byu.edu/~martinez/classes/678/Presentations/Clawson.pdf
    categories = [['alt.atheism', 'talk.religion.misc'],
                  ['comp.graphics', 'comp.windows.x'],
                  ['comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware'],
                  ['rec.sport.baseball', 'sci.crypt']]

    min_size = max(50, args.fixk)

    if "imdb" in args.train:
        ########## IMDB MOVIE REVIEWS ###########
        data = load_imdb(args.train, shuffle=True, rnd=2356, vct=vct, min_size=min_size,
                               fix_k=args.fixk)  # should brind data as is
    elif "aviation" in args.train:
        raise Exception("We are not ready for that data yet")
    elif "20news" in args.train:
        ########## 20 news groups ######
        data = load_20newsgroups(categories=categories[0], vectorizer=vct, min_size=min_size,
                                       fix_k=args.fixk)  # for testing purposes
    elif "dummy" in args.train:
        ########## DUMMY DATA###########
        data = load_dummy("C:/Users/mramire8/Documents/code/python/data/dummy", shuffle=True,
                                rnd=2356, vct=vct, min_size=0, fix_k=args.fixk)
    else:
        raise Exception("We do not know that dataset")

    print("Data %s" % args.train)
    print("Data size %s" % len(data.train.data))
    #print(data.train.data[0])
    #### COST MODEL
    parameters = parse_parameters(args.cost_model)

    print "Cost Parameters %s" % parameters

    cost_model = set_cost_model(parameters)

    print "\nCost Model: %s" % cost_model.__class__.__name__

    #### ACCURACY MODEL
    # try:
    # #     accu_parameters = parse_parameters(args.accu_model)
    # except ValueError:
    accu_parameters = parse_parameters_mat(args.accu_model)
    # else
    #     print("Error: Accuracy parameters didn't work")

    print "Accuracy Parameters %s" % accu_parameters
    #if "fixed" in args.accu_function:
    #    accuracy_model = base_models.FixedAccuracyModel(accuracy_value=.7)
    #elif "log" in args.accu_function:
    #    accuracy_model = base_models.LogAccuracyModel(model=parameters)
    #elif "linear" in args.accu_function:
    #    accuracy_model = base_models.LRAccuracyModel(model=parameters)
    #else:
    #    raise Exception("We need a defined cost function options [fixed|log|linear]")
    #
    #print "\nAccuracy Model: %s " % accuracy_model

    #### CLASSIFIER
    #### Informed priors
    #feature_counts = np.ones(x_train.shape[0]) * x_train
    #feature_frequencies = feature_counts / np.sum(feature_counts)
    #alpha = feature_frequencies
    alpha = 1
    clf = MultinomialNB(alpha=alpha)
    print "\nClassifier: %s" % clf

    #### EXPERT MODEL
    #expert = baseexpert.BaseExpert()
    if "fixed" in args.expert:
        expert = baseexpert.FixedAccuracyExpert(accuracy_value=accu_parameters[0],
                                                cost_function=cost_model.cost_function)  #average value of accuracy of the experts
    elif "true" in args.expert:
        expert = baseexpert.TrueOracleExpert(cost_function=cost_model.cost_function)
    elif "linear" in args.expert:
        #expert = baseexpert.LRFunctionExpert(model=[0.0019, 0.6363],cost_function=cost_model.cost_function)
        raise Exception("We do not know linear yet!!")
    elif "log" in args.expert:
        expert = baseexpert.LogFunctionExpert(model=accu_parameters, cost_function=cost_model.cost_function)
    elif "direct" in args.expert:
        expert = baseexpert.LookUpExpert(accuracy_value=accu_parameters, cost_function=cost_model.cost_function)
    else:
        raise Exception("We need a defined cost function options [fixed|log|linear]")
        #expert = baseexpert.TrueOracleExpert(cost_function=cost_model.cost_function)
    print "\nExpert: %s " % expert

    #### ACTIVE LEARNING SETTINGS
    step_size = args.step_size
    bootstrap_size = args.bootstrap
    evaluation_points = 200
    eval_range = 1 if (args.budget / evaluation_points) <= 0 else args.budget / evaluation_points
    print("\nExperiment: step={0}, BT={1}, plot points={2}, fixk:{3}, minsize:{4}".format(step_size, bootstrap_size,
                                                                                          evaluation_points, args.fixk,
                                                                                          50))

    t0 = time.time()
    ### experiment starts
    for t in range(args.trials):
        print "*" * 60
        print "Trial: %s" % t
        # TODO shuffle the data??
        #student = baselearner.BaseLearner(model=clf, cost_model=cost_model, accuracy_model=accuracy_model, budget=args.budget,
        #                                  seed=t)
        student = randomsampling.RandomSamplingLearner(model=clf, accuracy_model=None, budget=args.budget, seed=t)
        print "\nStudent: %s " % student
        train_indices = []
        train_x = []
        train_y = []
        pool = Bunch()
        pool.data = data.train.bow.tocsr()   # full words, for training
        pool.fixk = data.train.bowk.tocsr()  # k words BOW for querying
        pool.target = data.train.target
        pool.predicted = []
        pool.kwords = np.array(data.train.kwords)  # k words
        pool.remaining = set(range(pool.data.shape[0]))  # indices of the pool

        #for x in pool.fixk:
        #    print x.todense().sum()

        bootstrapped = False

        current_cost = 0
        iteration = 0
        while 0 < student.budget and len(pool.remaining) > step_size and iteration <= args.maxiter:

            if not bootstrapped:
                ## random bootstrap
                #bt = randomsampling.BootstrapRandom(random_state=t * 10)

                ## random from each bootstrap
                bt = randomsampling.BootstrapFromEach(t * 10)

                query_index = bt.bootstrap(pool=pool, k=bootstrap_size)
                bootstrapped = True
                print "Bootstrap: %s " % bt.__class__.__name__
                print
            else:
                query_index = student.pick_next(pool=pool, k=step_size)

            query = pool.fixk[query_index]  # query with k words

            query_size = [len(vct_analizer(x)) for x in pool.kwords[query_index]]

            #if query_size[0] >50:
            #    print "*** %s" % pool.kwords[query_index]

            ground_truth = pool.target[query_index]
            #labels, spent = expert.label(unlabeled=query, target=ground_truth)
            if iteration == 0: ## bootstrap uses ground truth
                labels = ground_truth
            else:
                #labels = expert.label_instances(query, ground_truth)
                labels = expert.label_instances(query_size, ground_truth)
                #spent = expert.estimate_instances(pool.kwords[query_index])
            spent = expert.estimate_instances(query_size)

            query_cost = np.array(spent).sum()
            current_cost += query_cost

            train_indices.extend(query_index)

            # remove labels from pool
            pool.remaining.difference_update(query_index)

            # add labels to training
            train_x = pool.data[train_indices]  ## train with all the words

            # update labels with the expert labels
            #train_y = pool.target[train_indices]
            train_y.extend(labels)
            if train_x.shape[0] != len(train_y):
                raise Exception("Training data corrupted!")

            # retrain the model
            current_model = student.train(train_x, train_y)
            # evaluate and save results
            y_probas = current_model.predict_proba(data.test.bow)

            #auc = metrics.roc_auc_score(data.test.target, y_probas[:,1])
            auc = metrics.roc_auc_score(data.test.target, y_probas[:, 1])

            pred_y = current_model.classes_[np.argmax(y_probas, axis=1)]

            accu = metrics.accuracy_score(data.test.target, pred_y)

            print (
            "TS:{0}\tAccu:{1:.3f}\tAUC:{2:.3f}\tCost:{3:.2f}\tCumm:{4:.2f}\tSpent:{5}".format(len(train_indices), accu,
                                                                                              auc, query_cost,
                                                                                              current_cost, spent))

            ## the results should be based on the cost of the labeling
            if iteration > 0: # bootstrap iteration

                student.budget -= query_cost ## Bootstrap doesn't count

                #x_axis_range = int(current_cost / eval_range)
                x_axis_range = current_cost
                x_axis[x_axis_range].append(current_cost)
                ## save results
                #accuracies[len(train_indices)].append(accu)
                #aucs[len(train_indices)].append(auc)
                accuracies[x_axis_range].append(accu)
                aucs[x_axis_range].append(auc)
            iteration += 1
    print("Elapsed time %.3f" % (time() - t0))
    print_results(x_axis, accuracies, aucs)




if __name__ == '__main__':
    main()

