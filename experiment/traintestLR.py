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
from sklearn.linear_model import LogisticRegression
from datautil.textutils import StemTokenizer
from strategy import randomsampling
from datautil.load_data import *
from expert import baseexpert
import pickle
import time
from experiment_utils import *

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawTextHelpFormatter)
ap.add_argument('--train',
                metavar='TRAIN',
                default="evergreen",
                help='training data (libSVM format)')

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
                default=2000,
                help='budget')

ap.add_argument('--step-size',
                metavar='STEP_SIZE',
                type=int,
                default=5,
                help='instances to acquire at every iteration')

ap.add_argument('--bootstrap',
                metavar='BOOTSTRAP',
                type=int,
                default=10,
                help='size of the initial labeled dataset')

ap.add_argument('--cost-model',
                metavar='COST_MODEL',
                type=str,
                default="[1,0]",
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
                default="true",
                help='Expert type [fixed|log|linear|true]')

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

ap.add_argument('--fixk',
                metavar='FIXK',
                type=int,
                default=None,
                help='fixed k number of words')
ap.add_argument('--maxiter',
                metavar='MAXITER',
                type=int,
                default=200,
                help='Max number of iterations')

ap.add_argument('--student',
                metavar='STUDENT',
                type=str,
                default="unc",
                help='Max number of iterations')

ap.add_argument('--classifier',
                metavar='CLASSIFIER',
                type=str,
                default="lr",
                help='underlying classifier')

args = ap.parse_args()
rand = np.random.mtrand.RandomState(args.seed)

print args
print



def main():
    accuracies = defaultdict(lambda: [])

    aucs = defaultdict(lambda: [])

    x_axis = defaultdict(lambda: [])

    vct = CountVectorizer(encoding='latin-1', min_df=5, max_df=1.0, binary=True, ngram_range=(1, 3),
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

    min_size = max(10, args.fixk)

    if args.fixk < 0:
        args.fixk = None

    # data = load_dataset(args.train, args.fixk, categories[0], vct, min_size, percent=.5)
    # fixk_saved = "{0}{1}.p".format(args.train, args.fixk)

    data, vct = load_from_file(args.train, categories, args.fixk, min_size, vct)

    print("Data %s" % args.train)
    print("Data size %s" % len(data.train.data))

    #### COST MODEL
    parameters = parse_parameters_mat(args.cost_model)
    print "Cost Parameters %s" % parameters
    cost_model = set_cost_model(args.cost_function, parameters=parameters)
    print "\nCost Model: %s" % cost_model.__class__.__name__

    #### ACCURACY MODEL
    accu_parameters = parse_parameters_mat(args.accu_model)

    #### CLASSIFIER
    clf = set_classifier(args.classifier)
    print "\nClassifier: %s" % clf

    #### EXPERT MODEL

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
    elif "neutral" in args.expert:
        exp_clf = LogisticRegression(penalty='l1', C=1)
        exp_clf.fit(data.test.bow, data.test.target)
        expert = baseexpert.NeutralityExpert(exp_clf, threshold=args.neutral_threshold,
                                         cost_function=cost_model.cost_function)
    else:
        raise Exception("We need a defined cost function options [fixed|log|linear]")

    exp_clf = LogisticRegression(penalty='l1', C=args.expert_penalty)
    exp_clf.fit(data.test.bow, data.test.target)
    print "\nExpert: %s " % expert
    coef = exp_clf.coef_[0]
    # print_features(coef, vct.get_feature_names())
    #### ACTIVE LEARNING SETTINGS
    step_size = args.step_size
    bootstrap_size = args.bootstrap
    evaluation_points = 200

    print("\nExperiment: step={0}, BT={1}, plot points={2}, fixk:{3}, minsize:{4}".format(step_size, bootstrap_size,
                                                                                          evaluation_points, args.fixk,
                                                                                          50))

    t0 = time.time()
    tac = []
    tau = []
    ### experiment starts
    for t in range(args.trials):
        trial_accu = []

        trial_aucs = []

        print "*" * 60
        print "Trial: %s" % t
        if  args.student in "unc":
            student = randomsampling.UncertaintyLearner(model=clf, accuracy_model=None, budget=args.budget, seed=t,
                                                        subpool=250)
        else:
            student = randomsampling.RandomSamplingLearner(model=clf, accuracy_model=None, budget=args.budget, seed=t)

        print "\nStudent: %s " % student

        train_indices = []
        train_x = []
        train_y = []
        pool = Bunch()
        pool.data = data.train.bow.tocsr()   # full words, for training
        if args.fixk is None:
            pool.fixk = data.train.bow.tocsr()
        else:
            pool.fixk = data.train.bowk.tocsr()  # k words BOW for querying
        pool.target = data.train.target
        pool.predicted = []
        # pool.kwords = np.array(data.train.kwords)  # k words
        pool.remaining = set(range(pool.data.shape[0]))  # indices of the pool


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

            # query = pool.fixk[query_index]  # query with k words
            query = pool.data[query_index]
            # print query_index
            # query_size = [len(vct_analizer(x)) for x in pool.kwords[query_index]]
            query_size = [1]*query.shape[0]

            ground_truth = pool.target[query_index]

            if iteration == 0: ## bootstrap uses ground truth
                labels = ground_truth
                spent = [0] * len(ground_truth)
            else:
                labels = expert.label_instances(query, ground_truth)
                spent = expert.estimate_instances(query_size)

            query_cost = np.array(spent).sum()
            current_cost += query_cost

            # train_indices.extend(query_index)

            # remove labels from pool
            pool.remaining.difference_update(query_index)

            # add labels to training
            # train_x = pool.data[train_indices]  ## train with all the words

            # update labels with the expert labels
            useful_answers = np.array([[x, y] for x, y in zip(query_index, labels) if y is not None])
            if useful_answers.shape[0] != 0:
                train_indices.extend(useful_answers[:, 0])
                # add labels to training
                train_x = pool.data[train_indices]  ## train with all the words
                # update labels with the expert labels
                train_y.extend(useful_answers[:, 1])


            if train_x.shape[0] != len(train_y):
                raise Exception("Training data corrupted!")

            # retrain the model
            current_model = student.train(train_x, train_y)
            # evaluate and save results
            y_probas = current_model.predict_proba(data.test.bow)

            auc = metrics.roc_auc_score(data.test.target, y_probas[:, 1])

            pred_y = current_model.classes_[np.argmax(y_probas, axis=1)]

            accu = metrics.accuracy_score(data.test.target, pred_y)

            print (
            "TS:{0}\tAccu:{1:.3f}\tAUC:{2:.3f}\tCost:{3:.2f}\tCumm:{4:.2f}\tSpent:{5}".format(len(train_indices), accu,
                                                                                              auc, query_cost,
                                                                                              current_cost, format_spent(spent)))

            ## the results should be based on the cost of the labeling
            if iteration > 0: # bootstrap iteration

                student.budget -= query_cost ## Bootstrap doesn't count

                #x_axis_range = int(current_cost / eval_range)
                x_axis_range = current_cost
                x_axis[x_axis_range].append(current_cost)
                ## save results
                accuracies[x_axis_range].append(accu)
                aucs[x_axis_range].append(auc)
                trial_accu.append([x_axis_range, accu])
                trial_aucs.append([x_axis_range, auc])

            iteration += 1
            # end of budget loop

        tac.append(trial_accu)
        tau.append(trial_aucs)
        #end trial loop
    if args.cost_function not in "uniform":
        accuracies = extrapolate_trials(tac, cost_25=parameters[1][1], step_size=args.step_size)
        aucs = extrapolate_trials(tau, cost_25=parameters[1][1], step_size=args.step_size)

    print("Elapsed time %.3f" % (time.time() - t0))
    print_extrapolated_results(accuracies, aucs)

if __name__ == '__main__':
    main()

