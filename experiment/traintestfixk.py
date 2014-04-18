__author__ = 'maru'
__copyright__ = "Copyright 2013, ML Lab"
__version__ = "0.1"
__status__ = "Development"

import sys
import os

sys.path.append(os.path.abspath("."))

from experiment_utils import print_results,  parse_parameters_mat, set_cost_model
import argparse
import numpy as np
from sklearn.datasets.base import Bunch
from datautil.load_data import load_dataset
from sklearn import linear_model
import time

from sklearn import metrics
from collections import defaultdict

from datautil.textutils import StemTokenizer
from strategy import randomsampling
from expert import baseexpert
from sklearn.feature_extraction.text import CountVectorizer

#############  COMMAND LINE PARAMETERS ##################
ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawTextHelpFormatter)
ap.add_argument('--train',
                metavar='TRAIN',
                default="20news",
                help='training data (libSVM format)')

ap.add_argument('--neutral_threshold',
                metavar='NEUTRAL',
                type=float,
                default=.3,
                help='neutrality threshold of uncertainty')

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
                help='cost function of the x-axis [uniform|log|linear]')

ap.add_argument('--fixk',
                metavar='FIXK',
                type=int,
                default=10,
                help='fixed k number of words')

ap.add_argument('--maxiter',
                metavar='MAXITER',
                type=int,
                default=2000,
                help='Max number of iterations')

ap.add_argument('--seed',
                metavar='SEED',
                type=int,
                default=8765432,
                help='Max number of iterations')

args = ap.parse_args()
rand = np.random.mtrand.RandomState(args.seed)

print args
print


####################### MAIN ####################
def main():
    accuracies = defaultdict(lambda: [])

    aucs = defaultdict(lambda: [])

    x_axis = defaultdict(lambda: [])

    vct = CountVectorizer(encoding='ISO-8859-1', min_df=5, max_df=1.0, binary=True, ngram_range=(1, 3),
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

    min_size = max(100, args.fixk)

    data = load_dataset(args.train, args.fixk, categories[0], vct, min_size)

    print("Data %s" % args.train)
    print("Data size %s" % len(data.train.data))

    parameters = parse_parameters_mat("[1,0]")

    print "Cost Parameters %s" % parameters

    cost_model = set_cost_model("uniform", parameters=parameters)
    print "\nCost Model: %s" % cost_model.__class__.__name__


    #### STUDENT CLASSIFIER
    clf = linear_model.LogisticRegression(penalty="l1", C=1)
    print "\nStudent Classifier: %s" % clf

    #### EXPERT CLASSIFIER

    exp_clf = linear_model.LogisticRegression(penalty='l1', C=.3)
    exp_clf.fit(data.test.bow, data.test.target)
    expert = baseexpert.NeutralityExpert(exp_clf, threshold=args.neutral_threshold,
                                         cost_function=cost_model.cost_function)
    print "\nExpert: %s " % expert

    #### ACTIVE LEARNING SETTINGS
    step_size = args.step_size
    bootstrap_size = args.bootstrap
    evaluation_points = 200

    print("\nExperiment: step={0}, BT={1}, plot points={2}, fixk:{3}, minsize:{4}".format(step_size, bootstrap_size,
                                                                                          evaluation_points, args.fixk,
                                                                                          min_size))

    t0 = time.time()
    ### experiment starts
    for t in range(args.trials):
        print "*" * 60
        print "Trial: %s" % t

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

        bootstrapped = False

        current_cost = 0
        iteration = 0
        while 0 < student.budget and len(pool.remaining) > step_size and iteration <= args.maxiter:

            if not bootstrapped:
                ## random from each bootstrap
                bt = randomsampling.BootstrapFromEach(t * 10)

                query_index = bt.bootstrap(pool=pool, k=bootstrap_size)
                bootstrapped = True
                print "Bootstrap: %s " % bt.__class__.__name__
                print
            else:
                query_index = student.pick_next(pool=pool, k=step_size)

            query = pool.fixk[query_index]  # query with k words

            query_size = [len(x.split('\\b\\w+\\b')) for x in pool.kwords[query_index]]

            #if query_size[0] >50:
            #    print "*** %s" % pool.kwords[query_index]

            ground_truth = pool.target[query_index]
            #labels, spent = expert.label(unlabeled=query, target=ground_truth)
            if iteration == 0: ## bootstrap uses ground truth
                labels = ground_truth
            else:
                #labels = expert.label_instances(query, ground_truth)
                labels = expert.label_instances(query, ground_truth)
                #spent = expert.estimate_instances(pool.kwords[query_index])
            spent = expert.estimate_instances(query_size)

            ### accumulate the cost of the query
            query_cost = np.array(spent).sum()
            current_cost += query_cost

            ## add data recent acquired to train
            useful_answers = np.array([[x, y] for x, y in zip(query_index, labels) if y is not None])

            # train_indices.extend(query_index)
            if useful_answers.shape[0] != 0:
                train_indices.extend(useful_answers[:, 0])

                # add labels to training
                train_x = pool.data[train_indices]  ## train with all the words

                # update labels with the expert labels
                #train_y = pool.target[train_indices]
                train_y.extend(useful_answers[:, 1])

            if train_x.shape[0] != len(train_y):
                raise Exception("Training data corrupted!")

            # remove labels from pool
            pool.remaining.difference_update(query_index)

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
                                                                                              current_cost, spent))

            ## the results should be based on the cost of the labeling
            if iteration > 0: # bootstrap iteration

                student.budget -= query_cost ## Bootstrap doesn't count

                x_axis_range = current_cost
                x_axis[x_axis_range].append(current_cost)
                ## save results
                accuracies[x_axis_range].append(accu)
                aucs[x_axis_range].append(auc)

            iteration += 1

    print("Elapsed time %.3f" % (time.time() - t0))
    print_results(x_axis, accuracies, aucs)


if __name__ == '__main__':
    main()

