__author__ = 'maru'

import ast
from collections import defaultdict

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from learner.adaptive_lr import LogisticRegressionAdaptive, LogisticRegressionAdaptiveV2
import matplotlib.pyplot as plt

from strategy import base_models


def extrapolate_trials(trials, cost_25=8.2, step_size=10):
    cost_delta = cost_25 * step_size  # Cost of 25 words based on user study

    extrapolated = defaultdict(lambda: [])

    for data in trials:
        # print("j:%s" % j)
        trial_data = np.array(data)
        # print trial_data
        i = 0
        current_c = np.ceil(trial_data[0, 0] / cost_delta) * cost_delta

        # print "starting at %s ending at %s" % (current_c, trial_data.shape[0])

        while i < trial_data.shape[0] - 1:  # while reaching end of rows
            a = trial_data[i]
            a1 = trial_data[i + 1]
            # print("P1:{0}\t{2}\tP2{1}".format(a,a1,current_c))
            if a[0] <= current_c <= a1[0]:
                m = (a1[1] - a[1]) / (a1[0] - a[0]) * (current_c - a[0])
                z = m + a[1]
                extrapolated[current_c].append(z)
                # np.append(extrapolated, [current_c,z])
                # print("{0},z:{1}".format(current_c,z))
                current_c += cost_delta
            if a1[0] < current_c:
                i += 1

    return extrapolated


def parse_parameters(str_parameters):
    parts = str_parameters.split(",")
    params = [float(xi) for xi in parts]
    return params


def parse_parameters_mat(str_parameters):
    params = ast.literal_eval(str_parameters)

    return params


def set_expert_model():
    pass


def set_classifier(cl_name, **kwargs):
    clf = None
    if cl_name in "mnb":
        alpha = 1
        if 'parameter' in kwargs:
            alpha = kwargs['parameter']
        clf = MultinomialNB(alpha=alpha)
    elif cl_name == "lr":
        c = 1
        if 'parameter' in kwargs:
            c = kwargs['parameter']
        clf = LogisticRegression(penalty="l1", C=c)
    elif cl_name == "lrl2":
        c = 1
        if 'parameter' in kwargs:
            c = kwargs['parameter']
        clf = LogisticRegression(penalty="l2", C=c)
    elif cl_name == "lradapt":
        c = 1
        if 'parameter' in kwargs:
            c = kwargs['parameter']
        clf = LogisticRegressionAdaptive(penalty="l1", C=c)
    elif cl_name == "lradaptv2":
        c = 1
        if 'parameter' in kwargs:
            c = kwargs['parameter']
        clf = LogisticRegressionAdaptiveV2(penalty="l1", C=c)
    else:
        raise ValueError("We need a classifier name for the student [lr|mnb]")
    return clf


def format_spent(spent):
    string = ""
    for s in spent:
        string = string + "{0:.2f}, ".format(s)
    return string


def set_cost_model(cost_function, parameters):
    if "uniform" in cost_function:
        # uniform cost model
        cost_model = base_models.BaseCostModel()
    elif "log" in cost_function:
        # linear cost model f(d) = x0*ln(|d|) + x1
        cost_model = base_models.LogCostModel(parameters=parameters)
    elif "linear" in cost_function:
        # linear cost model f(d) = x0*|d| + x1
        cost_model = base_models.FunctionCostModel(parameters=parameters)
    elif "direct" in cost_function:
        # linear cost model f(d) = x0*|d| + x1
        cost_model = base_models.LookUpCostModel(parameters=parameters)
    else:
        raise Exception("We need a defined cost function options [uniform|log|linear|direct]")
    return cost_model


def print_results(x_axis, accuracies, aucs, ts=None):
    # print the cost x-axis
    print
    print "Number of x points %s" % len(x_axis.keys())
    axis_x = sorted(x_axis.keys())
    counts = [len(x_axis[xi]) for xi in axis_x]
    axis_y = [np.mean(x_axis[xi]) for xi in axis_x]
    axis_z = [np.std(x_axis[xi]) for xi in axis_x]
    print "Id\tCost_Mean\tCost_Std"
    for a, b, c, d in zip(axis_x, axis_y, axis_z, counts):
        print "%d\t%0.3f\t%0.3f\t%d" % (a, b, c, d)


    # print the accuracies

    x = sorted(accuracies.keys())
    y = [np.mean(accuracies[xi]) for xi in x]
    z = [np.std(accuracies[xi]) for xi in x]
    w = [np.size(accuracies[xi]) for xi in x]
    print
    print "Cost\tAccu_Mean\tAccu_Std"
    for a, b, c, d in zip(axis_y, y, z, w):
        print "%0.3f\t%0.3f\t%0.3f\t%d" % (a, b, c, d)

    x = sorted(aucs.keys())
    y = [np.mean(aucs[xi]) for xi in x]
    z = [np.std(aucs[xi]) for xi in x]
    print
    print "Cost\tAUC_Mean\tAUC_Std"
    for a, b, c in zip(axis_y, y, z):
        print "%0.3f\t%0.3f\t%0.3f" % (a, b, c)
    if ts is not None:
        x = sorted(ts.keys())
        y = [np.mean(ts[xi]) for xi in x]
        z = [np.std(ts[xi]) for xi in x]
        print
        print "Cost\tTS_Mean\tTS_Std"
        for a, b, c in zip(axis_y, y, z):
            print "%0.3f\t%0.3f\t%0.3f" % (a, b, c)


def plot_performance(x, y, title, xaxis, yaxis):
    plt.clf()
    plt.title(title)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.legend()
    plt.plot(x, y, '--bo')
    # plt.hold(True)
    # x = np.array([min(x), max(x)])
    # y = intercept + slope * x
    # plt.plot(x, y, 'r-')
    plt.savefig('{1}-{0}.pdf'.format(yaxis, title))
    # plt.show()

def oracle_accuracy(oracle, file_name="out", cm=None, num_trials=5):
    # print the cost x-axis
    print
    print "Oracle Accuracy over %s" % len(oracle.keys())

    # print the accuracies

    x = sorted(oracle.keys())
    y = [np.mean(oracle[xi]) for xi in x]
    z = [np.std(oracle[xi]) for xi in x]
    w = [np.size(oracle[xi]) for xi in x]
    step = x[1]- x[0]
    print
    print "Cost\tAccu_Mean\tAccu_Std"
    for a, b, c, d in zip(x, y, z, w):
        print "%0.3f\t%0.3f\t%0.3f\t%d" % (a, 1.*b/step, c, d)
    # plot_performance(x, y, "Oracle Accuracy Performance " + file_name, "Cost", "Oracle Accuracy")
    print_file(x, y, z, "{}-accuracy.txt".format("oracle-"+file_name))

    print "\nCost\tAccu_t1\tAccu_t2\tAccu_t3\tAccu_t4\tAccu_t5"
    trials = [oracle[xi] for xi in x]
    for xi, t in zip(x,trials):
        # print "%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f" % (xi, *t)
        line = [xi]
        line.extend(t)
        print "\t".join(["{0:.3f}".format(i) for i in line])
    # plot_performance(x, y, "Oracle Accuracy Performance " + file_name, "Cost", "Oracle Accuracy")
    print_file2(x, trials, "{}-accuracy.txt".format("oracle-bytrial"+file_name))

    if cm is not None:
        print "\nORACLE CONFUSION MATRIX AVERAGE"
        print "\nCost\tT0\tF1\tF0\tT1\t"
        trials = [cm[xi] for xi in x]
        all_t = []
        for xi, t in zip(x, trials):
            # print "%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f" % (xi,t[0],t[1],t[2],t[3],t[4])
            sum = np.array(t[0], dtype=np.float)
            for ti in t[1:]:
                sum = sum + np.array(ti)
            all_t.append("%0.1f \t" % xi + "\t".join(["{0[0]} {0[1]} {1[0]} {1[1]}".format(v[0],v[1]) for v in t])) ## all trials
            ave = sum/num_trials
            # print ave
            print "%0.3f\t%s" % (xi,"{0[0][0]}\t{0[0][1]}\t{0[1][0]}\t{0[1][1]}".format(ave))
            #["{0[0]} {0[1]} {1[0]} {1[1]}".format(*v) for v in t]
        print_file_cm(x, cm,"{}-cm-accuracy.txt".format("oracle-"+file_name), num_trials=num_trials)
        print "\nAll trials of confusion matrix"
        print "\n".join(all_t)


def print_extrapolated_results(accuracies, aucs, file_name="out"):
    # print the cost x-axis
    print
    print "Number of x points %s" % len(accuracies.keys())

    # print the accuracies

    x = sorted(accuracies.keys())
    y = [np.mean(accuracies[xi]) for xi in x]
    z = [np.std(accuracies[xi]) for xi in x]
    w = [np.size(accuracies[xi]) for xi in x]
    print
    print "Cost\tAccu_Mean\tAccu_Std"
    for a, b, c, d in zip(x, y, z, w):
        print "%0.3f\t%0.3f\t%0.3f\t%d" % (a, b, c, d)

    # plot_performance(x, y, "Accuracy Performance " + file_name, "Cost", "Accuracy")
    print_file(x, y, z, "{}-accuracy.txt".format(file_name))

    x = sorted(aucs.keys())
    y = [np.mean(aucs[xi]) for xi in x]
    z = [np.std(aucs[xi]) for xi in x]
    print
    print "Cost\tAUC_Mean\tAUC_Std"
    for a, b, c in zip(x, y, z):
        print "%0.3f\t%0.3f\t%0.3f" % (a, b, c)

    # plot_performance(x, y, "AUC Performance " + file_name, "Cost", "AUC")
    print_file(x, y, z, "{}-auc.txt".format(file_name))


def print_file(x, y, z, file_name):
    f = open(file_name, "w")
    f.write("COST\tMEAN\tSTDEV\n")
    for a, b, c in zip(x, y, z):
        f.write("{0:.3f}\t{1:.3f}\t{2:.3f}\n".format(a, b, c))
    f.close()


def print_file2(x, trial, file_name):
    f = open(file_name, "w")
    f.write("Cost\tAccu_t1\tAccu_t2\tAccu_t3\tAccu_t4\tAccu_t5\n")
    for a, t in zip(x, trial):
        # print a, t
        tline = "\t".join(["{0:.3f}".format(i) for i in t])
        # f.write("{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}\t{4:.3f}\t{5:.3f}\n".format(a,*t))
        f.write("{0:.3f}\t{1}\n".format(a,tline))
    f.close()


def print_file_cm(x, cm, file_name, num_trials=5.0):
    f = open(file_name, "w")
    f.write("Cost\tT0\tF1\tF0\tT1\n")
    trials = [cm[xi] for xi in x]

    for xi, t in zip(x, trials):
        sum = np.array(t[0], dtype=np.float)
        for ti in t[1:]:
            sum = sum + np.array(ti)
        ave = sum/num_trials
        # print ave
        f.write("%0.3f\t%s\n" % (xi,"{0[0][0]}\t{0[0][1]}\t{0[1][0]}\t{0[1][1]}".format(ave)))
    f.close()



def format_list(list):
    string = ""
    for r in list:
        for c in r:
            string = string + "{0}\t".format(c)
        string = string + ", "
    return string


def print_features(coef, names):
    """ Print sorted list of non-zero features/weights. """
    ### coef = clf.coef_[0]
    ### names = vec.get_feature_names()
    print "*" * 50
    print("Number of Features: %s" % len(names))
    print "\n".join('%s\t%.2f' % (names[j], coef[j]) for j in np.argsort(coef)[::-1] if coef[j] != 0)
    print "*" * 50


def split_data_sentences(data, sent_detector, vct, limit=0):
    sent_train = []
    labels = []
    tokenizer = vct.build_tokenizer()
    print ("Spliting into sentences... Limit:", limit)
    ## Convert the documents into sentences: train
    for t, sentences in zip(data.target, sent_detector.batch_tokenize(data.data)):

        if limit is None:
            sents = [s for s in sentences if len(tokenizer(s)) > 1]
        elif limit > 0:
            sents = [s for s in sentences if len(s.strip()) > limit]
        elif limit == 0:
            sents = [s for s in sentences]
        sent_train.extend(sents)  # at the sentences separately as individual documents
        labels.extend([t] * len(sents))  # Give the label of the document to all its sentences

    return labels, sent_train #, dump


def split_data_first1sentences(data, sent_detector, vct, limit=0):
    '''
    This function is for testing purposes only, do not use in a model
    :param data:
    :param sent_detector:
    :param vct:
    :param limit:
    :return:
    '''
    sent_train = []
    labels = []
    dumped_labels = []
    tokenizer = vct.build_tokenizer()
    dump = []
    f1 = []
    print ("Spliting into sentences... Limit:", limit)
    ## Convert the documents into sentences: train
    for t, sentences in zip(data.target, sent_detector.batch_tokenize(data.data)):
        # sents = [s for s in sentences if len(s) > 1]
        f1.extend([sentences[0]])
        if limit is None:
            sents = [s for s in sentences if len(tokenizer(s)) > 1]
        elif limit > 0:
            sents = [s for s in sentences if len(s.strip()) > limit]
        elif limit == 0:
            sents = [s for s in sentences]
        dump2 = [s for s in sentences if len(s.strip()) <= limit]
        dump.extend(dump2)
        dumped_labels.extend([t] * len(dump2))
        sent_train.extend(sents)  # at the sentences separately as individual documents
        labels.extend([t] * len(sents))  # Give the label of the document to all its sentences
        # dump.extend(dp)

    # print "Removing %s sentences" % len(dump)
    # print "\n".join(dump)

    return labels, sent_train, dump, dumped_labels, f1


def split_into_sentences(data, sent_detector, vct):
    sent_train = []
    tokenizer = vct.build_tokenizer()

    # print ("Spliting into sentences...")
    ## Convert the documents into sentences: train
    for sentences in sent_detector.batch_tokenize(data):
        sents = [s for s in sentences if len(tokenizer(s)) > 1]
        sent_train.extend(sents)  # at the sentences separately as individual documents

    return sent_train


import re


def clean_html(data):
    sent_train = []
    print ("Cleaning text ... ")
    for text in data:
        doc = text.replace("<br>", ". ")
        # doc = doc.replace("\r\n", ". ")
        doc = doc.replace("<br />", ". ")
        doc = re.sub(r"\.", ". ", doc)
        # doc = re.sub(r"x*\.x*", ". ", doc)
        sent_train.extend([doc])

    return sent_train


def get_student(clf, cost_model, sent_clf, sent_token, vct, args):
    from strategy import structured
    cheating = args.cheating

    if args.student in "rnd_sr":
        student = structured.AALRandomThenSR(model=clf, accuracy_model=None, budget=args.budget,
                                                     seed=args.seed, vcn=vct, subpool=250, cost_model=cost_model)
        student.set_sent_score(student.score_max)
        student.fn_utility = student.utility_rnd
    elif args.student in "rnd_srre":
        student = structured.AALRandomThenSR(model=clf, accuracy_model=None, budget=args.budget,
                                                     seed=args.seed, vcn=vct, subpool=250, cost_model=cost_model)
        student.set_sent_score(student.score_max)
        student.fn_utility = student.utility_rnd
    elif args.student in "rnd_srcs":
        student = structured.AALRandomThenSR(model=clf, accuracy_model=None, budget=args.budget,
                                                     seed=args.seed, vcn=vct, subpool=250, cost_model=cost_model)
        student.set_sent_score(student.score_max)
        student.fn_utility = student.utility_rnd
        student.class_sensitive_utility()

    elif args.student in "rnd_srmv":
        student = structured.AALRandomThenSR(model=clf, accuracy_model=None, budget=args.budget,
                                                     seed=args.seed, vcn=vct, subpool=250, cost_model=cost_model)
        student.set_sent_score(student.score_max)
        student.fn_utility = student.utility_rnd
        student.majority_vote_utility()

    elif args.student in "rnd_first1":
        student = structured.AALRandomThenSR(model=clf, accuracy_model=None, budget=args.budget,
                                                     seed=args.seed, vcn=vct, subpool=250, cost_model=cost_model)
        student.set_sent_score(student.score_fk)
        student.fn_utility = student.utility_rnd
    elif args.student in "rnd_rnd":
        student = structured.AALRandomThenSR(model=clf, accuracy_model=None, budget=args.budget,
                                                     seed=args.seed, vcn=vct, subpool=250, cost_model=cost_model)
        student.set_sent_score(student.score_rnd)
        student.fn_utility = student.utility_rnd

    else:
        raise ValueError("Oops! We do not know that anytime strategy. Try again.")

    student.set_score_model(clf)  # student classifier
    student.set_sentence_model(sent_clf)  # cheating part, use and expert in sentences
    student.set_cheating(cheating)
    student.limit = args.limit
    if args.calibrate:
        student.set_sent_score(student.score_p0)
        student.calibratescores = True
    student.sent_detector = sent_token
    return student
