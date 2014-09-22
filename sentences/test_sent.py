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
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from sklearn import metrics
import matplotlib.pyplot as plt
rand = np.random.RandomState(9878654)


#############  COMMAND LINE PARAMETERS ##################
ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawTextHelpFormatter)
ap.add_argument('--train',
                metavar='TRAIN',
                default="imdb",
                help='training data (libSVM format)')


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


args = ap.parse_args()

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

print args
print



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


def main():


    vct = TfidfVectorizer(encoding='ISO-8859-1', min_df=1, max_df=1.0, binary=False, ngram_range=(1, 1),
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

    min_size = 10  # max(10, args.fixk)

    args.fixk = None

    data, vct = load_from_file(args.train, [categories[3]], args.fixk, min_size, vct, raw=True)

    print("Data %s" % args.train)
    print("Data size %s" % len(data.train.data))


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


    #### EXPERT CLASSIFIER: SENTENCES
    print("Training sentence expert")
    labels, sent_train = split_data_sentences(expert_data.sentence.train, sent_detector)

    expert_data.sentence.train.data = sent_train
    expert_data.sentence.train.target = np.array(labels)
    expert_data.sentence.train.bow = vct.transform(expert_data.sentence.train.data)

    sent_clf = linear_model.LogisticRegression(penalty='l1', C=args.expert_penalty)
    sent_clf.fit(expert_data.sentence.train.bow, expert_data.sentence.train.target)

    #### TESTING THE CLASSIFERS

    test_target, test_data = split_data_sentences(data.test,sent_detector)
    test_data_bow = vct.transform(test_data)

    #pred_sent = sent_clf.predict(test_data_bow)
    pred_ora = exp_clf.predict(test_data_bow)
    y_probas = sent_clf.predict_proba(test_data_bow)
    pred_sent = sent_clf.classes_[np.argmax(y_probas, axis=1)]
    ## just based on one class probability
    # order = np.argsort(y_probas[:,0])
    order = np.argsort(y_probas.max(axis=1))
    print "ORACLE\tSENTENCE\tMAX-SENT"
    # for i in order[:500]:
    #     print pred_ora[i],pred_sent[i], y_probas[i,0], test_data[i]
    for i in order[-500:]:
        print pred_ora[i],pred_sent[i], y_probas[i,0], test_data[i]
    print "Accuracy of Sentences Classifier", metrics.accuracy_score(test_target, pred_sent)
    print "Class distribution: %s" % pred_sent.sum()
    print "Size of data: %s" % pred_sent.shape[0]
    sizes = [50, 100, 500, 1000, 2000, 3000, 4000, 20000]
    clf = linear_model.LogisticRegression(penalty='l1', C=1)
    bootstrap = rand.permutation(len(test_data))
    x = []
    y = []
    for s in sizes:
        indices = bootstrap[:s]

        train_x = expert_data.sentence.train.bow[indices[:s]]
        train_y = expert_data.sentence.train.target[indices[:s]]

        clf.fit(train_x, train_y)

        predictions = clf.predict(test_data_bow)
        scores = metrics.accuracy_score(test_target,predictions)
        ## print clf.__class__.__name__
        print "Accuracy {0}: {1}".format(s, scores)
        y.append(scores)
    plt.clf()
    plt.title("Accuracy")
    plt.xlabel("Labels")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.plot(sizes, y, '--bo', label="sent")
    plt.show()

## MAIN FUNCTION
if __name__ == '__main__':
    main()


