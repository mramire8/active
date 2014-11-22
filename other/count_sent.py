# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Sentence Distribution
# 
# Sentence distribution for IMDB dataset 
# 
# * How many sentences per document in general
# * How many sentences per document per class

# <codecell>

__author__ = 'mramire8'
__copyright__ = "Copyright 2013, ML Lab"
__version__ = "0.1"
__status__ = "Development"

import sys
import os

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("../"))
sys.path.append(os.path.abspath("../experiment/"))


from experiment.experiment_utils import clean_html
import numpy as np
from datautil.load_data import load_from_file

from collections import  Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from scipy.sparse import vstack

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.style.use('bmh')


def print_features(coef, names):
    """ Print sorted list of non-zero features/weights. """
    print "\n".join('%s/%.2f' % (names[j], coef[j]) for j in np.argsort(coef)[::-1] if coef[j] != 0)


def get_data(train, cats, vct, raw):

    min_size = None
    fixk=None

    data, vct2 = load_from_file(train, cats, fixk, min_size, vct, raw=raw)

    print("Data %s" % train)
    print("Data size %s" % len(data.train.data))

    ### SENTENCE TRANSFORMATION
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

    ## delete <br> to "." to recognize as end of sentence
    data.train.data = clean_html(data.train.data)
    data.test.data = clean_html(data.test.data)

    print("Train:{}, Test:{}, {}".format(len(data.train.data), len(data.test.data), data.test.target.shape[0]))

    ## convert document to matrix
    data.train.bow = vct.fit_transform(data.train.data)
    data.test.bow = vct.transform(data.test.data)
    return data


####################### MAIN ####################

def split_data_sentences(data, sent_detector, vct, limit=0):
    sent_train = []
    labels = []
    tokenizer = vct.build_tokenizer()
    size_per_doc = []
    print ("Spliting into sentences... Limit:", limit)
    ## Convert the documents into sentences: train
    for t, sentences in zip(data.train.target, sent_detector.batch_tokenize(data.train.data)):

        if limit is None:
            sents = [s for s in sentences if len(tokenizer(s)) > 1]
        elif limit > 0:
            sents = [s for s in sentences if len(s.strip()) > limit]
            size_per_doc.append(len(sentences))
        elif limit == 0:
            sents = [s for s in sentences]
        sent_train.extend(sents)  # at the sentences separately as individual documents
        labels.extend([t] * len(sents))  # Give the label of the document to all its sentences

    return labels, sent_train, size_per_doc


def distribution_per_sentence(data_name, sent_per_doc):
   
    print "Sentence distribution"

    sent_per_doc = np.array(sent_per_doc)
    sents = Counter(sent_per_doc)
    plt.figure(figsize=(9,4.5))
    n, bins, patches = plt.hist(sents.keys(), weights=sents.values(),  bins=range(100))
    plt.title("Document Number of Sentence Distribution {0} (mean={1:.2f}) N={2}".format(data_name, sent_per_doc.mean(),
                                                                                         len(sent_per_doc)),
              fontsize=14, family='Arial')
    plt.xlabel("Number of sentences")
    plt.ylabel("Frequency")
    plt.show()
#     for b in n:
#         print "bin: %s" % (n)
    
    
rand = np.random.mtrand.RandomState(987654)

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

vct = TfidfVectorizer(encoding='latin1', min_df=5, max_df=1.0, binary=False, ngram_range=(1, 1),
                  token_pattern='\\b\\w+\\b')  #, tokenizer=StemTokenizer())

categories = [['alt.atheism', 'talk.religion.misc'],
              ['comp.graphics', 'comp.windows.x'],
              ['comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware'],
              ['rec.sport.baseball', 'sci.crypt']]



# <codecell>

## Get training data
data = get_data("imdb", [categories[0]], vct, True)  # expert: classifier, data contains train and test

# <codecell>

#Get sentences, targets, and number of sentences per document
X, y, sent_per_doc = split_data_sentences(data, sent_detector, vct, limit=2)

# <markdowncell>

# ## Sentence Distribution per Document
# 
# The following graph is the distribution of number of sentences per document in the training split fo the data. Size of the bin is one and there are 100 bins. 
# 
# **Note:**
# 
# Most of the data has less than 20 sentences (average=18.7). Few documents have one, two and three sentences (~500 documents)

# <codecell>

distribution_per_sentence("imdb", sent_per_doc)

# <markdowncell>

# ## Sentence Distribution per Class
# 
# The graph shows the distributions of number of sentences per document per class (true class)
# 
# **Note:**
# 
# The documents of class y=1 seem to have less number of sentences ($\bar{x_0}=18.9 > \bar{x_1}=17.5 $), however the difference is only two sentences at the most. 

# <codecell>

def sent_dist_per_class(data_name, sents, target):
    c0 = sents[target ==0]
    c1 = sents[target ==1]
    plt.figure(figsize=(9,4.5))
    n, bins, patches = plt.hist([c0,c1], bins=range(100), label=['y=0','y=1'],  histtype='step', fill=True, alpha=0.5)
    plt.title("Document Number of Sentence per class {0} (mean={1:.2f}, {2:.2f}) (N={3},{4})".format(data_name, c0.mean(), c1.mean(),
                                                                                         len(c0), len(c1)),
              fontsize=14, family='Arial')
    plt.xlabel("Number of sentences")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    

# <codecell>

sent_dist_per_class("imdb", np.array(sent_per_doc), data.train.target)

# <codecell>


