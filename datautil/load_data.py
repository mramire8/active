__author__ = 'mramire8'

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import load_files
from sklearn.datasets import fetch_20newsgroups
from sklearn.cross_validation import train_test_split, ShuffleSplit
import numpy as np

import xml.etree.ElementTree as ET
from xml.etree.ElementTree import XMLParser
from goose import Goose
from lxml import etree

from bs4 import BeautifulSoup
from boilerpipe.extract import Extractor
from os import listdir
# import bunch
# from bunch import Bunch
from sklearn.datasets import base as bunch
import os
import pickle
import json
from sklearn.utils.validation import check_random_state

if "nt" in os.name:
    IMDB_HOME = 'C:/Users/mramire8/Documents/Research/Oracle confidence and Interruption/dataset/aclImdb/raw-data'
    AVI_HOME  = 'C:/Users/mramire8/Documents/Research/Oracle confidence and Interruption/dataset/sraa/sraa/sraa/partition1/data'
    # AVI_HOME  = 'C:/Users/mramire8/Documents/Research/Oracle confidence and Interruption/dataset/sraa/sraa/sraa/partition1/dummy'
    TWITTER_HOME="C:/Users/mramire8/Documents/Datasets/twitter"
else:
    IMDB_HOME = '/Users/maru/Dataset/aclImdb'
    AVI_HOME  = '/Users/maru/Dataset/aviation/data'
    TWITTER_HOME="/Users/maru/Dataset/twitter"

def keep_header_subject(text, keep_subject=False):
    """
    Given text in "news" format, strip the headers, by removing everything
    before the first blank line.
    """
    _before, _blankline, after = text.partition('\n\n')

    sub = [l for l in _before.split("\n") if "Subject:" in l]
    if keep_subject:
        final = sub[0] + "\n" + after
    else:
        final = after
    return final


def load_20newsgroups(categories=None, vectorizer=CountVectorizer(min_df=5, max_df=1.0, binary=False), min_size=None,
                      fix_k=None, raw=False):
    print "Loading 20 newsgroups dataset for categories:", categories
    data = bunch.Bunch()
    data.train = fetch_20newsgroups(subset='train', categories=categories, remove=('headers','footers', 'quotes'),
                                    shuffle=True, random_state=42)

    data.train.data = [keep_header_subject(text) for text in data.train.data]

    data.test = fetch_20newsgroups(subset='test', categories=categories, remove=('headers','footers', 'quotes'),
                                   shuffle=True, random_state=42)

    data.test.data = [keep_header_subject(text) for text in data.test.data]

    print 'data loaded'

    categories = data.train.target_names

    print "%d categories" % len(categories)
    print
    if not raw:
        data = process_data(data, fix_k, min_size, vectorizer)

    return data


def load_imdb(path, subset="all", shuffle=True, rnd=2356, vct=CountVectorizer(), fix_k=None, min_size=None, raw=False):
    """
    load text files from IMDB movie reviews from folders to memory
    :param path: path of the root directory of the data
    :param subset: what data will be loaded, train or test or all
    :param shuffle:
    :param rnd: ranom seed value
    :param vct: vectorizer
    :return: :raise ValueError:
    """
    #analizer = vct.build_tokenizer()
    #  C:\Users\mramire8\Documents\Research\Oracle confidence and Interruption\dataset\aclImdb\raw-data

    data = bunch.Bunch()

    if subset in ('train', 'test'):
        data[subset] = load_files("{0}/{1}".format(IMDB_HOME, subset), encoding="latin-1", load_content=True,
                                  random_state=rnd)
    elif subset == "all":
        data["train"] = load_files("{0}/{1}".format(IMDB_HOME, "train"), encoding="latin-1", load_content=True,
                                   random_state=rnd)
        data["test"] = load_files("{0}/{1}".format(IMDB_HOME, "test"), encoding="latin-1", load_content=True,
                                  random_state=rnd)
    else:
        raise ValueError(
            "subset can only be 'train', 'test' or 'all', got '%s'" % subset)
    if not raw:
        data = process_data(data, fix_k, min_size, vct)

    return data


def load_aviation(path, subset="all", shuffle=True, rnd=2356, vct=CountVectorizer(), fix_k=None, min_size=None, raw=False, percent=.5):
    """
    load text files from Aviation-auto dataset from folders to memory. It will return a 25-75 percent train test split
    :param path: path of the root directory of the data
    :param subset: what data will be loaded, train or test or all
    :param shuffle:
    :param rnd: random seed value
    :param vct: vectorizer
    :return: :raise ValueError:
    """

    data = bunch.Bunch()

    if subset in ('train', 'test'):
        # data[subset] = load_files("{0}/{1}".format(AVI_HOME, subset), encoding="latin1", load_content=True,
        #                           random_state=rnd)
        raise Exception("We are not ready for train test aviation data yet")
    elif subset == "all":
        data = load_files(AVI_HOME, encoding="latin1", load_content=True,
                                   random_state=rnd)
        data.data = [keep_header_subject(text) for text in data.data]
        # data["test"] = load_files("{0}/{1}".format(AVI_HOME, "test"), encoding="latin1", load_content=True,
        #                           random_state=rnd)
    else:
        raise ValueError(
            "subset can only be 'train', 'test' or 'all', got '%s'" % subset)
    
    # train_x, test_x, train_y, test_y = train_test_split(data.data, data.target, test_size=0.25,
    #     random_state=rnd)

    indices = ShuffleSplit(len(data.data), n_iter=1, test_size=percent, random_state=rnd)
    for train_ind, test_ind in indices:

        data = bunch.Bunch(train=bunch.Bunch(data=[data.data[i] for i in train_ind], target=data.target[train_ind],
                                             target_names=data.target_names),
                              test=bunch.Bunch(data=[data.data[i] for i in test_ind], target=data.target[test_ind],
                                             target_names=data.target_names))
    # if shuffle:
    #     random_state = np.random.RandomState(rnd)
    #     indices = np.arange(data.train.target.shape[0])
    #     random_state.shuffle(indices)
    #     data.train.filenames = data.train.filenames[indices]
    #     data.train.target = data.train.target[indices]
    #     # Use an object array to shuffle: avoids memory copy
    #     data_lst = np.array(data.train.data, dtype=object)
    #     data_lst = data_lst[indices]
    #     data.train.data = data_lst.tolist()

    if not raw:
        data = process_data(data, fix_k, min_size, vct)

    return data



## convert the tweet into a data format of text documents
# from sklearn.datasets.base import Bunch

def preprocess(string, lowercase, collapse_urls, collapse_mentions):
    import re
    if not string:
        return ""
    if lowercase:
        string = string.lower()
#     tokens = []
    if collapse_urls:
        string = re.sub('http\S+', 'THIS_IS_A_URL', string)
    if collapse_mentions:
        string = re.sub('@\S+', 'THIS_IS_A_MENTION', string)
#     if prefix:
#         tokens = ['%s%s' % (prefix, t) for t in tokens]
    return string

def timeline_to_doc(user, *args):
    tweets = []
    for tw in user:
        tweets.append(preprocess(tw['text'], *args))
    return tweets

def user_to_doc(users, *args):
    timeline = []
    user_names = []
    user_id = []

    for user in users:
        timeline.append(timeline_to_doc(user, *args))
        user_names.append(user[0]['user']['name'])
        user_id.append(user[0]['user']['screen_name'])
    return user_id, user_names, timeline


def bunch_users(class1, class2, vct, lowercase, collapse_urls, collapse_mentions, rnd, class_name=None):
    labels = None
    if labels is None:
        labels = [0,1]

    user_id, user_names, timeline = user_to_doc(class1, lowercase, collapse_urls, collapse_mentions)
    user_id2, user_names2, timeline2 = user_to_doc(class2, lowercase, collapse_urls, collapse_mentions)
    target = [labels[0]] * len(user_id)
    user_id.extend(user_id2)
    user_names.extend(user_names2)
    timeline.extend(timeline2)
    target.extend([labels[1]] * len(user_id2))
    user_text = ["######".join(t) for t in timeline]
    data = bunch.Bunch(data=user_text, target=target, user_id=user_id,
                       user_name=user_names, user_timeline=timeline)

    # data = {'data':timeline, 'target':np.array(target), 'user_id':user_id, 'user_name':user_names, 'user_text':user_text}


    random_state = np.random.RandomState(rnd)

    indices = np.arange(len(data.target))
    random_state.shuffle(indices)
    data.target = np.array(data.target)[indices]
    data_lst = np.array(data.data, dtype=object)
    data_lst = data_lst[indices]
    data.data = data_lst.tolist()
    data.user_id = np.array(data.user_id)[indices]
    data.user_name = np.array(data.user_name)[indices]
    data.user_timeline = np.array(data.user_timeline)[indices]
    data.target_names = class_name
    return data

import datetime


def get_date(date_str):

    return datetime.datetime.strptime(date_str.strip('"'), "%a %b %d %H:%M:%S +0000 %Y")


def convert_tweet_2_data(data_path, vct, rnd):
    """
    Convert tweet time lines into dataset
    :param data_path:
    :param vct:
    :return: bunch.Bunch
        Bunch with the data in train and test from twitter bots and human accounts
    """
    good = get_tweets_file(data_path + "/good.json")
    print "Real users %s" % (len(good))

    bots = get_tweets_file(data_path + "/bots.json")
    print "Bot users %s" % (len(bots))

    gds = [g for g in good if get_date(g[0]['created_at']).year > 2013]
    bts = [b for b in bots if get_date(b[0]['created_at']).year > 2013]

    data = bunch_users(gds,bts, vct, True, True, True, rnd, class_name=['good', 'bots'])

    return data


def get_tweets_file(path):

    f = open(path)

    i = 0
    users = []
    data=[]
    last = 0
    for line in f:
        data = line.split("]][[")
        last = len(data)

    for i,tweets in enumerate(data):
            if i == 0:
                t = json.loads(tweets[1:] + "]")
            elif i == (last-1):
                t = json.loads("["+tweets[:-1])
            else:
                t = json.loads("["+tweets+"]")
            users.append(t)

    return users


def load_twitter(path, subset="all", shuffle=True, rnd=2356, vct=CountVectorizer(), fix_k=None, min_size=None, raw=False, percent=.5):
    """
    load text files from twitter data
    :param path: path of the root directory of the data
    :param subset: what data will be loaded, train or test or all
    :param shuffle:
    :param rnd: random seed value
    :param vct: vectorizer
    :return: :raise ValueError:
    """

    data = bunch.Bunch()

    if subset in ('train', 'test'):
        raise Exception("We are not ready for train test aviation data yet")
    elif subset == "all":
        data = convert_tweet_2_data(TWITTER_HOME, vct, rnd)
    else:
        raise ValueError(
            "subset can only be 'train', 'test' or 'all', got '%s'" % subset)

    indices = ShuffleSplit(len(data.data), n_iter=1, test_size=percent, random_state=rnd)
    for train_ind, test_ind in indices:

        data = bunch.Bunch(train=bunch.Bunch(data=[data.data[i] for i in train_ind], target=data.target[train_ind],
                                             target_names=data.target_names),
                           test=bunch.Bunch(data=[data.data[i] for i in test_ind], target=data.target[test_ind],
                                            target_names=data.target_names))
    # if shuffle:
    #     random_state = np.random.RandomState(rnd)
    #     indices = np.arange(data.train.target.shape[0])
    #     random_state.shuffle(indices)
    #     data.train.target = data.train.target[indices]
    #     # Use an object array to shuffle: avoids memory copy
    #     data_lst = np.array(data.train.data, dtype=object)
    #     data_lst = data_lst[indices]
    #     data.train.data = data_lst.tolist()

    if not raw:
        data = process_data(data, fix_k, min_size, vct)

    return data

ARXIV_HOME = 'C:/Users/mramire8/Documents/Datasets/arxiv'
def load_arxiv(path, categories=None, subset="all", shuffle=True, rnd=2356, vct=CountVectorizer(), fix_k=None, min_size=None, raw=False, percent=.5):
    """
    load text files from Aviation-auto dataset from folders to memory. It will return a 25-75 percent train test split
    :param path: path of the root directory of the data
    :param subset: what data will be loaded, train or test or all
    :param shuffle:
    :param rnd: random seed value
    :param vct: vectorizer
    :return: :raise ValueError:
    """

    data = bunch.Bunch()

    if subset in ('train', 'test'):
        raise Exception("We are not ready for train test arxiv data yet")
    elif subset == "all":
        data = load_files(ARXIV_HOME, encoding="latin1", load_content=True,
                                   random_state=rnd, categories=categories)
    else:
        raise ValueError(
            "subset can only be 'train', 'test' or 'all', got '%s'" % subset)

    indices = ShuffleSplit(len(data.data), n_iter=1, test_size=percent, random_state=rnd)

    for train_ind, test_ind in indices:
        data = bunch.Bunch(train=bunch.Bunch(data=[data.data[i] for i in train_ind], target=data.target[train_ind],
                                             target_names=data.target_names),
                           test=bunch.Bunch(data=[data.data[i] for i in test_ind], target=data.target[test_ind],
                                             target_names=data.target_names))
    if not raw:
        data = process_data(data, fix_k, min_size, vct)

    return data


def load_dummy(path, subset="all", shuffle=True, rnd=2356, vct=CountVectorizer(), fix_k=None, min_size=None, raw=False):
    """
    load text files from IMDB movie reviews from folders to memory
    :param path: path of the root directory of the data
    :param subset: what data will be loaded, train or test or all
    :param shuffle:
    :param rnd: random seed value
    :param vct: vectorizer
    :return: :raise ValueError:
    """

    data = bunch.Bunch()
    if subset in ('train', 'test'):
        data[subset] = load_files("{0}/{1}".format(path, subset), charset="latin1", load_content=True, random_state=rnd)
    elif subset == "all":
        data["train"] = load_files("{0}/{1}".format(path, "train"), charset="latin1", load_content=True,
                                   random_state=rnd)
        data["test"] = load_files("{0}/{1}".format(path, "test"), charset="latin1", load_content=True, random_state=rnd)
    else:
        raise ValueError(
            "subset can only be 'train', 'test' or 'all', got '%s'" % subset)
    if not raw:
        data = process_data(data, fix_k, min_size, vct)
    return data


def process_data(data, fix_k, min_size, vct, silent=True):
    # create a fixed k version of the data
    analizer = vct.build_tokenizer()

    fixk = bunch.Bunch()

    fixk.all = data.train.data

    if fix_k is not None:
        # TODO check the size by simple split or by analizer?
        fixk.kwords = [" ".join(analizer(doc)[0:fix_k]) for doc in data.train.data]
        #fixk.kwords = [" ".join(doc.split(" ")[0:fix_k]) for doc in data.train.data]
    else:
        fixk.kwords = data.train.data
    print "Total Documents: %s" % len(fixk.kwords) if silent else ""
    fixk.target = data.train.target
    print "Minimum size: %s" % min_size if silent else ""
    if min_size is not None:
        filtered = [(x, y, z) for x, y, z in zip(data.train.data, fixk.kwords, fixk.target)
                             if len(analizer(x)) >= min_size]

        fixk.all = [x[0] for x in filtered]    # all words
        fixk.kwords = [x[1] for x in filtered]  # k words
        fixk.target = np.array([x[2] for x in filtered], dtype=int)  # targets

    print "Fix k: %s" % fix_k  if silent else ""
    print "Docs left: %s" % len(fixk.all)  if silent else ""

    print "Vectorizing ..."  if silent else ""
    # add the target values
    # add a field for the vectorized data
    data.train.data = fixk.all  # raw documents

    try:
        data.train.bow = vct.transform(fixk.all)  # docs with all the words bow
    except ValueError:
        data.train.bow = vct.fit_transform(fixk.all)  # docs with all the words bow

    data.train.bowk = vct.transform(fixk.kwords)   # docs with k words bow
    data.train.kwords = fixk.kwords   # docs with k words
    data.train.target = fixk.target
    data.test.bow = vct.transform(data.test.data)  # traget

    return data


def load_dataset(name, fixk, categories, vct, min_size, raw=False, percent=.5):
    data = bunch.Bunch()

    if "imdb" in name:
        ########## IMDB MOVIE REVIEWS ###########
        # data = bunch.Bunch(load_imdb(name, shuffle=True, rnd=2356, vct=vct, min_size=min_size, fix_k=fixk, raw=raw))  # should brind data as is
        data = load_imdb(name, shuffle=True, rnd=2356, vct=vct, min_size=min_size,
                         fix_k=fixk, raw=raw)  # should brind data as is
    elif "aviation" in name:
        ########## sraa dataset ######
        data = load_aviation(name, shuffle=True, rnd=2356, vct=vct, min_size=min_size,
                         fix_k=fixk, raw=raw, percent=percent)
    elif "arxiv" in name:
        ########## sraa dataset ######
        data = load_arxiv(name, categories=categories, shuffle=True, rnd=2356, vct=vct, min_size=None,
                         fix_k=None, raw=raw, percent=percent)
    elif "20news" in name:
        ########## 20 news groups ######
        data = load_20newsgroups(categories=categories, vectorizer=vct, min_size=min_size,
                                 fix_k=fixk, raw=raw)
    elif "bgender" in name:
        ########## 20 news groups ######
        data = load_bloggender(name, shuffle=True, rnd=2356, vct=vct, min_size=min_size,
                         fix_k=fixk, raw=raw, percent=percent)
    elif "gmo" in name:
        ########## article pro-con gmo ######
        data = load_gmo(name, shuffle=True, rnd=2356, vct=vct, min_size=min_size,
                         fix_k=fixk, raw=raw, percent=percent)
    elif "evergreen" in name:
        ########## evergreen content blogs  ######
        data = load_evergreen(name, shuffle=True, rnd=2356, vct=vct, min_size=min_size,
                         fix_k=fixk, raw=raw, percent=percent)
    elif "pan" in name:
        ########## author gender classification from blogs ######
        data = load_blogpan(name, shuffle=True, rnd=2356, vct=vct, min_size=min_size,
                         fix_k=fixk, raw=raw, percent=percent)
    elif "webkb" in name:
        # raise Exception("We are not ready for that data yet")
        data = load_webkb(name, categories=categories, shuffle=True, rnd=2356, vct=vct, min_size=min_size,
                         fix_k=fixk, raw=raw, percent=percent)
    elif "biocreative" in name:
        # raise Exception("We are not ready for that data yet")
        data = load_biocreative(name, shuffle=True, rnd=2356, vct=vct, min_size=min_size,
                         fix_k=fixk, raw=raw, percent=percent)
    elif "twitter" in name:
        # raise Exception("We are not ready for that data yet")
        data = load_twitter(name, shuffle=True, rnd=2356, vct=vct, min_size=min_size,
                         fix_k=fixk, raw=raw, percent=percent)
    elif "dummy" in name:
        ########## DUMMY DATA###########
        data = load_dummy("C:/Users/mramire8/Documents/code/python/data/dummy", shuffle=True, rnd=2356,
                          vct=vct, min_size=0, fix_k=fixk, raw=raw)
    else:
        raise Exception("We do not know that dataset")

    return data


def load_dictionary(datafile=""):
    f = open(datafile)
    with f:
        line = f.readlines()

    line = [l.strip() for l in line]
    return line


def load_documents(datafile="", header=True):
    f = open(datafile)
    feature_names = []
    if header:
        feature_names = f.readline().split()  # skip the header
    # print ('HEADER NAMES: \n %s' % feature_names)
    docs = []
    with f:
        # uniqueid	truelabel	text	words 	seenwords	avgtime
        line = f.readlines()
    docs = [l.strip().split('\t') for l in line]
    #b = [ai for ai in a if ai % 2 == 0]  # another way to do filtering when loading the datasets
    return docs, feature_names


def load_from_file(train, categories, fixk, min_size, vct, raw=True):
    fixk_saved = "{0}-MIN{1}.p".format(train, min_size)
    try:
        print "Loading existing file... %s " % train
        fixk_file = open(fixk_saved, "rb")
        data = pickle.load(fixk_file)
        fixk_file.close()
        # vectorizer = open("{0}vectorizer.p".format(train), "rb")
        # vct = pickle.load(vectorizer)
        # vectorizer.close()
    except (IOError, ValueError):
        print "Loading from scratch..."
        data = load_dataset(train, fixk, categories[0], vct, min_size, percent=.5)
        fixk_file = open(fixk_saved, "wb")
        pickle.dump(data, fixk_file)
        fixk_file.close()
        # vectorizer = open("{0}vectorizer.p".format(train), "wb")
        # pickle.dump(vct, vectorizer)
        # vectorizer.close()

    return data, vct


BLOGGEN_HOME = "C:/Users/mramire8/Documents/Datasets/textclassification/raw data/author-profiling-gender/gender/blog-gender-dataset.tsv"
def load_bloggender(path, subset="all", shuffle=True, rnd=2356, vct=CountVectorizer(), fix_k=None, min_size=None, raw=False, percent=.5):
    import csv
    docs = []
    labels = []
    clases = ['F', 'M']
    with open(BLOGGEN_HOME, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='"')
        for row in reader:
            if len(row[0])>0 and len(row[1])>0:
                docs.append(row[0])
                labels.append(clases.index(row[1].strip().upper()))
    data = bunch.Bunch()
    data.data = docs
    data.target=np.array(labels)


    indices = ShuffleSplit(len(data.data), n_iter=1, test_size=percent, random_state=rnd)
    for train_ind, test_ind in indices:

        data = bunch.Bunch(train=bunch.Bunch(data=[data.data[i] for i in train_ind], target=data.target[train_ind], target_names=clases),
                              test=bunch.Bunch(data=[data.data[i] for i in test_ind], target=data.target[test_ind], target_names=clases))

    if not raw:
        data = process_data(data, fix_k, min_size, vct)

    return data



PAN13_HOME = "C:/Users/mramire8/Documents/Datasets/textclassification/raw data/author-profiling-gender/gender/blogs/blogs"
def load_pan13(path, subset="all", shuffle=True, rnd=2356, vct=CountVectorizer(), fix_k=None, min_size=None, raw=False, percent=.5):

    data = bunch.Bunch()

    if subset in ('train', 'test'):
        # data[subset] = load_files("{0}/{1}".format(AVI_HOME, subset), encoding="latin1", load_content=True,
        #                           random_state=rnd)
        raise Exception("We are not ready for train test aviation data yet")
    elif subset == "all":
        data = load_files(PAN13_HOME, encoding="latin1", load_content=True,
                                   random_state=rnd)
        data.data = [keep_header_subject(text) for text in data.data]
    else:
        raise ValueError(
            "subset can only be 'train', 'test' or 'all', got '%s'" % subset)

    for iDoc in data.data:
        pass

    if not raw:
        data = process_data(data, fix_k, min_size, vct)

    return data


EVERGREEN_HOME = "C:/Users/mramire8/Documents/Datasets/textclassification/raw data/evergreen"
def get_content(url):
    g = Goose({'enable_image_fetching':False})
    article = g.extract(url=url)
    # article = g.extract(raw_html=url)
    text = "{0} {1}".format(article.title, article.cleaned_text)
    return text


def read_evergreenjs(filename):
    import csv
    docs = []
    labels = []
    # i =0
    ## EVERGREEN = 0, NON-EVERGREEN=1

    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='"')
        header = None
        for row in reader:
            # print row
            if header is None:
                header = row
            else:

                ## Original boiler plate the text has not punctuation
                # content = json.loads(row[header.index('boilerplate')])
                # content['title']
                # if len(content)>1 and content['body'] is not None:
                #     docs.append(content['body'])
                #     labels.append(int(row[header.index('label')]))

                ## EXTRACT BODY-ISH OF THE HTML FILE
                url = "{0}/raw_content/{1}.".format(EVERGREEN_HOME, row[header.index('urlid')])
                text = open(url).read()
                soup = BeautifulSoup(text)
                # print "*"*50
                # remove non-text tags
                for tag in ['script', 'style', 'a', 'img']:
                    for el in soup.find_all(tag):
                        el.extract()

                extractor = Extractor(extractor='ArticleExtractor', html=unicode(soup.get_text()))

                ## ADD CONTENT AND LABEL TO THE LIST
                docs.append(extractor.getText())
                # docs.append(get_content(url))
                labels.append(int(row[header.index('label')]))
            # print i
            # i+=1
    return docs, labels


def load_evergreen(path, subset="all", shuffle=True, rnd=2356, vct=CountVectorizer(), fix_k=None, min_size=None, raw=False, percent=.5):
    docs = []
    labels = []
    ## EVERGREEN = 0, NON-EVERGREEN=1
    clases = ['EVERGREEN', 'SEASONAL']
    filename = "{0}/{1}".format(EVERGREEN_HOME, "train.tsv")
    docs, labels = read_evergreenjs(filename)
    # filename = "{0}/{1}".format(EVERGREEN_HOME, "test.tsv")
    # docst, labelst = read_evergreenjs(filename)
    # data = bunch.Bunch(train=bunch.Bunch(data=docs, target=np.array(labels)),
    #                    test=bunch.Bunch(data=docst, target=np.array(labelst)))
    data = bunch.Bunch()
    data.data = docs
    data.target=np.array(labels)


    indices = ShuffleSplit(len(data.data), n_iter=1, test_size=percent, random_state=rnd)
    for train_ind, test_ind in indices:

        data = bunch.Bunch(train=bunch.Bunch(data=[data.data[i] for i in train_ind], target=data.target[train_ind], target_names=clases),
                              test=bunch.Bunch(data=[data.data[i] for i in test_ind], target=data.target[test_ind], target_names=clases))

    if not raw:
        data = process_data(data, fix_k, min_size, vct)

    return data


def create_gmo(file):
    docs, _ = load_documents(file, header=False)
    content = []
    iDoc = []
    for line in docs:
        text = line[0]
        if "Document Number:" in text and len(iDoc)>0:
            content.append("\n".join(iDoc))
            iDoc = []
        iDoc.append(text)
    return content


def load_gmo(path, subset="all", shuffle=True, rnd=2356, vct=CountVectorizer(), fix_k=None, min_size=None, raw=False, percent=.5):
    GMO_HOME = "C:/Users/mramire8/Documents/Datasets/textclassification/raw data/gmo-hedging/GMOHedging_v1.0/gmo-anti/{}"
    parts = create_gmo(GMO_HOME.format("anti_GMO"))
    labels = np.zeros(len(parts))
    parts.extend(create_gmo(GMO_HOME.format("pro_GMO")))
    labels = np.append(labels, np.ones(len(parts)-len(labels)))
    data = bunch.Bunch()
    data.data = parts
    data.target = labels
    indices = ShuffleSplit(len(data.data), n_iter=1, test_size=percent, random_state=rnd)
    for train_ind, test_ind in indices:

        data = bunch.Bunch(train=bunch.Bunch(data=[data.data[i] for i in train_ind], target=data.target[train_ind]),
                              test=bunch.Bunch(data=[data.data[i] for i in test_ind], target=data.target[test_ind]))

    if not raw:
        data = process_data(data, fix_k, min_size, vct)

    return data


def clean_xml_pan(xml_text, parser=None):


    text = ""
    # try:
    root = ET.fromstring(xml_text, parser=parser)
    for post in root.findall("post"):
        text += "\n" + post.text.strip()
    # except Exception:
    #     print xml_text

    return text


def load_blogpan(path, subset="all", shuffle=True, rnd=2356, vct=CountVectorizer(), fix_k=None, min_size=None, raw=False, percent=.5):
    """
    load text files from author gender profiling dataset from folders to memory.
    :param path: path of the root directory of the data
    :param subset: what data will be loaded, train or test or all
    :param shuffle:
    :param rnd: random seed value
    :param vct: vectorizer
    :return: :raise ValueError:
    """
    PAN13_HOME = "C:/Users/mramire8/Documents/Datasets/textclassification/raw data/author-profiling-gender/gender-profiling/blogs/blogs"

    data = bunch.Bunch()

    if subset in ('train', 'test'):
        # data[subset] = load_files("{0}/{1}".format(AVI_HOME, subset), encoding="latin1", load_content=True,
        #                           random_state=rnd)
        raise Exception("We are not ready for train test aviation data yet")
    elif subset == "all":
        data = load_files(PAN13_HOME, encoding="latin1", load_content=True,
                                   random_state=rnd)
        # parser = XMLParser(encoding="latin-1", recover=True)
        parser = etree.XMLParser(recover=True)
        data.data = [clean_xml_pan(text, parser=parser) for text in data.data]
        # data["test"] = load_files("{0}/{1}".format(AVI_HOME, "test"), encoding="latin1", load_content=True,
        #                           random_state=rnd)
    else:
        raise ValueError(
            "subset can only be 'train', 'test' or 'all', got '%s'" % subset)

    # train_x, test_x, train_y, test_y = train_test_split(data.data, data.target, test_size=0.25,
    #     random_state=rnd)

    indices = ShuffleSplit(len(data.data), n_iter=1, test_size=percent,  random_state=rnd)
    for train_ind, test_ind in indices:

        data = bunch.Bunch(train=bunch.Bunch(data=[data.data[i] for i in train_ind], target=data.target[train_ind]),
                              test=bunch.Bunch(data=[data.data[i] for i in test_ind], target=data.target[test_ind]))
    # if shuffle:
    #     random_state = np.random.RandomState(rnd)
    #     indices = np.arange(data.train.target.shape[0])
    #     random_state.shuffle(indices)
    #     data.train.filenames = data.train.filenames[indices]
    #     data.train.target = data.train.target[indices]
    #     # Use an object array to shuffle: avoids memory copy
    #     data_lst = np.array(data.train.data, dtype=object)
    #     data_lst = data_lst[indices]
    #     data.train.data = data_lst.tolist()

    if not raw:
        data = process_data(data, fix_k, min_size, vct)

    return data

# from sklearn.datasets import fetch_mldata

def load_biocreative(path, subset="all", shuffle=True, rnd=2356, vct=CountVectorizer(), fix_k=None, min_size=None, raw=False, percent=.5):
    # target = []
    # target_names = []
    # filenames = []
    #
    # folders = [f for f in sorted(listdir(container_path))
    #            if isdir(join(container_path, f))]
    #
    # if categories is not None:
    #     folders = [f for f in folders if f in categories]
    #
    # for label, folder in enumerate(folders):
    #     target_names.append(folder)
    #     folder_path = join(container_path, folder)
    #     documents = [join(folder_path, d)
    #                  for d in sorted(listdir(folder_path))]
    #     target.extend(len(documents) * [label])
    #     filenames.extend(documents)
    #
    # # convert to array for fancy indexing
    # filenames = np.array(filenames)
    # target = np.array(target)
    #
    # if shuffle:
    #     random_state = check_random_state(random_state)
    #     indices = np.arange(filenames.shape[0])
    #     random_state.shuffle(indices)
    #     filenames = filenames[indices]
    #     target = target[indices]
    #
    # if load_content:
    #     data = [open(filename, 'rb').read() for filename in filenames]
    #     if encoding is not None:
    #         data = [d.decode(encoding, decode_error) for d in data]
    #     return Bunch(data=data,
    #                  filenames=filenames,
    #                  target_names=target_names,
    #                  target=target,
    #                  DESCR=description)
    #
    # return Bunch(filenames=filenames,
    #              target_names=target_names,
    #              target=target,
    #              DESCR=description)
    raise Exception("We are not ready for that data yet")


WEBKB_HOME='C:/Users/mramire8/Documents/Datasets/webkb/webkb'


def clean_html_text(html_text):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_text)
    return soup.get_text()


def get_sub_filenames(input_dir):

    names = []
    for path, subdirs, files in os.walk(input_dir):
        for filename in files:
            names.append(os.path.join(path, filename))
    return names


def load_files_sub(container_path, description=None, categories=None,
               load_content=True, shuffle=True, encoding=None,
               charset=None, charset_error=None,
               decode_error='strict', random_state=0):

    """
    Adapted from load_file of sklearn, this loads files from directories and subdirectories
    :param container_path:
    :param description:
    :param categories:
    :param load_content:
    :param shuffle:
    :param encoding:
    :param charset:
    :param charset_error:
    :param decode_error:
    :param random_state:
    :return:
    """
    from os.path import isdir
    from os import listdir
    from os.path import join

    target = []
    target_names = []
    filenames = []

    ## get the folders
    folders = [f for f in sorted(listdir(container_path))
               if isdir(join(container_path, f))]

    # get categories

    if categories is not None:
        folders = [f for f in folders if f in categories]

    for label, folder in enumerate(folders):
        target_names.append(folder)
        folder_path = join(container_path, folder)
        ## get all files from subfolders
        documents = [join(folder_path, d)
                     for d in sorted(get_sub_filenames(folder_path))]
        target.extend(len(documents) * [label])
        filenames.extend(documents)

    # convert to array for fancy indexing
    filenames = np.array(filenames)
    target = np.array(target)

    if shuffle:
        random_state = check_random_state(random_state)
        indices = np.arange(filenames.shape[0])
        random_state.shuffle(indices)
        filenames = filenames[indices]
        target = target[indices]

    if load_content:
        data = [open(filename, 'rb').read() for filename in filenames]
        if encoding is not None:
            data = [d.decode(encoding, decode_error) for d in data]
        return bunch.Bunch(data=data,
                     filenames=filenames,
                     target_names=target_names,
                     target=target,
                     DESCR=description)

    return bunch.Bunch(filenames=filenames,
                 target_names=target_names,
                 target=target,
                 DESCR=description)


def load_webkb(path, categories=None, subset="all", shuffle=True, rnd=2356, vct=CountVectorizer(), fix_k=None, min_size=None, raw=False, percent=.5):
    """
    Read and process data from  webkb dataset. Documents are files in html format
    :param path: loacation of the root directory of the data
    :param categories: categories to load COURSE, DEPARTMENT, FACULTY, OTHER, PROJECT, STAFF, STUDENT
    :param subset: --unused at the moment --
    :param shuffle: --unused at the moment --
    :param rnd: random seed value
    :param vct: vectorizer for feature vector representation
    :param fix_k: truncate data a the k-th word, none if including all words
    :param min_size: minimum size document acceptable to load
    :param raw: return data without feature vectores
    :param percent: Percentage to split train-test dataset e.g. .25 will produce a 75% training, 25% test
    :return: Bunch :
        .train.data     text of data
        .train.target   target vector
        .train.bow      feature vector of full documents
        .train.bowk     feature of k-words documents
        .train.kwords   text of k-word documents
        .test.data      test text data
        .test.target    test target vector
        .text.bow       feature vector of test documents
    :raise ValueError:
    """
    data = bunch.Bunch()

    if subset in ('train', 'test'):
        # data[subset] = load_files("{0}/{1}".format(AVI_HOME, subset), encoding="latin1", load_content=True,
        #                           random_state=rnd)
        raise Exception("We are not ready for train test webkb data yet")
    elif subset == "all":
        data = load_files_sub(WEBKB_HOME, encoding="latin1", load_content=True, random_state=rnd)
        data.data = [clean_html_text(text) for text in data.data]

    else:
        raise ValueError(
            "subset can only be 'train', 'test' or 'all', got '%s'" % subset)

    if categories is not None:
        labels = [(data.target_names.index(cat), cat) for cat in categories]
        # Sort the categories to have the ordering of the labels
        labels.sort()
        labels, categories = zip(*labels)
        mask = np.in1d(data.target, labels)
        data.filenames = data.filenames[mask]
        data.target = data.target[mask]
        # searchsorted to have continuous labels
        data.target = np.searchsorted(labels, data.target)
        data.target_names = list(categories)
        # Use an object array to shuffle: avoids memory copy
        data_lst = np.array(data.data, dtype=object)
        data_lst = data_lst[mask]
        data.data = data_lst.tolist()

    if shuffle:
        random_state = check_random_state(rnd)
        indices = np.arange(data.target.shape[0])
        random_state.shuffle(indices)
        data.filenames = data.filenames[indices]
        data.target = data.target[indices]
        # Use an object array to shuffle: avoids memory copy
        data_lst = np.array(data.data, dtype=object)
        data_lst = data_lst[indices]
        data.data = data_lst.tolist()

    indices = ShuffleSplit(len(data.data), n_iter=1, test_size=percent,  random_state=rnd)
    for train_ind, test_ind in indices:

        data = bunch.Bunch(train=bunch.Bunch(data=[data.data[i] for i in train_ind], target=data.target[train_ind],
                                             target_names=data.target_names),
                           test=bunch.Bunch(data=[data.data[i] for i in test_ind], target=data.target[test_ind],
                                            target_names=data.target_names))

    if not raw:
        data = process_data(data, fix_k, min_size, vct)

    return data


def split_data(data, splits=2.0, rnd=987654321):
    """

    :param data: is a bunch with data.data and data.target
    :param splits: number of splits (translates into percentages
    :param rnd: random number
    :return: two bunches with the split
    """
    percent = 1.0 / splits
    indices = ShuffleSplit(len(data.data), n_iter=1, test_size=percent, random_state=rnd)
    part1 = bunch.Bunch()
    part2 = bunch.Bunch()
    for train_ind, test_ind in indices:
        part1 = bunch.Bunch(train=bunch.Bunch(data=[data.data[i] for i in train_ind], target=data.target[train_ind]))#, target_names=data.target_names))
        part2 = bunch.Bunch(train=bunch.Bunch(data=[data.data[i] for i in test_ind], target=data.target[test_ind])) #, target_names=data.target_names))

    return part1, part2