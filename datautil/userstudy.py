__author__ = 'mramire8'


import sys, os
sys.path.append(os.path.abspath("."))
import argparse
from sklearn.feature_extraction.text import CountVectorizer
from bunch import Bunch
from datautil.textutils import StemTokenizer
from datautil.load_data import *
from strategy import randomsampling
import codecs
import random


ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawTextHelpFormatter)
ap.add_argument('--train',
                metavar='TRAIN',
                #default="C:/Users/mramire8/Documents/Research/Oracle confidence and Interruption/newExperiment/evidencefit/data/evidence-data-nolabel.txt",
                default="20news",
                help='training data (libSVM format)')
ap.add_argument('--file',
                metavar='FILE',
                #default="C:/Users/mramire8/Documents/Research/Oracle confidence and Interruption/newExperiment/evidencefit/data/evidence-data-nolabel.txt",
                default="file.txt",
                help='output file name')
ap.add_argument('--packsize',
                metavar='packsize',
                type=int,
                default=40,
                help='number of instances per pack')

args = ap.parse_args()

def main():
    vct = CountVectorizer(encoding='ISO-8859-1', min_df=5, max_df=1.0, binary=True, ngram_range=(1, 1),
                          token_pattern='\\b\\w+\\b', tokenizer=StemTokenizer())
    vct_analizer = vct.build_analyzer()
    print("Start loading ...")
    # data fields: data, bow, file_names, target_names, target

    ########## NEWS GROUPS ###############
    # easy to hard. see "Less is More" paper: http://axon.cs.byu.edu/~martinez/classes/678/Presentations/Clawson.pdf
    categories = [['alt.atheism', 'talk.religion.misc'],
                  ['comp.graphics', 'comp.windows.x'],
                  ['comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware'],
                  ['rec.sport.baseball', 'sci.crypt']]

    if "imdb" in args.train:
        ########## IMDB MOVIE REVIEWS ###########
        data = Bunch(load_imdb(args.train, shuffle=True, rnd=2356, vct=vct))  # should brind data as is
    elif "aviation" in args.train:
        raise Exception("We are not ready for that data yet")
    elif "20news" in args.train:
        ########## 20 news groups ######
        data = Bunch(load_20newsgroups(categories=categories[0], vectorizer=vct, min_size=50))  # for testing purposes
    elif "dummy" in args.train:
        ########## DUMMY DATA###########
        data = Bunch(load_dummy("C:/Users/mramire8/Documents/code/python/data/dummy", shuffle=True,rnd=2356,vct=vct))
    else:
        raise Exception("We do not know that dataset")

    print("Data %s" % args.train)
    total = len(data.train.data)
    print("Data size %s" % total)
    #print(data.train.data[0])

    ## prepare pool for the sampling
    pool = Bunch()
    pool.data = data.train.bow.tocsr()   # full words, for training
    pool.target = data.train.target
    pool.predicted = []
    pool.remaining = set(range(pool.data.shape[0]))  # indices of the pool

    bt = randomsampling.BootstrapFromEach(87654321)
    for i in range(7):
        query_index = bt.bootstrap(pool=pool, k=args.packsize)  # get instances from each class
        filename = "{0}-P{1}.txt".format(args.train,i)
        f = codecs.open(filename, 'a+', 'utf-8')
        #print documents in file
        random.shuffle(query_index)
        for di in query_index:
            x = unicode(data.train.data[di].replace("\n","<br>"))
            #y = data.train.target[di]
            y = data.train.target_names[data.train.target[di]]
            #f.write(str(i))
            #f.write("\t")
            #f.write(str(y))
            #f.write("\t")
            #f.write(x)
            #f.write("\n")

        f.close()
        pool.remaining.difference_update(query_index) # remove the used ones


    #print("hi there", file="file.txt")

if __name__ == '__main__':
    main()

