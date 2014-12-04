__author__ = 'mramire8'
from nltk import word_tokenize,RegexpTokenizer
from nltk.stem import PorterStemmer


class StemTokenizer(object):

    def __init__(self):
        self.wnl = PorterStemmer()
        self.mytokenizer = RegexpTokenizer('\\b\\w+\\b')

    def __call__(self, doc):
        #return [self.wnl.stem(t) for t in word_tokenize(doc)]
        return [self.wnl.stem(t) for t in self.mytokenizer.tokenize(doc)]


class TwitterSentenceTokenizer(object):
    def __init__(self):
        pass

    def batch_tokenize(self, twitter_objs):
        return [sent.split("######") for sent in twitter_objs]

    def tokenize(self, twitter_objs):
        return twitter_objs.split("######")

    def __call__(self, twitter_objs):
        return twitter_objs

    def __str__(self):
        return self.__class__.__name__
