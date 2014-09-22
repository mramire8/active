__author__ = 'mramire8'
__copyright__ = "Copyright 2013, ML Lab"
__version__ = "0.1"
__status__ = "Development"


import numpy as np
import copy
import random


class BaseLearner(object):

    def __init__(self, model=None, cost_model=None, accuracy_model=None, budget=None, seed=1234567, subpool=None):
        self.current_model = model
        self.cost_model = cost_model
        self.accuracy_model = accuracy_model
        self.budget = budget
        self.seed = seed
        self.randgen = np.random.RandomState()
        self.randgen.seed(seed)
        random.seed(4321)
        self.subpool = subpool

    def pick_next(self, pool=None, k=1):
        '''
        pick the indices of instances for the next query, based on query strategy
        :param pool:
        :param k: step size, how many to pick
        :return: indices of chosen instances
        '''
        raise Exception("No strategy was provided")

    def train(self, train_data=None, train_labels=None):
        '''
        Return a new copy of a retrain classifier based on paramters. If no data, return an un-trained classifier
        :param train_data: training data
        :param train_labels: target values
        :return: trained classifier
        '''
        clf = copy.copy(self.current_model)
        if train_data is not None:
            clf.fit(train_data, train_labels)
        return clf

    def predict_cost(self, picked=None):
        return self.cost_model.cost_function(picked)

    def predict_label(self, picked=None):
        pass

    def get_current_model(self):
        return self.current_model
