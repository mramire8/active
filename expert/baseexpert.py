__author__ = 'mramire8'

import numpy as np
import sys
import os
sys.path.append(os.path.abspath('../strategy'))
from strategy.base_models import *
from sklearn.linear_model import LogisticRegression


class BaseExpert(object):

    def __init__(self, model=None, seed=1234567):
        self.expert = model
        self.seed = seed
        #self.estimate_cost = self.cost
        #pass

    def label_instance(self, unlabeled, target=None):
        '''
        given an instance  label, provide label
        :param unlabeled: index of instances to label
        :return: array of labels
        '''
        #use model to return the lables?
        raise Exception("no model was provided!")

    def estimate_cost(self, instance=None):
        raise Exception("no cost function provided!")

    def label(self, unlabeled=None, target=None):
        # TODO check for individual instances instead of k
        #labels = [self.label_instance(unlabeled=instance, target=y) for instance,y in zip(unlabeled,target)]
        #label_cost = [self.estimate_cost(instance=instance) for instance in unlabeled]
        labels = self.label_instances(unlabeled, target)
        label_cost = self.estimate_instances(unlabeled)
        return labels, label_cost

    def label_instances(self, unlabeled, target):
        return [self.label_instance(unlabeled=instance, target=y) for instance,y in zip(unlabeled,target)]

    def estimate_instances(self, unlabeled):
        return [self.estimate_cost(instance=instance) for instance in unlabeled]


class FixedAccuracyExpert(BaseExpert):

    def __init__(self, accuracy_value=None, cost_function=None, seed=None):
        super(FixedAccuracyExpert,self).__init__(model=None, seed=seed)
        self.accuracy_model = FixedAccuracyModel(accuracy_value=accuracy_value, seed=seed)
        self.estimate_cost = cost_function

    def label_instance(self, unlabeled, target=None):
        return self.accuracy_model.predict_label(instance=unlabeled, target=target)

    def __str__(self):
        string = "{0}(accuracy_value={1}, seed={2})".format(self.__class__.__name__, self.accuracy_model.accuracy_value, self.seed)
        return string

    def __repr__(self):
        string = "{0}(accuracy_value={1}, seed={2})".format(self.__class__.__name__, self.accuracy_model.accuracy_value, self.seed)
        return string


class TrueOracleExpert(BaseExpert):

    def __init__(self, model=None, seed=None, cost_function=None):
        super(TrueOracleExpert,self).__init__(model=None, seed=seed)
        self.estimate_cost = cost_function

    def label_instance(self, unlabeled, target=None):
        return target

    def __str__(self):
        string = self.__class__.__name__
        return string

    def __repr__(self):
        string = self.__class__.__name__
        return string


class LRFunctionExpert(BaseExpert):

    def __init__(self, model=None, seed=None, cost_function=None):
        super(LRFunctionExpert,self).__init__(model=model, seed=seed)
        self.estimate_cost = cost_function
        self.model = np.array(model)
        self.accuracy_model = LRAccuracyModel(model=model)

    def get_features(self, unlabeled):
        return self.accuracy_model.get_features(unlabeled)

    def label_instance(self, unlabeled, target=None):
        return self.accuracy_model.predict_label(instance=unlabeled, target=target)

    def __str__(self):
        string = self.__class__.__name__ % "(model=" % self.model.tostring() % ", seed="% self.seed %")"
        return string

    def __repr__(self):
        string = self.__class__.__name__ % "(model=" % self.model.tostring() % ", seed="% self.seed %")"
        return string


class LogFunctionExpert(LRFunctionExpert):

    def __init__(self, model=None, seed=None, cost_function=None):
        super(LogFunctionExpert,self).__init__(model=model, seed=seed, cost_function=cost_function)
        self.model = np.array(model)
        self.accuracy_model = LogAccuracyModel(model=model, seed=seed)

    def __str__(self):
        string = "{0}(model=({1},{2}), seed={3})".format(self.__class__.__name__, self.model[0], self.model[1], self.seed)
        ##string = self.__class__.__name__ % "(model=(" % str(self.model[0]) % "," % str(self.model[1]) % ") , seed="% self.seed %")"
        return string

    def __repr__(self):
        string = "{0}(model=({1},{2}), seed={3})".format(self.__class__.__name__, self.model[0], self.model[1], self.seed)
        return string


class LookUpExpert(BaseExpert):

    def __init__(self, accuracy_value=None, cost_function=None, seed=None):
        super(LookUpExpert,self).__init__(model=None, seed=seed)
        self.accuracy_model = LookUpAccuracyModel(model=accuracy_value, seed=seed)
        self.estimate_cost = cost_function

    def label_instance(self, unlabeled, target=None):
        return self.accuracy_model.predict_label(instance=unlabeled, target=target)

    def __str__(self):
        string = "{0}(table={1}, seed={2})".format(self.__class__.__name__, self.accuracy_model.model, self.seed)
        return string

    def __repr__(self):
        string = "{0}(table={1}, seed={2})".format(self.__class__.__name__, self.accuracy_model.model, self.seed)
        return string


class NeutralityExpert(BaseExpert):

    def __init__(self, model=LogisticRegression(), cost_function=None, threshold=.4, seed=1234567):
        super(NeutralityExpert, self).__init__(model=None, seed=seed)
        super(NeutralityExpert, self).__init__(model=None, seed=seed)
        self.model = model
        self.estimate_cost = cost_function
        self.neutral_threshold = threshold

    def label_instance(self, unlabeled, target=None):
        prediction = self.model.predict_proba(unlabeled)
        uncertainty = prediction.min()
        if uncertainty < self.neutral_threshold:
            return target
        else:
            return None

    def __str__(self):
        string = "{0}(clf={1}, neutral_threshold={2}, seed={3})".format(self.__class__.__name__, self.model,
                                                                        self.neutral_threshold, self.seed)
        return string

    def __repr__(self):
        string = "{0}(clf={1}, neutral_threshold={2}, seed={3})".format(self.__class__.__name__, self.model,
                                                                        self.neutral_threshold, self.seed)
        return string


class PredictingExpert(BaseExpert):

    def __init__(self, model=LogisticRegression(), cost_function=None,  seed=1234567):
        super(PredictingExpert, self).__init__(model=None, seed=seed)
        super(PredictingExpert, self).__init__(model=None, seed=seed)
        self.model = model
        self.estimate_cost = cost_function

    def label_instance(self, unlabeled, target=None):
        prediction = self.model.predict_proba(unlabeled)
        return self.model.classes_[np.argmax(prediction, axis=1)]
        # return prediction.argmax()

    def __str__(self):
        string = "{0}(clf={1}, seed={2})".format(self.__class__.__name__, self.model, self.seed)
        return string

    def __repr__(self):
        string = "{0}(clf={1}, seed={2})".format(self.__class__.__name__, self.model, self.seed)
        return string

