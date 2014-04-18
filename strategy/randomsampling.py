__author__ = 'mramire8'

from collections import defaultdict
import copy

import numpy as np
from sklearn.linear_model import LogisticRegression

from baselearner import BaseLearner
from sklearn.feature_extraction.text import CountVectorizer
from datautil.textutils import StemTokenizer

show_utilitly = False


class RandomSamplingLearner(BaseLearner):
    def __init__(self, model=None, accuracy_model=None, budget=None, seed=None):
        super(RandomSamplingLearner, self).__init__(model=model, accuracy_model=accuracy_model, budget=budget,
                                                    seed=seed)

    def pick_next(self, pool=None, k=1):
        try:
            list_pool = list(pool.remaining)
            indices = self.randgen.permutation(len(pool.remaining))
            return [list_pool[index] for index in indices[:k]]
        except Exception, err:
            print err
            raise Exception("Invalid pool object")
            #return random.sample(pool, k)

    def __str__(self):
        string = self.__class__.__name__ #% "(model=" % self.current_model % ", accuracy_model="% self.accuracy_model #\
        #% ", budget=" % \
        #str(self.budget) % \
        #", seed=)"
        ##", seed=" % str(self.seed) % ")"
        return string

    def __repr__(self):
        string = self.__class__.__name__ % "(model=" % self.current_model % ", accuracy_model=" % self.accuracy_model \
                 % ", budget=" % self.budget % ", seed=" % self.seed % ")"
        return string


class BootstrapRandom(BaseLearner):
    def __init__(self, random_state=None):
        #model=None, cost_model=None, accuracy_model=None, budget=None, seed=1234567
        super(BootstrapRandom, self).__init__(seed=random_state)
        self.bootstrap = self.pick_next

    def pick_next(self, pool=None, k=1):
        try:
            list_pool = list(pool.remaining)
            indices = self.randgen.permutation(len(pool.remaining))
            return [list_pool[index] for index in indices[:k]]
        except Exception, err:
            #raise Exception("Invalid pool object")
            print "*" * 10
            print "Invalid pool object"
            print err


class BootstrapFromEach(BaseLearner):
    def __init__(self, seed):
        super(BootstrapFromEach, self).__init__(seed=seed)

    def bootstrap(self, pool, k=2):
        k = int(k / 2)
        data = defaultdict(lambda: [])

        for i in pool.remaining:
            data[pool.target[i]].append(i)

        chosen = []
        for label in data.keys():
            candidates = data[label]
            indices = self.randgen.permutation(len(candidates))
            chosen.extend([candidates[index] for index in indices[:k]])

        return chosen


class UncertaintyLearner(BaseLearner):
    # def __init__(self, seed=0, subpool=None):
    #     super(UncertaintyLearner, self).__init__(seed=seed)
    #     self.subpool = subpool
    def __init__(self, model=None, accuracy_model=None, budget=None, seed=None, cost_model=None, subpool=None):
        super(UncertaintyLearner, self).__init__(model=model, accuracy_model=accuracy_model, budget=budget, seed=seed,
                                                 cost_model=cost_model, subpool=subpool)
        self.model = model

    def pick_next(self, pool=None, k=1):
        list_pool = list(pool.remaining)
        indices = self.randgen.permutation(len(pool.remaining))
        remaining = [list_pool[index] for index in indices]
        candidates = [c for c in pool.data]
        uncertainty = []
        if self.subpool is None:
            self.subpool = len(pool.remaining)

        for i in remaining[:self.subpool]:
            data_point = candidates[i]
            prob = self.model.predict_proba(data_point)[0]
            maxprob = max(prob)
            uncertainty.append(1 - maxprob)

        sorted_ind = np.argsort(uncertainty)[::-1]
        chosen = [remaining[x] for x in sorted_ind][:k]

        return chosen

    def train(self, train_data=None, train_labels=None):
        '''
        Return a new copy of a retrain classifier based on paramters. If no data, return an un-trained classifier
        :param train_data: training data
        :param train_labels: target values
        :return: trained classifier
        '''
        clf = super(UncertaintyLearner, self).train(train_data=train_data, train_labels=train_labels)
        self.model = clf
        return clf

    def __str__(self):
        string = "{0}(seed={seed})".format(self.__class__.__name__, seed=self.seed)
        return string

    def __repr__(self):
        string = self.__class__.__name__ % "(model=" % self.current_model % ", accuracy_model=" % self.accuracy_model \
                 % ", budget=" % self.budget % ", seed=" % self.seed % ")"
        return string


class AnytimeLearner(BaseLearner):
    # def __init__(self, seed=0, subpool=None):
    #     super(UncertaintyLearner, self).__init__(seed=seed)
    #     self.subpool = subpool
    def __init__(self, model=None, accuracy_model=None, budget=None, seed=None, vcn=None, subpool=None,
                 cost_model=None):
        super(AnytimeLearner, self).__init__(model=model, accuracy_model=accuracy_model, budget=budget, seed=seed,
                                             cost_model=cost_model)
        self.model = model
        self.neutral_model = LogisticRegression(penalty='l1')
        self.base_neutral = LogisticRegression(penalty='l1')
        self.neutral_train = None
        self.vct_neutral = None
        self.neutral_labels = np.array([])
        self.vcn = vcn
        self._kvalues = [10, 25, 50, 75, 100]
        self.subpool = subpool

    def pick_next(self, pool=None, step_size=1):
        list_pool = list(pool.remaining)
        indices = self.randgen.permutation(len(pool.remaining))
        remaining = [list_pool[index] for index in indices]

        uncertainty = []
        if self.subpool is None:
            self.subpool = len(pool.remaining)

        for i in remaining[:self.subpool]:
            # data_point = candidates[i]

            utility, k, unc = self.x_utility(pool.data[i], pool.text[i])

            if show_utilitly:
                print "%s\t %s \t %.3f" % (i, k, utility)

            uncertainty.append([utility, k, unc])
        uncertainty = np.array(uncertainty)
        unc_copy = uncertainty[:, 0]
        sorted_ind = np.argsort(unc_copy, axis=0)[::-1]
        chosen = [[remaining[x], uncertainty[x, 1]] for x in sorted_ind[:int(step_size)]]  #index and k of chosen
        util = [uncertainty[x] for x in sorted_ind[:int(step_size)]]
        # print util
        ## chosen returns the chosen and the k value associated with it
        return chosen, util

    def x_utility(self, instance, instance_text):

        prob = self.model.predict_proba(instance)
        unc = 1 - prob.max()

        utility = np.array([[self.obj_fn_p2(xik, k) * unc, k] for k, xik in self.getk(instance_text)])

        order = np.argsort(utility[:, 0], axis=0)[::-1]  ## descending order

        utility_sorted = utility[order, :]
        # print format_list(utility_sorted)
        if show_utilitly:
            print "\t{0:.5f}".format(unc),
        return utility_sorted[0, 0], utility_sorted[0, 1], unc  ## return the max

    def obj_fn_p2(self, instance_k, k):

        ## change the text to use the vectorizer
        # xik = self.vct_neutral.transform([instance_k])
        xik = self.vcn.transform([instance_k])
        # neu = self.neutral_model.predict_proba(xik) if self.neutral_model is not None else [1]
        neu = 1
        if self.neutral_model is not None:
            neu = self.neutral_model.predict_proba(xik)[0, 1]  # probability of being not-neutral

        costk = self.predict_cost(k)

        utility = neu / costk  ## u(x) * N(xk) / C(xk)

        # print utility

        if show_utilitly:
            print "\t{0:.3f}".format(neu),
            print "\t{0:.3f}".format(costk),

        return utility

    def getk(self, doc_text):
        '''
        Return a set of subinstance of k words in classifier format
        :param doc_text:
        :return: set of subinstances of doc_text of fixk size
        '''

        qk = []
        analize = self.vcn.build_tokenizer()
        for k in self._kvalues:
            qk.append(" ".join(analize(doc_text)[0:k]))
        return zip(self._kvalues, qk)

    def update_neutral(self, train_data, train_labels):
    ## add the neutral instance to the neutral set
    ## TODO: create the neutral dataset
    ## recompute neutral data
    ## eliminate features using the student model
    # try:
    # coef = self.model.coef_[0]
    # names = self.vcn.get_feature_names()
    #
    # vocab = [names[j] for j in np.argsort(coef)[::-1] if coef[j] != 0]
    #
    # self.vct_neutral = CountVectorizer(encoding='ISO-8859-1', min_df=5, max_df=1.0, binary=False, ngram_range=(1, 3),
    #                   token_pattern='\\b\\w+\\b', tokenizer=StemTokenizer(), vocabulary=vocab)
    #
    # train_x = self.vct_neutral.fit_transform(train_data)
    #     if not isinstance(train_data, list):
        try:
            clf = copy.copy(self.base_neutral)
            clf.fit(train_data, train_labels)
        # else:
        #     clf = None
        except ValueError:
            clf = None
        # print clf
        return clf

    def train_all(self, train_data=None, train_labels=None, neu_train=None, neu_labels=None):
        '''
        Return a new copy of a retrain classifier based on paramters. If no data, return an un-trained classifier
        :param train_data: training data
        :param train_labels: target values
        :return: trained classifier
        @param neu_train:
        @param neu_labels:
        '''
        clf = super(AnytimeLearner, self).train(train_data=train_data, train_labels=train_labels)
        self.model = clf

        self.neutral_model = self.update_neutral(neu_train, neu_labels)
        return clf

    def __str__(self):
        string = "{0}(seed={seed})".format(self.__class__.__name__, seed=self.seed)
        return string

    def __repr__(self):
        string = self.__class__.__name__ % "(model=" % self.current_model % ", accuracy_model=" % self.accuracy_model \
                 % ", budget=" % self.budget % ", seed=" % self.seed % ")"
        return string


class AnytimeLearnerZeroUtility(AnytimeLearner):

    def __init__(self, model=None, accuracy_model=None, budget=None, seed=None, vcn=None, subpool=None,
             cost_model=None):
        super(AnytimeLearnerZeroUtility, self).__init__(model=model, accuracy_model=accuracy_model, budget=budget, seed=seed,
                                         cost_model=cost_model, vcn=vcn, subpool=subpool)

    def x_utility(self, instance, instance_text):

        utility = np.array([[self.obj_fn_p2(xik, k), k] for k, xik in self.getk(instance_text)])

        order = np.argsort(utility[:, 0], axis=0)[::-1]  ## descending order

        utility_sorted = utility[order, :]

        if show_utilitly:
            print "\t{0:.5f}".format(1),

        return utility_sorted[0, 0], utility_sorted[0, 1], 1  ## return the max

    def obj_fn_p2(self, instance_k, k):

        ## change the text to use the vectorizer

        xik = self.vcn.transform([instance_k])
        # neu = self.neutral_model.predict_proba(xik) if self.neutral_model is not None else [1]
        neu = 1

        if self.neutral_model is not None:
            neu = self.neutral_model.predict_proba(xik)[0, 1]  # probability of being not-neutral

        costk = self.predict_cost(k)

        utility = neu / costk  ## N(xk) / C(xk)

        # print utility

        if show_utilitly:
            print "\t{0:.3f}".format(neu),
            print "\t{0:.3f}".format(costk),

        return utility

class AnytimeLearnerDiff(BaseLearner):
    # def __init__(self, seed=0, subpool=None):
    #     super(UncertaintyLearner, self).__init__(seed=seed)
    #     self.subpool = subpool
    def __init__(self, model=None, accuracy_model=None, budget=None, seed=None, vcn=None, subpool=None,
                 cost_model=None, lambda_value=None):
        super(AnytimeLearnerDiff, self).__init__(model=model, accuracy_model=accuracy_model, budget=budget, seed=seed,
                                             cost_model=cost_model)
        self.model = model
        self.neutral_model = LogisticRegression(penalty='l1')
        self.base_neutral = LogisticRegression(penalty='l1')
        self.neutral_train = None
        self.vct_neutral = None
        self.neutral_labels = np.array([])
        self.vcn = vcn
        self._kvalues = [10, 25, 50, 75, 100]
        self.subpool = subpool
        self.lambda_value = lambda_value

    def pick_next(self, pool=None, step_size=1):
        list_pool = list(pool.remaining)
        indices = self.randgen.permutation(len(pool.remaining))
        remaining = [list_pool[index] for index in indices]

        uncertainty = []
        if self.subpool is None:
            self.subpool = len(pool.remaining)

        for i in remaining[:self.subpool]:
            # data_point = candidates[i]

            utility, k, unc = self.x_utility(pool.data[i], pool.text[i])

            if show_utilitly:
                print "%s\t %s \t %.3f" % (i, k, utility)

            uncertainty.append([utility, k, unc])
        uncertainty = np.array(uncertainty)
        unc_copy = uncertainty[:, 0]
        sorted_ind = np.argsort(unc_copy, axis=0)[::-1]
        chosen = [[remaining[x], uncertainty[x, 1]] for x in sorted_ind[:int(step_size)]]
        util = [uncertainty[x] for x in sorted_ind[:int(step_size)]]
        # print util
        ## chosen returns the chosen and the k value associated with it
        return chosen, util

    def x_utility(self, instance, instance_text):

        prob = self.model.predict_proba(instance)
        unc = 1 - prob.max()

        utility = np.array([[self.obj_fn_p2(unc, xik, k), k] for k, xik in self.getk(instance_text)])

        order = np.argsort(utility[:, 0], axis=0)[::-1]  ## descending order

        utility_sorted = utility[order, :]
        # print format_list(utility_sorted)
        if show_utilitly:
            print "\t{0:.5f}".format(unc),
        return utility_sorted[0, 0], utility_sorted[0, 1], unc  ## return the max

    def obj_fn_p2(self, uncertainty, instance_k, k):

        ## change the text to use the vectorizer
        # xik = self.vct_neutral.transform([instance_k])
        xik = self.vcn.transform([instance_k])
        # neu = self.neutral_model.predict_proba(xik) if self.neutral_model is not None else [1]
        neu = 1
        if self.neutral_model is not None:
            neu = self.neutral_model.predict_proba(xik)[0, 1]  # probability of being not-neutral

        costk = self.predict_cost(k)

        utility = (uncertainty * neu) - (self.lambda_value *costk)  ## u(x) * N(xk) / C(xk)

        # print utility

        if show_utilitly:
            print "\t{0:.3f}".format(neu),
            print "\t{0:.3f}".format(costk),

        return utility

    def getk(self, doc_text):
        '''
        Return a set of subinstance of k words in classifier format
        :param doc_text:
        :return: set of subinstances of doc_text of fixk size
        '''

        qk = []
        analize = self.vcn.build_tokenizer()
        for k in self._kvalues:
            qk.append(" ".join(analize(doc_text)[0:k]))
        return zip(self._kvalues, qk)

    def update_neutral(self, train_data, train_labels):
    ## add the neutral instance to the neutral set
    ## TODO: create the neutral dataset
    ## recompute neutral data
    ## eliminate features using the student model
    # try:
    # coef = self.model.coef_[0]
    # names = self.vcn.get_feature_names()
    #
    # vocab = [names[j] for j in np.argsort(coef)[::-1] if coef[j] != 0]
    #
    # self.vct_neutral = CountVectorizer(encoding='ISO-8859-1', min_df=5, max_df=1.0, binary=False, ngram_range=(1, 3),
    #                   token_pattern='\\b\\w+\\b', tokenizer=StemTokenizer(), vocabulary=vocab)
    #
    # train_x = self.vct_neutral.fit_transform(train_data)
        if not isinstance(train_data, list):
            clf = copy.copy(self.base_neutral)
            clf.fit(train_data, train_labels)
        else:
            clf = None
            # except ValueError:
        #     clf = None
        return clf

    def train_all(self, train_data=None, train_labels=None, neu_train=None, neu_labels=None):
        '''
        Return a new copy of a retrain classifier based on paramters. If no data, return an un-trained classifier
        :param train_data: training data
        :param train_labels: target values
        :return: trained classifier
        @param neu_train:
        @param neu_labels:
        '''
        clf = super(AnytimeLearnerDiff, self).train(train_data=train_data, train_labels=train_labels)
        self.model = clf

        self.neutral_model = self.update_neutral(neu_train, neu_labels)
        return clf

    def __str__(self):
        string = "{0}(model={1}, neutral-model={2}, subpool={3}, lambda={4})".format(self.__class__.__name__,self.model,self.neutral_model, self.subpool, self.lambda_value)
        return string

    def __repr__(self):
        string = "{0}(model={1}, neutral-model={2}, subpool={3}, lambda={4})".format(self.__class__.__name__,self.model,self.neutral_model, self.subpool, self.lambda_value)
        return string


class AnytimeLearnerV2(BaseLearner):
    def __init__(self, model=None, accuracy_model=None, budget=None, seed=None, vcn=None, subpool=None,
                 cost_model=None):
        super(AnytimeLearnerV2, self).__init__(model=model, accuracy_model=accuracy_model, budget=budget, seed=seed,
                                               cost_model=cost_model)
        self.model = model
        self.neutral_model = LogisticRegression(penalty='l1')
        self.base_neutral = LogisticRegression(penalty='l1')
        self.neutral_train = None
        self._kvalues = [10, 25, 50, 75, 100]
        self.subpool = subpool

    def pick_next(self, pool=None, alldata=None, step_size=1):
        list_pool = list(pool.remaining)
        indices = self.randgen.permutation(len(pool.remaining))
        remaining = [list_pool[index] for index in indices]

        uncertainty = []
        if self.subpool is None:
            self.subpool = len(pool.remaining)

        for i in remaining[:self.subpool]:
            utility, k = self.x_utility(pool.data[i], i, alldata)
            uncertainty.append([utility, k])
        uncertainty = np.array(uncertainty)
        unc_copy = uncertainty[:, 0]
        sorted_ind = np.argsort(unc_copy, axis=0)[::-1]
        chosen = [[remaining[x], uncertainty[x, 1]] for x in sorted_ind[:int(step_size)]]
        ## chosen returns the chosen and the k value associated with it
        return chosen

    def x_utility(self, instance, instance_index, alldata):

        prob = self.model.predict_proba(instance)
        unc = 1 - prob.max()

        utility = np.array([[self.obj_fn_p2(xik, k) * unc, k] for k, xik in self.getk(instance_index, alldata)])

        order = np.argsort(utility[:, 0], axis=0)[::-1]  ## descending order

        utility_sorted = utility[order, :]
        # print utility_sorted
        return utility_sorted[0, 0], utility_sorted[0, 1]  ## return the max

    def obj_fn_p2(self, instance_k, k):

        ## change the text to use the vectorizer
        xik = self.vct_neutral.transform([instance_k])
        # neu = self.neutral_model.predict_proba(xik) if self.neutral_model is not None else [1]
        neu = 1
        if self.neutral_model is not None:
            neu = self.neutral_model.predict_proba(xik)[0, 1]  # probability of being not-neutral

        costk = self.predict_cost(k)

        utility = neu / costk  ## u(x) * N(xk) / C(xk)
        # print utility
        return utility

    def getk(self, doc_index, alldata):
        '''
        Return a set of subinstance of k words in classifier format
        :param doc_text:
        :return: set of subinstances of doc_text of fixk size
        '''

        qk = []

        for k in self._kvalues:
            qk.append(alldata[k].bow.tocsr()[doc_index])
        return zip(self._kvalues, qk)

    def update_neutral(self, train_data, train_labels):
    ## add the neutral instance to the neutral set
    ## TODO: create the neutral dataset
    ## recompute neutral data
    ## eliminate features using the student model
        try:
            coef = self.model.coef_[0]

            vocab = [j for j in np.argsort(coef)[::-1] if coef[j] != 0]

            clf = copy.copy(self.base_neutral)
            clf.fit(train_data.tocsc()[:, vocab], train_labels)

        except Exception:
            clf = None
        return clf

    def train_all(self, train_data=None, train_labels=None, neu_train=None, neu_labels=None):
        '''
        Return a new copy of a retrain classifier based on paramters. If no data, return an un-trained classifier
        :param train_data: training data
        :param train_labels: target values
        :return: trained classifier
        @param neu_train:
        @param neu_labels:
        '''
        clf = super(AnytimeLearnerV2, self).train(train_data=train_data, train_labels=train_labels)
        self.model = clf

        self.neutral_model = self.update_neutral(neu_train, neu_labels)
        return clf

    def __str__(self):
        string = "{0}(seed={seed})".format(self.__class__.__name__, seed=self.seed)
        return string

    def __repr__(self):
        string = self.__class__.__name__ % "(model=" % self.current_model % ", accuracy_model=" % self.accuracy_model \
                 % ", budget=" % self.budget % ", seed=" % self.seed % ")"
        return string
