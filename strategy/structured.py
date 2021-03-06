__author__ = 'mramire8'

import nltk
from scipy.sparse import diags
from numpy.random import RandomState
from randomsampling import *
import random


class AALStructuredFixk(AnytimeLearner):
    def __init__(self, model=None, accuracy_model=None, budget=None, seed=None, vcn=None, subpool=None,
                 cost_model=None):
        super(AALStructuredFixk, self).__init__(model=model, accuracy_model=accuracy_model, budget=budget, seed=seed,
                                                cost_model=cost_model, vcn=vcn, subpool=subpool)
        self.score_model = None
        self.sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        self._kvalues = [1]
        self.sentence_separator = ' ... '

    # def pick_next(self, pool=None, step_size=1):
    # pass


    def x_utility(self, instance, instance_text):

        # for each document
        # sepearte into sentences and then obtaine the k sentences

        # TODO change uncertianty
        # prob = self.model.predict_proba(instance)
        # unc = 1 - prob.max()
        unc = 1
        utility = np.array([[self.score(xik, k) * unc, k] for k, xik in self.getk(instance_text)])

        order = np.argsort(utility[:, 0], axis=0)[::-1]  # # descending order

        utility_sorted = utility[order, :]
        # print format_list(utility_sorted)
        if show_utilitly:
            print "\t{0:.5f}".format(unc),
        return utility_sorted[0, 0], utility_sorted[0, 1], unc  # # return the max


    def getk(self, doc_text):
        '''
        Get the document split into k sentences
        :param doc_text: instance text (full document)
        :return: first k sentences of the document
        '''
        sents = self.sent_detector.tokenize(doc_text)  # split in sentences
        # analize = self.vcn.build_tokenizer()

        qk = []
        lens = []
        for k in self._kvalues:
            ksent_text = self.sentence_separator.join(sents[:k])  ## take the first k sentences
            qk.append(ksent_text)
            lens.append(len(sents))
            # return zip(self._kvalues, qk)
        return zip(lens, qk)

    def getlastk(self, doc_text):
        '''
        Get the document split into k sentences
        :param doc_text: instance text (full document)
        :return: first k sentences of the document
        '''
        sents = self.sent_detector.tokenize(doc_text)  # split in sentences
        # analize = self.vcn.build_tokenizer()

        qk = []
        lens = []
        for k in self._kvalues:
            ksent_text = self.sentence_separator.join(sents[-k:])  ## take the first k sentences
            qk.append(ksent_text)
            lens.append(len(sents))
            # return zip(self._kvalues, qk)
        return zip(lens, qk)

    def score(self, instance_k, k):
        '''
        Compute the score of the first k sentences of the document
        :param instance_k:
        :param k:
        :return:
        '''
        # # change the text to use the vectorizer

        xik = self.vcn.transform([instance_k])
        # neu = self.neutral_model.predict_proba(xik) if self.neutral_model is not None else [1]

        neu = 1
        # TODO change the neutrality
        # if self.neutral_model is not None:   ## if the model has been built yet
        #     neu = self.neutral_model.predict_proba(xik)[0, 1]  # probability of being not-neutral

        costk = self.predict_cost(k)

        utility = neu / costk  ## N(xk)*score(xk) / C(xk)

        # print utility

        if show_utilitly:
            print "\t{0:.3f}".format(neu),
            print "\t{0:.3f}".format(costk),

        return utility

    def set_score_model(self, model):
        self.score_model = model

    def set_kvalues(self, kvalues):
        self._kvalues = kvalues


class AALStructured(AALStructuredFixk):
    def __init__(self, model=None, accuracy_model=None, budget=None, seed=None, vcn=None, subpool=None,
                 cost_model=None):
        super(AALStructured, self).__init__(model=model, accuracy_model=accuracy_model, budget=budget, seed=seed,
                                            cost_model=cost_model, vcn=vcn, subpool=subpool)
        self.score_model = None
        self.sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        self._kvalues = [1]

    # def score(self, instance_k, k):
    # '''
    # Compute the score of the first k sentences of the document
    #     :param instance_k:
    #     :param k:
    #     :return:
    #     '''
    #     ## change the text to use the vectorizer
    #
    #     xik = self.vcn.transform([instance_k])
    #     # neu = self.neutral_model.predict_proba(xik) if self.neutral_model is not None else [1]
    #
    #     neu = 1
    #     if self.neutral_model is not None:   ## if the model has been built yet
    #         neu = self.neutral_model.predict_proba(xik)[0, 1]  # probability of being not-neutral
    #
    #     costk = self.predict_cost(k)
    #
    #
    #     sentence_score
    #
    #
    #     utility = neu / costk  ## N(xk)*score(xk) / C(xk)
    #
    #     # print utility
    #
    #     if show_utilitly:
    #         print "\t{0:.3f}".format(neu),
    #         print "\t{0:.3f}".format(costk),
    #
    #     return utility
    def get_scores_sent(self, sentences):
        return sentences.sum(axis=1)

    def get_scores_sent_max(self, sentences):
        mx = [sentences[i].max() for i in xrange(sentences.shape[0])]
        mi = [sentences[i].min() for i in xrange(sentences.shape[0])]
        s = [np.max([np.abs(x), np.abs(i)]) for x, i in zip(mx, mi)]

        return np.array(s)

    def get_scores_sent3(self, sentences):
        mx = [sentences[i].max() for i in xrange(sentences.shape[0])]
        mi = [sentences[i].min() for i in xrange(sentences.shape[0])]
        s = [x + i for x, i in zip(mx, mi)]
        return np.array(s)

    def getk(self, doc_text):
        '''
        Get the document split into k sentences
        :param doc_text: instance text (full document)
        :return: first k sentences of the document
        '''
        mm, sents = self.sentence2values(doc_text)

        # print mm.sum(axis=1)[:10]
        # scores = mm.sum(axis=1)
        qk = []
        scores = self.get_scores_sent(mm)
        order_sentences = np.argsort(np.abs(scores), axis=0)[::-1]
        kscores = []
        for k in self._kvalues:
            topk = order_sentences[:k].A1  # top scored sentence in order of appearance in the document
            topk.sort()
            topk_sentences = [sents[s] for s in topk]  # get text of top k sentences
            # print "top scores: ",
            # print np.array(scores[topk][:k].A1)
            # print "%s\t" % topk,
            qk.append(self.sentence_separator.join(topk_sentences))

            # bot = order_sentences[-k:].A1
            # bot.sort()
            # bottom = [sents[s] for s in bot]
            # print "===BOTTOM==="
            # print "...".join(bottom)
            kscores.append(scores[topk][:k].A1)

        return zip(self._kvalues, qk), kscores

    def getkmax(self, doc_text):
        ''' 
        Max value of the features
        '''
        return self.getk_sent(doc_text, self.get_scores_sent_max)

    def getk3(self, doc_text):
        """
        get the top k sentences by max-min
        :param doc_text: raw text of the document
        :return: list top k scored sentences in order of appeareance
        """
        return self.getk_sent(doc_text, self.get_scores_sent3)

    def getk_sent(self, doc_text, score_fn):
        '''
        Get the document split into k sentences
        :param doc_text: instance text (full document)
        :return: first k sentences of the document
        '''
        mm, sents = self.sentence2values(doc_text)
        qk = []
        scores = score_fn(mm)
        order_sentences = np.argsort(np.abs(scores), axis=0)[::-1]
        kscores = []

        for k in self._kvalues:
            topk = order_sentences[:k]  # top scored sentence in order of appearance in the document
            topk.sort()
            topk_sentences = [sents[s] for s in topk]  # get text of top k sentences
            # print "%s\t" % topk,
            qk.append(self.sentence_separator.join(topk_sentences))

            bot = order_sentences[-k:]
            bot.sort()
            bottom = [sents[s] for s in bot]
            kscores.append(scores[topk][:k])

        return zip(self._kvalues, qk), kscores

    def sentence2values(self, doc_text):
        np.set_printoptions(precision=4)
        sents = self.sent_detector.tokenize(doc_text)
        #analize = self.vcn.build_tokenizer()
        sents_feat = self.vcn.transform(sents)
        coef = self.score_model.coef_[0]
        dc = diags(coef, 0)

        mm = sents_feat * dc  # sentences feature vectors \times diagonal of coeficients. sentences by features
        return mm, sents


# #######################################################################################################################
# # ANYTIME ACTIVE LEARNING: STRUCTURED UNCERTAINTY READING
########################################################################################################################

class AALStructuredReading(AnytimeLearner):
    def __init__(self, model=None, accuracy_model=None, budget=None, seed=None, vcn=None, subpool=None,
                 cost_model=None):
        super(AALStructuredReading, self).__init__(model=model, accuracy_model=accuracy_model, budget=budget, seed=seed,
                                                   cost_model=cost_model, vcn=vcn, subpool=subpool)
        self.score_model = None
        self.sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        self._kvalues = [1]
        self.sentence_separator = ' ... '
        self.sent_model = copy.copy(model)
        self.base_neutral = copy.copy(model)  ## we will use same for student and sentence model
        self.cheating = False
        self.counter = 0
        self.curret_doc = None
        self.score = self.score_base
        self.fn_utility = self.utility_base
        self.rnd_vals = RandomState(4321)
        self.limit = 0
        self.calibration_threshold = (.5, .5)
        self.logit_scores = False

    def pick_next(self, pool=None, step_size=1):

        list_pool = list(pool.remaining)
        indices = self.randgen.permutation(len(pool.remaining))
        remaining = [list_pool[index] for index in indices]

        uncertainty = []
        if self.subpool is None:
            self.subpool = len(pool.remaining)
        sent_text = []
        for i in remaining[:self.subpool]:
            # data_point = candidates[i]

            utility, sent_max, text, _ = self.x_utility(pool.data[i], pool.text[i])
            sent_text.append(text)
            uncertainty.append([utility, sent_max])

        uncertainty = np.array(uncertainty)
        unc_copy = uncertainty[:, 0]
        sorted_ind = np.argsort(unc_copy, axis=0)[::-1]
        chosen = [[remaining[x], uncertainty[x, 1]] for x in sorted_ind[:int(step_size)]]  #index and k of chosen
        # util = [uncertainty[x] for x in sorted_ind[:int(step_size)]]
        # TODO: remove chosen_text later this is only for debugging
        chosen_text = [sent_text[t] for t in sorted_ind[:int(step_size)]]
        return chosen#, chosen_text

    def pick_next_cal(self, pool=None, step_size=1):
        from sklearn import preprocessing
        import time
        list_pool = list(pool.remaining)
        indices = self.randgen.permutation(len(pool.remaining))
        remaining = [list_pool[index] for index in indices]

        if self.subpool is None:
            self.subpool = len(pool.remaining)

        text_sent = []
        all_scores = []
        all_p0 = []
        docs = []
        unc_utility = []
        t0 = time.time()
        for i in remaining[:self.subpool]:
            unc_utility.append(self.fn_utility(pool.data[i]))
            utilities, sent_bow, sent_txt = self.x_utility_cal(pool.data[i], pool.text[i])  # utulity for every sentences in document
            all_scores.extend(utilities) ## every score
            docs.append(sent_bow)  ## sentences for each document list of list
            text_sent.append(sent_txt)  ## text sentences for each document
            all_p0 = np.concatenate((all_p0, utilities))

        ## Calibrate scores
        n = len(all_scores)
        if n != len(all_p0):
            raise Exception("Oops there is something wrong! We don't have the same size")

        # all_p0 = np.array(all_p0)

        order = all_p0.argsort()[::-1] ## descending order
        ## generate scores equivalent to max prob
        ordered_p0 = all_p0[order]
        from sys import maxint

        upper = self.calibration_threshold[1]
        lower = self.calibration_threshold[0]
        if upper is not .5:
            c0_scores = preprocessing.scale(ordered_p0[ordered_p0 >= upper])
            c1_scores = -1. * preprocessing.scale(ordered_p0[ordered_p0 <= lower])
            mid = len(ordered_p0) - ((ordered_p0 >= upper).sum() + (ordered_p0 <= lower).sum())
            middle = np.array([-maxint]*mid)
            print "Threshold:", lower, upper
            print "middle:", len(middle),
            a = np.concatenate((c0_scores, middle,c1_scores))
            print "all:", len(a), 1.*len(middle)/len(a), len(c0_scores), len(c1_scores)
        else:
            c0_scores = preprocessing.scale(ordered_p0[ordered_p0 > upper])
            c1_scores = -1. * preprocessing.scale(ordered_p0[ordered_p0 <= lower])
            print "Threshold:", lower, upper
            a = np.concatenate((c0_scores, c1_scores))
            print "all:", len(a), 1.*len(c0_scores)/len(a), len(c0_scores), len(c1_scores)


        new_scores = np.zeros(n)
        new_scores[order] = a
        cal_scores = self._reshape_scores(new_scores, docs)
        # p0 = self._reshape_scores(all_p0, docs)

        selected_sent = [np.argmax(row) for row in cal_scores]  # get the sentence index of the highest score per document
        selected = [docs[i][k] for i, k in enumerate(selected_sent)]  # get the bow of the sentences with the highest score
        selected_score = [np.max(row) for row in cal_scores]  ## get the max utility score per document
        test_sent = list_to_sparse(selected)  # bow of each sentence selected
        selected_text = [text_sent[i][k] for i,k in enumerate(selected_sent)]

        ## pick document-sentence
        if self.logit_scores:
            joint = np.array(unc_utility) * self._logit(selected_score)
        else:
            joint = np.array(unc_utility) * selected_score

        final_order = joint.argsort()[::-1]
        chosen = [[remaining[x], test_sent[x]] for x in final_order[:int(step_size)]]  #index and k of chosen
        chosen_text = [selected_text[x] for x in final_order[:int(step_size)]]
        #todo: human vs simulated expert
        return chosen#, chosen_text

    def _calibrate_scores(self, n_scores, bounds=(.5,1)):
        """
        Create an array with values uniformly distributed with int range given by bounds
        :param n_scores: number of scores to generate
        :param bounds: lower and upper bound of the array
        :return: narray
        """
        delta = 1.* (bounds[1] - bounds[0]) / (n_scores -1)
        calibrated = (np.ones(n_scores)*bounds[1]) - (np.array(range(n_scores))*delta)
        return calibrated

    def set_calibration_threshold(self, thr):
        """
        set the threshold for the calibration
        :param thr: tuple
            lower bound and upper bound
        """
        self.calibration_threshold = thr

    def _reshape_scores2(self, scores, sent_mat):
        """
        Reshape scores to have the same structure as the list of list sent_mat
        :param scores: one dimensional array of scores
        :param sent_mat: list of lists
        :return: narray : reshaped scores
        """
        sr = []
        i = 0
        for row in sent_mat:
            sc = []
            for col in row:
                sc.append(scores[i])
                i = i+1
            sr.append(sc)
        return np.array(sr)

    def _reshape_scores(self, scores, sent_mat):
        """
        Reshape scores to have the same structure as the list of list sent_mat
        :param scores: one dimensional array of scores
        :param sent_mat: list of lists
        :return: narray : reshaped scores
        """
        sr = []
        i = 0
        lengths = np.array([d.shape[0] for d in sent_mat])
        lengths = lengths.cumsum()
        for j in lengths:
            sr.append(scores[i:j])
            i = j
        return sr

    def _logit(self, x):
        return 1. / (1. + np.exp(-np.array(x)))

    def utility_base(self, instance):
        raise Exception("We need a utility function")

    def utility_unc(self, instance):
        prob = self.model.predict_proba(instance)
        unc = 1 - prob.max()
        return unc

    def utility_rnd(self, instance):
        return self.rnd_vals.rand()
        # return self.randgen.random_sample()

    def utility_one(self, instance):
        return 1.0


    def x_utility(self, instance, instance_text):
        # prob = self.model.predict_proba(instance)
        # unc = 1 - prob.max()

        util_score = self.fn_utility(instance)

        sentences_indoc, sent_text = self.getk(instance_text)
        self.counter = 0
        self.curret_doc = instance
        utility = np.array([self.score(xik) * util_score for xik in sentences_indoc])

        order = np.argsort(utility, axis=0)[::-1]  ## descending order

        utility_sorted = utility[order]
        # print utility_sorted[0], util_score
        return utility_sorted[0], sentences_indoc[order[0]], sent_text[order[0]], order[0]

    def x_utility_cal(self, instance, instance_text):

        sentences_indoc, sent_text = self.getk(instance_text)
        self.counter = 0
        self.curret_doc = instance
        # utility = np.array([self.score(xik) for xik in sentences_indoc])
        utility = self.score(sentences_indoc)
        if len(sent_text) <=1:
            utility = np.array([utility])
        # order = np.argsort(utility, axis=0)[::-1]  ## descending order
        #
        # utility_sorted = utility[order]
        # return utility_sorted, sentences_indoc[order], np.array(sent_text)[order]
        return utility, sentences_indoc, np.array(sent_text)

    def score_base(self, sentence):
        """
        Return the score for the sentences
        :param sentence: feature vector of sentences
        :return: confidence score
        """
        raise Exception("There should be a score sentence")

    def score_max(self, sentence):
        """
        Return the score for the sentences, as the confidence of the sentence classifier maxprob value
        :param sentence: feature vector of sentences
        :return: confidence score
        """

        if self.sent_model is not None:  ## if the model has been built yet
            pred = self.sent_model.predict_proba(sentence)
            if sentence.shape[0] == 1:
                return pred.max()
            else:
                return pred.max(axis=1)
        return 1.0

    def score_p0(self, sentence):
        """
        Return the score for the sentences, as the confidence of the sentence classifier maxprob value
        :param sentence: feature vector of sentences
        :return: confidence score
        """

        if self.sent_model is not None:  ## if the model has been built yet
            pred = self.sent_model.predict_proba(sentence)
            if sentence.shape[0] == 1:
                return pred[0][0]
            else:
                return pred[:,0]
        return np.array([1.0] * sentence.shape[0])

    def score_rnd(self, sentence):
        if sentence.shape[0] ==1:
            return self.randgen.random_sample()
        else:
            return self.randgen.random_sample(sentence.shape[0])

    def score_fk(self, sentence):
        if sentence.shape[0] ==1:
            self.counter += 1
            if self.counter == 1:
                return 1.0
            else:
                return 0.0
                # self.counter -= 1
                # return self.counter
        else:
            return np.array([1] + [0]*(sentence.shape[0]-1))

    def getk(self, text):
        if not isinstance(text, list):
            text = [text]
        sents = self.sent_detector.batch_tokenize(text)

        if self.limit == 0:
            pass
        elif self.limit > 0:
            sents = [s for s in sents[0] if len(s.strip()) > self.limit]
            sents = [sents]  # trick to use the vectorizer properly
        ### do the matrix
        sents_bow = self.vcn.transform(sents[0])

        return sents_bow, sents[0]

    def text_to_features(self, list_text):
        return self.vcn.transform(list_text)

    def train_all(self, train_data=None, train_labels=None, sent_train=None, sent_labels=None):
        ## update underlying classifier
        ## update sentence classifier

        self.model = super(AnytimeLearner, self).train(train_data=train_data, train_labels=train_labels)

        if not self.cheating:
            self.sent_model = self.update_sentence(sent_train, sent_labels)

        return self.model

    def update_sentence(self, sent_train, sent_labels):
        # try:
        #     clf = copy.copy(self.base_neutral)
        #     clf.fit(sent_train, sent_labels)
        # except ValueError:
        #     clf = None
        # return clf
        if self.sent_model is None:
            self.sent_model = copy.copy(self.base_neutral)
        return self.sent_model.fit(sent_train, sent_labels)

    def set_cheating(self, cheat):
        self.cheating = cheat

    def get_cheating(self):
        return self.cheating

    def set_score_model(self, model):
        self.score_model = model

    def set_sentence_model(self, model):
        self.sent_model = model

    def set_sent_score(self, sent_score_fn):
        self.score = sent_score_fn

    def score_fk_max(self, sentence):
        self.counter += 1
        if self.counter <= self._kvalues[0]:
            if self.sent_model is not None:  ## if the model has been built yet
                pred = self.sent_model.predict_proba(sentence)
                return pred.max()
            else:  # this should not happen, there should be a model for this computation
                raise Exception("Oops: The sentences model is not available to compute a sentence score.")
        else:  ## if it is not the first sentence, return 0
            return 0.0

    def score_max_feat(self, sentence):
        return np.max(abs(sentence.max()), abs(sentence.min()))

    def score_max_sim(self, sentence):
        s = sentence.dot(self.curret_doc.T).max()
        return s

    def __str__(self):
        string = "{0}(seed={seed}".format(self.__class__.__name__, seed=self.seed)
        string += ", clf = {}, utility={}, score={}".format(self.model, self.fn_utility.__name__, self.score.__name__)
        string += ", sent_detector={})".format(self.sent_detector.__class__.__name__)
        return string

    def __repr__(self):
        string = self.__class__.__name__ % "(model=" % self.current_model % ", accuracy_model=" % self.accuracy_model \
                 % ", budget=" % self.budget % ", seed=" % self.seed % ")"
        return string

from scipy.sparse import vstack
def list_to_sparse(selected):
    test_sent = []
    for s in selected:
        if isinstance(test_sent, list):
            test_sent = s
        else:
            test_sent = vstack([test_sent, s], format='csr')
    return test_sent

## Doc: UNC Sentence: max confidence
class AALStructuredReadingMax(AALStructuredReading):
    def __init__(self, model=None, accuracy_model=None, budget=None, seed=None, vcn=None, subpool=None,
                 cost_model=None):
        super(AALStructuredReadingMax, self).__init__(model=model, accuracy_model=accuracy_model, budget=budget,
                                                      seed=seed,
                                                      cost_model=cost_model, vcn=vcn, subpool=subpool)
        self.score = self.score_max  # sentence score
        self.fn_utility = self.utility_unc  # document score


## Doc: UNC Sentence: firstk uniform confidence
class AALStructuredReadingFirstK(AALStructuredReading):
    def __init__(self, model=None, accuracy_model=None, budget=None, seed=None, vcn=None, subpool=None,
                 cost_model=None, fk=1):
        super(AALStructuredReadingFirstK, self).__init__(model=model, accuracy_model=accuracy_model, budget=budget,
                                                         seed=seed,
                                                         cost_model=cost_model, vcn=vcn, subpool=subpool)
        self.score = self.score_fk  # sentence score
        self.fn_utility = self.utility_unc  # document score
        self.first_k = fk


## Doc: UNC Sentence: firstk max confidence

class AALStructuredReadingFirstKMax(AALStructuredReading):
    def __init__(self, model=None, accuracy_model=None, budget=None, seed=None, vcn=None, subpool=None,
                 cost_model=None, fk=1):
        super(AALStructuredReadingFirstKMax, self).__init__(model=model, accuracy_model=accuracy_model, budget=budget,
                                                            seed=seed,
                                                            cost_model=cost_model, vcn=vcn, subpool=subpool)
        self.score = self.score_fk_max  # sentence score
        self.fn_utility = self.utility_unc  # document score
        self.first_k = fk

        # def score_fk_max(self, sentence):
        #     self.counter += 1
        #     if self.counter <= self._kvalues[0]:
        #         if self.sent_model is not None:   ## if the model has been built yet
        #             pred = self.sent_model.predict_proba(sentence)
        #             return pred.max()
        #         else:  # this should not happen, there should be a model for this computation
        #             raise Exception("Oops: The sentences model is not available to compute a sentence score.")
        #     else: ## if it is not the first sentence, return 0
        #         return 0.0


## Doc: UNC Sentence: random confidence
class AALStructuredReadingRandomK(AALStructuredReading):
    def __init__(self, model=None, accuracy_model=None, budget=None, seed=None, vcn=None, subpool=None,
                 cost_model=None, fk=1):
        super(AALStructuredReadingRandomK, self).__init__(model=model, accuracy_model=accuracy_model, budget=budget,
                                                          seed=seed,
                                                          cost_model=cost_model, vcn=vcn, subpool=subpool)
        self.score = self.score_rnd  # sentence score
        self.fn_utility = self.utility_unc  # document score
        self.first_k = fk


########################################################################################################################
## Doc: _ Sentence: max confidence
class AALStructuredReadingMaxMax(AALStructuredReading):
    def __init__(self, model=None, accuracy_model=None, budget=None, seed=None, vcn=None, subpool=None,
                 cost_model=None, fk=1):
        super(AALStructuredReadingMaxMax, self).__init__(model=model, accuracy_model=accuracy_model, budget=budget,
                                                         seed=seed,
                                                         cost_model=cost_model, vcn=vcn, subpool=subpool)
        self.score = self.score_max  # sentence score
        self.fn_utility = self.utility_one  # document score
        self.first_k = fk


########################################################################################################################
## ANYTIME ACTIVE LEARNING: STRUCTURED RANDOM READING
########################################################################################################################

class AALRandomReadingFK(AALStructuredReading):
    def __init__(self, model=None, accuracy_model=None, budget=None, seed=None, vcn=None, subpool=None,
                 cost_model=None, fk=1):
        super(AALRandomReadingFK, self).__init__(model=model, accuracy_model=accuracy_model, budget=budget, seed=seed,
                                                 cost_model=cost_model, vcn=vcn, subpool=subpool)
        self.score = self.score_fk  # sentence score
        self.fn_utility = self.utility_rnd  # document score
        self.first_k = fk


class AALRandomReadingMax(AALStructuredReading):
    def __init__(self, model=None, accuracy_model=None, budget=None, seed=None, vcn=None, subpool=None,
                 cost_model=None, fk=1):
        super(AALRandomReadingMax, self).__init__(model=model, accuracy_model=accuracy_model, budget=budget, seed=seed,
                                                  cost_model=cost_model, vcn=vcn, subpool=subpool)
        self.score = self.score_max  # sentence score
        self.fn_utility = self.utility_rnd  # document score
        self.first_k = fk


class AALRandomReadingRandom(AALStructuredReading):
    def __init__(self, model=None, accuracy_model=None, budget=None, seed=None, vcn=None, subpool=None,
                 cost_model=None, fk=1):
        super(AALRandomReadingRandom, self).__init__(model=model, accuracy_model=accuracy_model, budget=budget,
                                                     seed=seed,
                                                     cost_model=cost_model, vcn=vcn, subpool=subpool)
        self.score = self.score_rnd  # sentence score
        self.fn_utility = self.utility_rnd  # document score
        self.first_k = fk


########################################################################################################################
## TOP FROM EACH
########################################################################################################################

class AALTFEStructuredReading(AALStructuredReading):
    def __init__(self, model=None, accuracy_model=None, budget=None, seed=None, vcn=None, subpool=None,
                 cost_model=None, fk=1):
        super(AALTFEStructuredReading, self).__init__(model=model, accuracy_model=accuracy_model, budget=budget,
                                                      seed=seed,
                                                      cost_model=cost_model, vcn=vcn, subpool=subpool)
        self.score = self.score_max  # sentence score
        self.fn_utility = self.utility_unc  # document score
        self.first_k = fk

    def pick_next(self, pool=None, step_size=1):

        list_pool = list(pool.remaining)
        indices = self.randgen.permutation(len(pool.remaining))
        remaining = [list_pool[index] for index in indices]

        uncertainty = []
        if self.subpool is None:
            self.subpool = len(pool.remaining)
        sent_text = []
        pred_class = []
        for i in remaining[:self.subpool]:
            # data_point = candidates[i]

            utility, sent_max, text, _ = self.x_utility(pool.data[i], pool.text[i])
            pc = self.sent_model.predict(sent_max)
            # print pc
            pred_class.append(pc)
            sent_text.append(text)
            uncertainty.append([utility, sent_max])

        uncertainty = np.array(uncertainty)
        unc_copy = uncertainty[:, 0]
        sorted_ind = np.argsort(unc_copy, axis=0)[::-1]

        chosen_0 = [[remaining[x], uncertainty[x, 1]] for x in sorted_ind if pred_class[x] == 0]  #index and k of chosen
        chosen_1 = [[remaining[x], uncertainty[x, 1]] for x in sorted_ind if pred_class[x] == 1]  #index and k of chosen
        half = int(step_size / 2)
        chosen = chosen_0[:half]
        chosen.extend(chosen_1[:half])
        if len(chosen) < step_size:
            miss = step_size - len(chosen)
            if len(chosen_0) < step_size / 2:
                chosen.extend(chosen_1[half: (half + miss)])
            elif len(chosen_1) < step_size / 2:
                chosen.extend(chosen_0[half: (half + miss)])
            else:
                raise Exception("Oops, we cannot get the next batch. We ran out of instances.")
        # print "chosen len:", len(chosen_0), len(chosen_1)
        # util = [uncertainty[x] for x in sorted_ind[:int(step_size)]]
        return chosen


########################################################################################################################

class AALTFEStructuredReadingFK(AALTFEStructuredReading):
    def __init__(self, model=None, accuracy_model=None, budget=None, seed=None, vcn=None, subpool=None,
                 cost_model=None, fk=1):
        super(AALTFEStructuredReadingFK, self).__init__(model=model, accuracy_model=accuracy_model, budget=budget,
                                                        seed=seed,
                                                        cost_model=cost_model, vcn=vcn, subpool=subpool)
        self.score = self.score_fk  # sentence score
        self.fn_utility = self.utility_unc  # document score
        self.first_k = fk

        # def score_fk_max(self, sentence):
        #     self.counter += 1
        #     if self.counter <= self._kvalues[0]:
        #         if self.sent_model is not None:   ## if the model has been built yet
        #             pred = self.sent_model.predict_proba(sentence)
        #             return pred.max()
        #         else:  # this should not happen, there should be a model for this computation
        #             raise Exception("Oops: The sentences model is not available to compute a sentence score.")
        #     else: ## if it is not the first sentence, return 0
        #         return 0.0


########################################################################################################################
## ANYTIME ACTIVE LEARNING: UNCERTAINTY THEN STRUCTURED READING
########################################################################################################################


## Doc: _ Sentence: max confidence
class AALUtilityThenStructuredReading(AALStructuredReading):
    def __init__(self, model=None, accuracy_model=None, budget=None, seed=None, vcn=None, subpool=None,
                 cost_model=None, fk=1):
        super(AALUtilityThenStructuredReading, self).__init__(model=model, accuracy_model=accuracy_model, budget=budget,
                                                              seed=seed,
                                                              cost_model=cost_model, vcn=vcn, subpool=subpool)
        # self.score = self.score_max  # sentence score
        # self.fn_utility = self.utility_one  # document score
        self.first_k = fk
        self.human_mode = False
        self.calibratescores = False

    def pick_next(self, pool=None, step_size=1):
        list_pool = list(pool.remaining)
        indices = self.randgen.permutation(len(pool.remaining))
        remaining = [list_pool[index] for index in indices]

        if self.subpool is None:
            self.subpool = len(pool.remaining)

        # Select the top based on utility
        uncertainty = [self.fn_utility(pool.data[i]) for i in remaining[:self.subpool]]

        uncertainty = np.array(uncertainty)
        unc_copy = uncertainty[:]
        sorted_ind = np.argsort(unc_copy, axis=0)[::-1]
        chosen_x = [remaining[x] for x in sorted_ind[:int(step_size)]]  #index and k of chosen

        #After utility, pick the best sentence of each document
        chosen = self.pick_next_sentence(chosen_x, pool=pool)
        final = []
        if self.human_mode:
            for x, y in chosen[:step_size]:
                final.append([x, y[1]])  # index, text
        else:
            for x, y in chosen[:step_size]:
                final.append([x, y[0][0]]) #index, feature vector
        return final
        # return chosen

    def pick_next_sentence(self, chosen_x, pool):
        '''
        After picking documents, compute the best sentence based on x_utility function
        :param chosen_x:
        :param pool:
        :return:
        '''
        if not self.calibratescores:
            chosen = [[index, self.x_utility(pool.data[index], pool.text[index])] for index in chosen_x]
        else:
            sent_chosen = self.pick_next_sentence_cal(chosen_x, pool)
            chosen = [[index, sent_bow] for index, sent_bow in sent_chosen]
            # chosen = [[index, sent_bow] for index, sent_bow in sent_chosen]

        return chosen

    def pick_next_sentence_cal(self, chosen_x, pool):
        from sklearn import preprocessing

        # list_pool = list(pool.remaining)
        # indices = self.randgen.permutation(len(pool.remaining))
        remaining = [x for x in chosen_x]

        # if self.subpool is None:
        #     self.subpool = len(pool.remaining)
        # remaining = list(set(remaining[:self.subpool]) | set(chosen_x))

        text_sent = []
        all_scores = []
        all_p0 = []
        docs = []
        unc_utility = []
        used_index = []
        for i in remaining:
            utilities, sent_bow, sent_txt = self.x_utility_cal(pool.data[i], pool.text[i])  # utulity for every sentences in document
            all_scores.extend(utilities) ## every score
            docs.append(sent_bow)  ## sentences for each document list of list
            text_sent.append(sent_txt)  ## text sentences for each document
            all_p0 = np.concatenate((all_p0, utilities))
        ## Calibrate scores
        n = len(all_scores)
        if n != len(all_p0):
            raise Exception("Oops there is something wrong! We don't have the same size")

        # all_p0 = np.array(all_p0)

        order = all_p0.argsort()[::-1] ## descending order
        ## generate scores equivalent to max prob
        ordered_p0 = all_p0[order]
        from sys import maxint

        upper = self.calibration_threshold[1]
        lower = self.calibration_threshold[0]
        if upper is not .5:
            c0_scores = preprocessing.scale(ordered_p0[ordered_p0 >= upper])
            c1_scores = -1. * preprocessing.scale(ordered_p0[ordered_p0 <= lower])
            mid = len(ordered_p0) - ((ordered_p0 >= upper).sum() + (ordered_p0 <= lower).sum())
            middle = np.array([-maxint]*mid)
            print "Threshold seq:", lower, upper
            print "middle:", len(middle),
            a = np.concatenate((c0_scores, middle,c1_scores))
            print "all:", len(a), 1.*len(middle)/len(a), len(c0_scores), len(c1_scores)
        else:
            c0_scores = preprocessing.scale(ordered_p0[ordered_p0 > upper])
            c1_scores = -1. * preprocessing.scale(ordered_p0[ordered_p0 <= lower])
            print "Threshold seq:", lower, upper
            a = np.concatenate((c0_scores, c1_scores))
            print "all:", len(a), 1.*len(c0_scores)/len(a), len(c0_scores), len(c1_scores)


        new_scores = np.zeros(n)
        new_scores[order] = a
        cal_scores = self._reshape_scores(new_scores, docs)
        # p0 = self._reshape_scores(all_p0, docs)

        selected_sent = [np.argmax(row) for row in cal_scores]  # get the sentence index of the highest score per document
        selected = [docs[i][k] for i, k in enumerate(selected_sent)]  # get the bow of the sentences with the highest score
        selected_score = [np.max(row) for row in cal_scores]  ## get the max utility score per document
        test_sent = list_to_sparse(selected)  # bow of each sentence selected
        selected_text = [text_sent[i][k] for i,k in enumerate(selected_sent)]

        ## pick document-sentence
        # chosen = [[remaining[x], test_sent[x]] for x in remaining if x in chosen_x]
        chosen = [[x, test_sent[i]] for i, x in enumerate(remaining) if x in chosen_x]
        chosen_text = [selected_text[i] for i, x in enumerate(remaining) if x in chosen_x]

        #todo: human vs simulated expert
        return chosen#, chosen_text


    def x_utility(self, instance, instance_text):
        '''
        The utility of the sentences inside a document
        :param instance:
        :param instance_text:
        :return:
        '''

        sentences_indoc, sent_text = self.getk(instance_text)  #get subinstances

        self.counter = 0

        utility = np.array([self.score(xik) for xik in sentences_indoc])  # get the senteces score

        order = np.argsort(utility, axis=0)[::-1]  ## descending order, top scores

        return [sentences_indoc[i] for i in order[:self.first_k]], [sent_text[i] for i in order[:self.first_k]]

    def set_sent_score(self, sent_score_fn):
        self.score = sent_score_fn


class AALUtilityThenSR_Firstk(AALUtilityThenStructuredReading):
    def __init__(self, model=None, accuracy_model=None, budget=None, seed=None, vcn=None, subpool=None,
                 cost_model=None, fk=1):
        super(AALUtilityThenSR_Firstk, self).__init__(model=model, accuracy_model=accuracy_model, budget=budget,
                                                      seed=seed,
                                                      cost_model=cost_model, vcn=vcn, subpool=subpool)
        self.score = self.score_fk  # sentence score
        self.fn_utility = self.utility_unc  # document score
        self.first_k = fk


class AALUtilityThenSR_Max(AALUtilityThenStructuredReading):
    def __init__(self, model=None, accuracy_model=None, budget=None, seed=None, vcn=None, subpool=None,
                 cost_model=None, fk=1):
        super(AALUtilityThenSR_Max, self).__init__(model=model, accuracy_model=accuracy_model, budget=budget, seed=seed,
                                                   cost_model=cost_model, vcn=vcn, subpool=subpool)
        self.score = self.score_max  # sentence score
        self.fn_utility = self.utility_unc  # document score
        self.first_k = fk


class AALUtilityThenSR_RND(AALUtilityThenStructuredReading):
    def __init__(self, model=None, accuracy_model=None, budget=None, seed=None, vcn=None, subpool=None,
                 cost_model=None, fk=1):
        super(AALUtilityThenSR_RND, self).__init__(model=model, accuracy_model=accuracy_model, budget=budget, seed=seed,
                                                   cost_model=cost_model, vcn=vcn, subpool=subpool)
        self.score = self.score_rnd  # sentence score
        self.fn_utility = self.utility_unc  # document score
        self.first_k = fk


class AALTFEUtilityThenSR_Max(AALUtilityThenStructuredReading):
    def __init__(self, model=None, accuracy_model=None, budget=None, seed=None, vcn=None, subpool=None,
                 cost_model=None, fk=1):
        super(AALTFEUtilityThenSR_Max, self).__init__(model=model, accuracy_model=accuracy_model, budget=budget,
                                                      seed=seed,
                                                      cost_model=cost_model, vcn=vcn, subpool=subpool)
        self.score = self.score_max  # sentence score
        self.fn_utility = self.utility_unc  # document score
        self.first_k = fk

    def pick_next(self, pool=None, step_size=1):
        # return all order by utility of x
        chosen = super(AALTFEUtilityThenSR_Max, self).pick_next(pool=pool, step_size=self.subpool)

        pred_class = []
        chosen_0 = []
        chosen_1 = []
        for index_x, sent_x in chosen:
            pc = self.sent_model.predict(sent_x[0])
            pred_class.append(pc)
            if pc == 0:
                chosen_0.append([index_x, sent_x[0]])
            else:
                chosen_1.append([index_x, sent_x[0]])

        half = int(step_size / 2)
        chosen = chosen_0[:half]
        chosen.extend(chosen_1[:half])
        if len(chosen) < step_size:
            miss = step_size - len(chosen)
            if len(chosen_0) < step_size / 2:
                chosen.extend(chosen_1[half: (half + miss)])
            elif len(chosen_1) < step_size / 2:
                chosen.extend(chosen_0[half: (half + miss)])
            else:
                raise Exception("Oops, we cannot get the next batch. We ran out of instances.")

        return chosen


class AALRandomThenSR(AALUtilityThenStructuredReading):
    def __init__(self, model=None, accuracy_model=None, budget=None, seed=None, vcn=None, subpool=None,
                 cost_model=None, fk=1):
        super(AALRandomThenSR, self).__init__(model=model, accuracy_model=accuracy_model, budget=budget,
                                                 seed=seed,
                                                 cost_model=cost_model, vcn=vcn, subpool=subpool)
        self.score = self.score_max  # sentence score
        self.fn_utility = self.utility_unc  # document score
        self.first_k = fk
        self.human_mode = False

    def pick_random(self, pool=None, step_size=1):
        ## version 2:
        chosen_x = pool.remaining[pool.offset:(pool.offset + step_size)]

        #After utility, pick the best sentence of each document

        chosen = self.pick_next_sentence(chosen_x, pool=pool)

        return chosen

    def pick_next(self, pool=None, step_size=1):
        # return all order by utility of x
        chosen = self.pick_random(pool=pool, step_size=self.subpool) ## returns index, and feature vector
        final = []
        if self.human_mode:
            for x, y in chosen[:step_size]:
                final.append([x, y[1]])  # index, text
        else:
            try:
                for x, y in chosen[:step_size]:
                    final.append([x, y[0][0]]) #index, feature vector
            except IndexError:
                print chosen[:step_size]
        return final

    def x_utility_cs(self, instance, instance_text):
        """
        The utility of the sentences inside a document with class sensitive features
        :param instance:
        :param instance_text:
        :return:
        """

        sentences_indoc, sent_text = self.getk(instance_text)  #get subinstances

        self.counter = 0
        pred_doc = self.model.predict(instance)
        pred_probl = self.sent_model.predict_proba(sentences_indoc)
        pred_y = self.sent_model.classes_[np.argmax(pred_probl, axis=1)]

        order = np.argsort(pred_probl[:, pred_doc[0]], axis=0)[::-1]  # descending order, top scores

        best_sent = [sentences_indoc[i] for i in order]

        return best_sent[:self.first_k], [sent_text[i] for i in order]

    def x_utility_maj(self, instance, instance_text):
        """
        The utility of the sentences inside a document with class sensitive features by majority vote of the sentence
        classifier.
        :param instance:
        :param instance_text:
        :return:
        """

        sentences_indoc, sent_text = self.getk(instance_text)  # get subinstances

        self.counter = 0
        # pred_doc = self.model.predict(instance)
        pred_probl = self.sent_model.predict_proba(sentences_indoc)
        pred_y = self.sent_model.classes_[np.argmax(pred_probl, axis=1)]
        pred_doc = np.round(1. * pred_y.sum() / len(pred_y))  ## majority vote

        # utility = np.array([self.score(xik) for xik in sentences_indoc])  # get the senteces score
        order = np.argsort(pred_probl[:, pred_doc], axis=0)[::-1]  ## descending order, top scores

        best_sent = [sentences_indoc[i] for i in order]

        # if len(best_sent) > 0:
        return best_sent[:self.first_k]


    def set_x_utility(self, x_util_fn):
        self.x_utility = x_util_fn

    def class_sensitive_utility(self):
        self.set_x_utility(self.x_utility_cs)

    def majority_vote_utility(self):
        self.set_x_utility(self.x_utility_maj)

