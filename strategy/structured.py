__author__ = 'mramire8'

from randomsampling import *
import nltk
from scipy.sparse import diags


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
    #     pass


    def x_utility(self, instance, instance_text):

        # for each document
        # sepearte into sentences and then obtaine the k sentences

        # TODO change uncertianty
        # prob = self.model.predict_proba(instance)
        # unc = 1 - prob.max()
        unc = 1
        utility = np.array([[self.score(xik, k) * unc, k] for k, xik in self.getk(instance_text)])

        order = np.argsort(utility[:, 0], axis=0)[::-1]  ## descending order

        utility_sorted = utility[order, :]
        # print format_list(utility_sorted)
        if show_utilitly:
            print "\t{0:.5f}".format(unc),
        return utility_sorted[0, 0], utility_sorted[0, 1], unc  ## return the max


    def getk(self, doc_text):
        '''
        Get the document split into k sentences
        :param doc_text: instance text (full document)
        :return: first k sentences of the document
        '''
        sents = self.sent_detector.tokenize(doc_text)  # split in sentences
        #analize = self.vcn.build_tokenizer()

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
        #analize = self.vcn.build_tokenizer()

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
        ## change the text to use the vectorizer

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
    #     '''
    #     Compute the score of the first k sentences of the document
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
        s = [x+i for x, i in zip(mx, mi)]
        return np.array(s)

    def getk(self, doc_text):
        '''
        Get the document split into k sentences
        :param doc_text: instance text (full document)
        :return: first k sentences of the document
        '''
        mm,  sents = self.sentence2values(doc_text)

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
        return mm,  sents

########################################################################################################################
## ANYTIME ACTIVE LEARNING: STRUCTURED UNCERTAINTY READING
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
        self.sent_model = None
        self.cheating = False
        self.counter = 0
        self.score = self.score_base
        self.fn_utility = self.utility_base

    def pick_next(self, pool=None, step_size=1):

        list_pool = list(pool.remaining)
        indices = self.randgen.permutation(len(pool.remaining))
        remaining = [list_pool[index] for index in indices]

        uncertainty = []
        if self.subpool is None:
            self.subpool = len(pool.remaining)

        for i in remaining[:self.subpool]:
            # data_point = candidates[i]

            utility, sent_max = self.x_utility(pool.data[i], pool.text[i])

            uncertainty.append([utility, sent_max])

        uncertainty = np.array(uncertainty)
        unc_copy = uncertainty[:, 0]
        sorted_ind = np.argsort(unc_copy, axis=0)[::-1]
        chosen = [[remaining[x], uncertainty[x, 1]] for x in sorted_ind[:int(step_size)]]  #index and k of chosen
        # util = [uncertainty[x] for x in sorted_ind[:int(step_size)]]

        return chosen

    def utility_base(self, instance):
        raise Exception("We need a utility function")

    def utility_unc(self, instance):
        prob = self.model.predict_proba(instance)
        unc = 1 - prob.max()
        return unc

    def utility_rnd(self, instance):

        return self.randgen.random_sample()

    def utility_one(self, instance):
        return 1.0


    def x_utility(self, instance, instance_text):
        # prob = self.model.predict_proba(instance)
        # unc = 1 - prob.max()

        util_score = self.fn_utility(instance)

        sentences_indoc = self.getk(instance_text)
        self.counter = sentences_indoc.shape[0]
        utility = np.array([self.score(xik)*util_score for xik in sentences_indoc])

        order = np.argsort(utility, axis=0)[::-1]  ## descending order

        utility_sorted = utility[order]

        return utility_sorted[0], sentences_indoc[order[0]]

    def score_base(self, sentence):
        """
        Return the score for the sentences
        :param sentence: feature vector of sentences
        :return: confidence score
        """
        raise Exception("There should be a score sentence")


    def score_max(self, sentence):
        """
        Return the score for the sentences, as the confidence of the sentences classifier
        :param sentence: feature vector of sentences
        :return: confidence score
        """
        pred = self.sent_model.predict_proba(sentence)
        print self.sent_model.classes_[np.argmax(pred, axis=1)]
        return self.sent_model.classes_[np.argmax(pred, axis=1)]
        # return pred.max()

    def score_rnd(self, sentence):
        return self.randgen.random_sample()

    def score_fk(self, sentence):
        self.counter -= 1
        return self.counter

    def getk(self, text):
        if not isinstance(text, list):
            text = [text]
        sents = self.sent_detector.batch_tokenize(text)

        ### do the matrix??
        sents = self.vcn.transform(sents[0])

        return sents

    def text_to_features(self, list_text):
        return self.vcn.transform(list_text)

    def train_all(self, train_data=None, train_labels=None, sent_train=None, sent_labels=None):
        ## update underlying classifier
        ## update sentence classifier
        clf = super(AnytimeLearner, self).train(train_data=train_data, train_labels=train_labels)
        self.model = clf

        if not self.cheating:
            self.sent_model = self.update_sentence(sent_train, sent_labels)

        return clf

    def update_sentence(self, sent_train, sent_labels):
        try:
            clf = copy.copy(self.base_neutral)
            clf.fit(sent_train, sent_labels)
        # else:
        #     clf = None
        except ValueError:
            clf = None
        # print clf
        return clf

    def set_cheating(self, cheat):
        self.cheating = cheat

    def set_score_model(self, model):
        self.score_model = model

    def set_sentence_model(self, model):
        self.sent_model = model

    def __str__(self):
        string = "{0}(seed={seed})".format(self.__class__.__name__, seed=self.seed)
        return string

    def __repr__(self):
        string = self.__class__.__name__ % "(model=" % self.current_model % ", accuracy_model=" % self.accuracy_model \
                 % ", budget=" % self.budget % ", seed=" % self.seed % ")"
        return string


## Doc: UNC Sentence: max confidence
class AALStructuredReadingMax(AALStructuredReading):
    def __init__(self, model=None, accuracy_model=None, budget=None, seed=None, vcn=None, subpool=None,
                 cost_model=None):
        super(AALStructuredReadingMax, self).__init__(model=model, accuracy_model=accuracy_model, budget=budget, seed=seed,
                                                cost_model=cost_model, vcn=vcn, subpool=subpool)
        self.score = self.score_max  # sentence score
        self.fn_utility = self.utility_unc  # document score


## Doc: UNC Sentence: firstk confidence
class AALStructuredReadingFirstK(AALStructuredReading):
    def __init__(self, model=None, accuracy_model=None, budget=None, seed=None, vcn=None, subpool=None,
                 cost_model=None, fk=1):
        super(AALStructuredReadingFirstK, self).__init__(model=model, accuracy_model=accuracy_model, budget=budget, seed=seed,
                                                cost_model=cost_model, vcn=vcn, subpool=subpool)
        self.score = self.score_fk  # sentence score
        self.fn_utility = self.utility_unc  # document score
        self.first_k = fk


## Doc: UNC Sentence: random confidence
class AALStructuredReadingRandomK(AALStructuredReading):
    def __init__(self, model=None, accuracy_model=None, budget=None, seed=None, vcn=None, subpool=None,
                 cost_model=None, fk=1):
        super(AALStructuredReadingRandomK, self).__init__(model=model, accuracy_model=accuracy_model, budget=budget, seed=seed,
                                                cost_model=cost_model, vcn=vcn, subpool=subpool)
        self.score = self.score_rnd  # sentence score
        self.fn_utility = self.utility_unc  # document score
        self.first_k = fk


########################################################################################################################
## Doc: _ Sentence: max confidence
class AALStructuredReadingMaxMax(AALStructuredReading):
    def __init__(self, model=None, accuracy_model=None, budget=None, seed=None, vcn=None, subpool=None,
                 cost_model=None, fk=1):
        super(AALStructuredReadingMaxMax, self).__init__(model=model, accuracy_model=accuracy_model, budget=budget, seed=seed,
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
        super(AALRandomReadingRandom, self).__init__(model=model, accuracy_model=accuracy_model, budget=budget, seed=seed,
                                                cost_model=cost_model, vcn=vcn, subpool=subpool)
        self.score = self.score_rnd  # sentence score
        self.fn_utility = self.utility_rnd  # document score
        self.first_k = fk

########################################################################################################################
## ANYTIME ACTIVE LEARNING: UNCERTAINTY THEN STRUCTURED READING
########################################################################################################################


## Doc: _ Sentence: max confidence
class AALUtilityThenStructuredReading(AALStructuredReading):
    def __init__(self, model=None, accuracy_model=None, budget=None, seed=None, vcn=None, subpool=None,
                 cost_model=None, fk=1):
        super(AALUtilityThenStructuredReading, self).__init__(model=model, accuracy_model=accuracy_model, budget=budget, seed=seed,
                                                cost_model=cost_model, vcn=vcn, subpool=subpool)
        # self.score = self.score_max  # sentence score
        # self.fn_utility = self.utility_one  # document score
        self.first_k = fk

    def pick_next(self, pool=None, step_size=1):

        list_pool = list(pool.remaining)
        indices = self.randgen.permutation(len(pool.remaining))
        remaining = [list_pool[index] for index in indices]

        uncertainty = []
        if self.subpool is None:
            self.subpool = len(pool.remaining)

        uncertainty = [self.fn_utility(pool.data[i]) for i in remaining[:self.subpool]]

        uncertainty = np.array(uncertainty)
        unc_copy = uncertainty[:]
        sorted_ind = np.argsort(unc_copy, axis=0)[::-1]
        chosen_x = [remaining[x] for x in sorted_ind[:int(step_size)]]  #index and k of chosen
        chosen = self.pick_next_sentence(chosen_x, pool=None)
        return chosen

    def pick_next_sentence(self, chosen_x, pool):

        chosen = [[index, self.x_utility(pool.data[index], pool.text[index])] for index in chosen_x]

        return chosen


    def x_utility(self, instance, instance_text):
        '''
        The utility of the sentences inside a document
        :param instance:
        :param instance_text:
        :return:
        '''

        sentences_indoc = self.getk(instance_text)

        self.counter = sentences_indoc.shape[0]

        utility = np.array([self.score(xik) for xik in sentences_indoc])

        order = np.argsort(utility, axis=0)[::-1]  ## descending order

        # utility_sorted = utility[order]

        return sentences_indoc[order[0]]


class AALUtilityThenSR_Firstk(AALUtilityThenStructuredReading):
    def __init__(self, model=None, accuracy_model=None, budget=None, seed=None, vcn=None, subpool=None,
                 cost_model=None, fk=1):
        super(AALUtilityThenSR_Firstk, self).__init__(model=model, accuracy_model=accuracy_model, budget=budget, seed=seed,
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
