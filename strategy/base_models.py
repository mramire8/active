__author__ = 'mramire8'
import numpy as np


def accuracy_coin(target, p, rnd):
    i = rnd.random_sample()

    if i < p:
        return target
    else:
        return 1 - target


class BaseAccuracyModel(object):
    def __init__(self, seed=1234567):
        self.randgen = np.random.RandomState()
        self.randgen.seed(seed)

    def predict_label(self, instance=None, target=None):
        ## given instances predict label based on the linear function
        pass

    def get_features(self, instance):
        if isinstance(instance, int):
            x = instance
        elif isinstance(instance, str):
            x = len(instance.split(" "))
        else:
            x = instance.todense().sum()
        return [x, 1]


class LRAccuracyModel(BaseAccuracyModel):
    '''
    Class that models the Accuracy of the expert based on a Linear model
    '''

    def __init__(self, model=None, seed=None):
        '''
        Constructor LRAccuracyModel
        :param model: array with the parameters of the linear functione
        :param seed: random state seed value
        :return:
        '''
        super(LRAccuracyModel, self).__init__(seed=seed)
        self.model = np.array(model)

    def predict_label(self, instance=None, target=None):
        ## given instances predict label based on the linear function
        features = self.get_features(instance)
        #accuracy = self.model.predict(features)
        accu = accuracy = np.dot(features, self.model.T)
        accuracy = 1 if accuracy > 1 else accuracy
        accuracy = 0 if accuracy < 0 else accuracy
        #i = self.randgen.uniform()
        i = self.randgen.random_sample()
        print("feat:{2}\tacc:{0}\trnd:{1}".format(features, i, accu))

        if i < accuracy:
            return target
        else:
            return 1 - target

    def __str__(self):
        string = self.__class__.__name__ % "(model=" % self.model.tostring() % ", seed=" % self.seed % ")"
        return string

    def __repr__(self):
        string = self.__class__.__name__ % "(model=" % self.model.tostring() % ", seed=" % self.seed % ")"
        return string


class LogAccuracyModel(LRAccuracyModel):
    '''
    Class that models the Accuracy of the expert based on a logarithmic linear model
    '''

    def __init__(self, model=None, seed=None):
        '''
        Constructor LRAccuracyModel
        :param model: array with the parameters of the linear function
        :param seed: random state seed value
        :return:
        '''
        super(LRAccuracyModel, self).__init__(seed=seed)
        self.model = np.array(model)

    def get_features(self, instance):
        x = super(LogAccuracyModel, self).get_features(instance=instance)
        #if isinstance(instance, str):
        #    x = len(instance.split(" "))
        #else:
        #    x = instance.todense().sum()
        x = np.log(x[0])
        return [x, 1]


class PowerAccuracyModel(LogAccuracyModel):
    '''
    Class that models the Accuracy of the expert based on a logarithmic linear model
    '''

    def __init__(self, model=None, seed=None):
        '''
        Constructor LRAccuracyModel
        :param model: array with the parameters of the linear function
        :param seed: random state seed value
        :return:
        '''
        super(PowerAccuracyModel, self).__init__(seed=seed)
        self.model = np.array(model)

    def predict_label(self, instance=None, target=None):
        ## given instances predict label based on the linear function
        features = self.get_features(instance)
        #accuracy = self.model.predict(features)
        accu = accuracy = np.exp(np.dot(features, self.model.T))
        accuracy = 1 if accuracy > 1 else accuracy
        accuracy = 0 if accuracy < 0 else accuracy
        #i = self.randgen.uniform()
        i = self.randgen.random_sample()
        print("feat:{2}\tacc:{0}\trnd:{1}".format(features, i, accu))

        if i < accuracy:
            return target
        else:
            return 1 - target


class FixedAccuracyModel(BaseAccuracyModel):
    def __init__(self, accuracy_value=None, seed=123456):
        super(FixedAccuracyModel, self).__init__(seed=seed)
        self.accuracy_value = accuracy_value
        self.accuracy_value = 1 if self.accuracy_value > 1 else self.accuracy_value
        self.accuracy_value = 0 if self.accuracy_value < 0 else self.accuracy_value
        pass

    def predict_label(self, instance=None, target=None):
        '''
        predict the label of a instance based on a fixed accuracy value
        :param instance: data point to predict upon
        :param target: ground truth of the instance
        :return: index of the target
        '''
        ## given instances predict label based on the linear function
        #i = self.randgen.uniform()
        i = self.randgen.random_sample()
        if i < self.accuracy_value:
            return target
        else:
            return 1 - target

    def __str__(self):
        return self.__class__.__name__ + "(accuracy_value=" + str(self.accuracy_value) + ")"

    def __repr__(self):
        return self.__class__.__name__ + "(accuracy_value=" + str(self.accuracy_value) + ")"


class LookUpAccuracyModel(BaseAccuracyModel):
    '''
    Class that models the Accuracy of the expert based on a Linear model
    '''

    def __init__(self, model=None, seed=None):
        '''
        Constructor LRAccuracyModel
        :param model: array with x,y values of the lookup table
        :param seed: random state seed value
        :return:
        '''
        super(LookUpAccuracyModel, self).__init__(seed=seed)
        self.model = np.array(model)

    def get_acc_value0(self, value):
        lb = 0
        x_values = self.model.keys()
        for x in range(len(x_values) - 1):
            if lb <= value < (x_values[x] + x_values[x + 1]) / 2.0:
                return self.model[x]
            lb = x_values[x] + x_values[x + 1]
        return self.model[x_values[-1]]

    def get_acc_value(self, value):
        lb = 0
        x_values = self.model[:, 0]
        for x in range(len(x_values) - 1):
            if lb <= value < (x_values[x] + x_values[x + 1]) / 2.0:
                return self.model[x, 1]
            lb = (x_values[x] + x_values[x + 1]) / 2.0
        return self.model[-1, 1]


    def predict_label(self, instance=None, target=None):
        ## given instances predict label based on the linear function
        number_features = self.get_features(instance)[0]

        accu = accuracy = self.get_acc_value(number_features)
        accuracy = 1 if accuracy > 1 else accuracy
        accuracy = 0 if accuracy < 0 else accuracy

        i = self.randgen.random_sample()
        # print("feat:{2}\tacc:{0}\trnd:{1}".format(features, i, accu))

        if i < accuracy:
            return target
        else:
            return 1 - target

    def __str__(self):
        string = self.__class__.__name__ % "(model=" % self.model.tostring() % ", seed=" % self.seed % ")"
        return string

    def __repr__(self):
        string = self.__class__.__name__ % "(model=" % self.model.tostring() % ", seed=" % self.seed % ")"
        return string

######################################################################################################################
## COST MODELS
######################################################################################################################

class BaseCostModel(object):
    def __init__(self):
        self.cost_function = self.uniform_cost
        self.cost = self.cost_function

    def uniform_cost(self, instance=None):
        return 1

    def get_features(self, instance):
        if isinstance(instance, int) or isinstance(instance, float):
            x = instance
        elif isinstance(instance, str):
            x = len(instance.split(" "))
        else:
            x = instance.todense().sum()
        return [x, 1]


class FunctionCostModel(BaseCostModel):
    def __init__(self, parameters=None):
        super(FunctionCostModel, self).__init__()
        self.cost_function = self.function_cost
        self.cost = self.cost_function
        self.parameters = np.array(parameters)

    def function_cost(self, instance=None):
        #y = self.parameters[0] * x + self.parameters[1]
        y = np.dot(self.get_features(instance), self.parameters.T)
        return y

    def __str__(self):
        string = "{0}(model=({1},{2}), seed={3})".format(self.__class__.__name__, self.parameters[0],
                                                         self.parameters[1], self.seed)
        return string

    def __repr__(self):
        string = "{0}(model=({1},{2}), seed={3})".format(self.__class__.__name__, self.parameters[0],
                                                         self.parameters[1], self.seed)
        return string


class LogCostModel(FunctionCostModel):
    def __init__(self, parameters=None):
        super(LogCostModel, self).__init__()
        self.cost_function = self.function_cost
        self.cost = self.cost_function
        self.parameters = np.array(parameters)

    def get_features(self, instance):
        x = super(LogCostModel, self).get_features(instance=instance)
        x = np.log(x)
        return [x, 1]

    def __str__(self):
        return self.__class__.__name__ + "(parameters=" + str(self.parameters) + ")"

    def __repr__(self):
        return self.__class__.__name__ + "(parameters=" + str(self.parameters) + ")"


class LookUpCostModel(BaseCostModel):
    '''
    Class that models the Accuracy of the expert based on a Linear model
    '''

    def __init__(self, parameters=None):
        super(LookUpCostModel, self).__init__()
        self.cost_function = self.function_cost
        self.cost = self.cost_function
        self.parameters = np.array(parameters)

    def get_features(self, instance):
        x = super(LookUpCostModel, self).get_features(instance=instance)

        return x[0]

    def function_cost(self, instance=None):
        #y = self.parameters[0] * x + self.parameters[1]
        y = self.get_cost_value(self.get_features(instance))
        return y

    def get_cost_value(self, instance=None):
        lb = 0
        x_values = self.parameters[:, 0]
        for x in range(len(x_values) - 1):
            if lb <= instance < (x_values[x] + x_values[x + 1]) / 2.0:
                return self.parameters[x, 1]
            lb = (x_values[x] + x_values[x + 1]) / 2.0
        return self.parameters[-1, 1]

    def __str__(self):
        string = self.__class__.__name__ % "(model=" % self.model.tostring() % ", seed=" % self.seed % ")"
        return string

    def __repr__(self):
        string = self.__class__.__name__ % "(model=" % self.model.tostring() % ", seed=" % self.seed % ")"
        return string
