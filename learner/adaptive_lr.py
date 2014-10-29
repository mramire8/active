__author__ = 'mramire8'

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split, StratifiedKFold, KFold
from sklearn.grid_search import GridSearchCV
from sklearn.base import clone

class LogisticRegressionAdaptive(LogisticRegression):

    def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None):

        super(LogisticRegressionAdaptive,self).__init__(
            penalty=penalty, dual=dual, tol=tol, C=C,
            fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
            class_weight=class_weight, random_state=random_state)

        self.clf = LogisticRegression(
            penalty=penalty, dual=dual, tol=tol, C=10,
            fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
            class_weight=class_weight, random_state=random_state)
        self.c_average = []

    def fit(self, X, y):

        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y, test_size=0.1, random_state=555)

        # kcv = KFold(n=len(y), n_folds=5, shuffle=False, random_state=None)
        kcv = StratifiedKFold(y=y, n_folds=5, shuffle=False, random_state=None)
        # print self

        # print X.shape, X_train.shape
        # Set the parameters by cross-validation
        tuned_parameters = [{'C': [pow(10,x) for x in range(-3,4)]}]   #[0.001, 0.01, 0.1, 1, 10, 100, 1000]

        score = 'accuracy'

        clf = GridSearchCV(self.clf, tuned_parameters, scoring=score, cv=kcv)
        clf.fit(X, y)

        self.clf = clone(clf.best_estimator_, safe=True)
        self.C = clf.best_estimator_.C
        self.c_average.append(self.C)
        # print "best:", self.clf.C
        super(LogisticRegressionAdaptive,self).fit(X,y)
        return self.clf.fit(X, y)

    def get_c_ave(self):
        # import numpy as np
        return self.c_average

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self,X):
        return self.clf.predict_proba(X)

    def decision_function(self, X):
        return self.clf.decision_function(X)

    def transform(self, X, threshold=None):
        return self.clf.transform(X, threshold=threshold)

    def __repr__(self):
        return "%s - %s" % (self.__class__.__name__,self.clf)



class LogisticRegressionAdaptiveV2(LogisticRegression):

    def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None):

        super(LogisticRegressionAdaptiveV2,self).__init__(
            penalty=penalty, dual=dual, tol=tol, C=C,
            fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
            class_weight=class_weight, random_state=random_state)

        self.c_average = []

    def fit(self, X, y):

        kcv = StratifiedKFold(y=y, n_folds=5, shuffle=False, random_state=None)

        tuned_parameters = [{'C': [pow(10,x) for x in range(-3,4)]}]   #[0.001, 0.01, 0.1, 1, 10, 100, 1000]

        score = 'accuracy'

        clf = GridSearchCV(self.clf, tuned_parameters, scoring=score, cv=kcv)
        clf.fit(X, y)

        self.clf = clone(clf.best_estimator_, safe=True)
        self.C = clf.best_estimator_.C
        self.c_average.append(self.C)

        return super(LogisticRegressionAdaptiveV2,self).fit(X,y)

    def get_c_ave(self):
        return self.c_average

    def __repr__(self):
        return "%s - %s" % (self.__class__.__name__,self.clf)