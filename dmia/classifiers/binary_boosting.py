#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from copy import deepcopy
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import ClassifierMixin, BaseEstimator
from scipy.optimize import minimize

class BinaryBoostingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, lr=0.1, max_depth=3):
        self.base_regressor = DecisionTreeRegressor(criterion='friedman_mse',
                                                    splitter='best',
                                                    max_depth=max_depth)
        self.lr = lr
        self.n_estimators = n_estimators
        self.feature_importances_ = None
        self.pre_pred = None
        self.pred = None
        
        self.estimators_ = []
        self.param = []

    def loss_grad(self, original_y, pred_y):
        # Вычислите градиент на кажом объекте
        ### YOUR CODE ###
        grad = 2 * (pred_y - original_y)

        return grad

    def fit(self, X, original_y):
        # Храните базовые алгоритмы тут
        self.estimators_ = []
        self.param = []huhiu
        
        if self.pred is None:
            self.pred = np.zeros(shape = X.shape[0])

        for i in range(self.n_estimators):
            self.pre_pred = self.pred
            grad = self.loss_grad(original_y, self.pre_pred)
            # Настройте базовый алгоритм на градиент, это классификация или регрессия?
            ### YOUR CODE ###
            regressor = DecisionTreeRegressor(max_depth=3)
            regressor.fit(X,self.lr * grad) 
            b = regressor.predict(X)
            func = lambda x: np.sum((self.pre_pred + x * b - original_y)**2)
            a = minimize(func,1.1,method='Nelder-Mead',tol=1e-6).x[0]
            estimator = regressor 
            self.pred += a * b
            ### END OF YOUR CODE
            self.estimators_.append(estimator)
            self.param.append(a)
            #print np.sum((self.pred - original_y)**2), a

        self.out_ = self._outliers(np.copy(grad))
        #self.feature_importances_ = self._calc_feature_imps()

        return self

    def _predict(self, X):
        # Получите ответ композиции до применения решающего правила
        ### YOUR CODE ###
        y_pred = np.zeros(shape=X.shape[0])
        for i in range(self.n_estimators - 1):
            y_pred += self.param[i] * self.estimators_[i].predict(X)

        return y_pred

    def predict(self, X):
        # Примените к self._predict решающее правило
        ### YOUR CODE ###
        y_pred = np.zeros(shape=X.shape[0])
        for i in range(self.n_estimators):
            y_pred += self.param[i] * self.estimators_[i].predict(X)
            
        #print np.sum((self.pred - y_pred)**2)
        #print X.shape

        return y_pred

    def _outliers(self, grad):
        # Топ-10 объектов с большим отступом
        ### YOUR CODE ###
        _outliers = []
        for i in range(10):
            k = np.argmax(grad)
            l = np.argmin(grad)
            _outliers.append(k)
            _outliers.append(l
            grad[k] = 0
            grad[l] = 0

        return _outliers

    def _calc_feature_imps(self):
        # Посчитайте self.feature_importances_ с помощью аналогичных полей у базовых алгоритмов
        f_imps = None
        ### YOUR CODE ###
        #for est in self.estimmators_:
            
        
        
        return f_imps/len(self.estimators_)
