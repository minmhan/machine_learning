# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 14:28:11 2016

@author: minmhan
"""
import numpy as np

class Perceptron(object):
    """ Perceptron classifier.
    Parameters:
    eta: float (Learning rate between 0.0 and 1.0)
    n_iter: int (Passes over the training dataset)
    
    Attributes:
    w_: 1D-Array (Weights after fitting)
    errors_: list (Number of misclassification in every epoch)
    """
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        
    def fit(self, X, Y):
        """ Fit training data
        Parameters
        X: {array-like}, shape = [n_samples, n_features]
        y: {array-like}, shape = [n_samples] Target values
        REturns:
        self: object
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, Y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
        
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
        
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)