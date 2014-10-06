#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NaiveBayes1 class

http://www.kamishima.net/mlmpyja/
"""

import numpy as np
from abc import ABCMeta, abstractmethod

# public symbols
__all__ = ['BaseBinaryNaiveBayes', 'NaiveBayes1']


class BaseBinaryNaiveBayes(object):
  """
  Abstract Class for Naive Bayes whose classes and features are binary.
  """

  __metaclass__ = ABCMeta

  def __init__(self):
    self.pY_ = None
    self.pXgY_ = None

  @abstractmethod
  def fit(self, X, y):
    """
    Abstract method for fitting model
    """
    pass

  def predict(self, X):
    """
    Predict class
    """

    n_samples = X.shape[0]
    n_features = X.shape[1]

    # memory for return values
    y = np.empty(n_samples, dtype=np.int)

    # for each feature in X
    for i, xi in enumerate(X):
      # calculate joint probability
      logpXY = np.log(self.pY_) + np.sum(np.log(self.pXgY_[np.arange(n_features), xi, :]), axis=0)
      # predict class
      y[i] = np.argmax(logpXY)

    return y


class NaiveBayes1(BaseBinaryNaiveBayes):
  """
  Naive Bayes class (1)
  """

  def __init__(self):
    super(NaiveBayes1, self).__init__()

  def fit(self, X, y):
    """
    Fitting model
    """

    n_samples = X.shape[0]
    n_features = X.shape[1]
    # TODO improve assumptions
    n_classes = 2 # C
    n_fvalues = 2 # K

    # check the size of y
    if n_samples != len(y):
      raise ValueError('Mismatched number of samples.')

    ## train class distribution
    # count up n[yi=y]
    nY = np.zeros(n_classes, dtype=np.int)
    for i in xrange(n_samples):
      nY[y[i]] += 1

    # calculate pY_
    self.pY_ = np.empty(n_classes, dtype=np.float)
    for yi in xrange(n_classes):
      self.pY_[yi] = nY[yi] / np.float(n_samples)

    ## train feature distribution
    # count up n[x_ij=xj, yi=y]
    nXY = np.zeros((n_features, n_fvalues, n_classes), dtype=np.int)
    for i in xrange(n_samples):
      for j in xrange(n_features):
        nXY[j, X[i,j], y[i]] += 1

    # calculate pXgY_
    self.pXgY_ = np.empty((n_features, n_fvalues, n_classes), dtype=np.float)
    for j in xrange(n_features):
      for xi in xrange(n_fvalues):
        for yi in xrange(n_classes):
          self.pXgY_[j, xi, yi] = nXY[j, xi, yi] / np.float(nY[yi])

