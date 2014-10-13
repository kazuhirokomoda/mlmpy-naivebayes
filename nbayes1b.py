#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NaiveBayes1 class

http://www.kamishima.net/mlmpyja/
"""

import numpy as np
from abc import ABCMeta, abstractmethod

# public symbols
__all__ = ['BaseBinaryNaiveBayes', 'NaiveBayes1', 'NaiveBayes2']


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


class NaiveBayes2(BaseBinaryNaiveBayes):
  """
  Naive Bayes class (2)
  """

  def __init__(self):
    super(NaiveBayes2, self).__init__()

  def fit(self, X, y):
    """
    Fitting model

    ブロードキャスト機能を用いてfor文を減らす方法

    0. 最終結果を格納する配列を構成する次元に加え、データサンプルの次元も加えたfor文を作成する．(1)->(2) 
    1. 出力配列の次元数を for ループの数とする．
    2. 各 for ループごとに，出力配列の次元を割り当てる．
    3. 計算に必要な配列を生成する．このとき，ループ変数がループに割り当てた次元に対応するようにする．
    4. 冗長な配列を整理統合する．
    5. 要素ごとの演算をユニバーサル関数の機能を用いて実行する．
    6. np.sum() などの集約演算を適用して，最終結果を得る．    
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

    # (1) クラスの対応する添え字の要素のカウンタを一つずつ増やす
    """
    nY = np.zeros(n_classes, dtype=np.int)
    for i in xrange(n_samples):
      nY[y[i]] += 1
    """

    # (2) 各クラスごとに，現在の対象クラスの事例であったなら対応する要素のカウンタを一つずつ増やす
    """
    nY = np.zeros(n_classes, dtype=np.int)
    for yi in xrange(n_classes):
      for i in xrange(n_samples):
        if y[i] == yi:
          nY[yi] += 1
      # using universal function np.equal(==) and np.sum,
      # i for-loop can be eliminated. (nY can be defined as np.empty)
      #nY[yi] = np.sum(y==yi)
    """

    # (3) eliminating for loop from (2).
    # ary_i = np.arange(n_samples)[:, np.newaxis]
    # ary_y = y[ary_i]
    ary_y = y[:, np.newaxis]
    ary_yi = np.arange(n_classes)[np.newaxis, :]
    cmp_y = (ary_y == ary_yi)
    nY = np.sum(cmp_y, axis=0)


    # calculate pY_
    # (1)
    """
    self.pY_ = np.empty(n_classes, dtype=np.float)
    for yi in xrange(n_classes):
      self.pY_[yi] = nY[yi] / np.float(n_samples)
    """
    # (2)
    self.pY_ = np.true_divide(nY, n_samples)


    ## train feature distribution
    # count up n[x_ij=xj, yi=y]
    # (1)
    """
    nXY = np.zeros((n_features, n_fvalues, n_classes), dtype=np.int)
    for i in xrange(n_samples):
      for j in xrange(n_features):
        nXY[j, X[i,j], y[i]] += 1
    """

    # (2) クラスの分布の場合と同様に，各特徴値ごとに，対象の特徴値の場合にのみカウンタを増やすような実装にします．
    """
    nXY = np.zeros((n_features, n_fvalues, n_classes), dtype=np.int)
    for i in xrange(n_samples):
      for j in xrange(n_features):
        for yi in xrange(n_classes): # yiとxiのforループの順番は、逆の方がむしろnXYの次元の順番と整合性が取れる？
          for xi in xrange(n_fvalues):
            if y[i]==yi and X[i,j]==xi:
              nXY[j, xi, yi] += 1
    """

    # (3) eliminating for loop from (2).
    ary_xi = np.arange(n_fvalues)[np.newaxis, np.newaxis, :, np.newaxis]
    ary_yi = np.arange(n_classes)[np.newaxis, np.newaxis, np.newaxis, :]
    #ary_i = np.arange(n_samples)[:, np.newaxis, np.newaxis, np.newaxis]
    #ary_y = y[ary_i]
    ary_y = y[:, np.newaxis, np.newaxis, np.newaxis]
    # X の要素を，ループを割り当てた次元に対応するように配置した配列を直接的に生成する
    ary_X = X[:, :, np.newaxis, np.newaxis]

    cmp_X = (ary_X == ary_xi)
    cmp_y = (ary_y == ary_yi)
    cmp_Xandy = np.logical_and(cmp_X, cmp_y)

    nXY = np.sum(cmp_Xandy, axis=0)


    # calculate pXgY_
    # (1)
    """
    self.pXgY_ = np.empty((n_features, n_fvalues, n_classes), dtype=np.float)
    for j in xrange(n_features):
      for xi in xrange(n_fvalues):
        for yi in xrange(n_classes):
          self.pXgY_[j, xi, yi] = nXY[j, xi, yi] / np.float(nY[yi])
    """

    # (2)
    self.pXgY_ = np.true_divide(nXY, nY[np.newaxis, np.newaxis, :])
