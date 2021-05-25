# -*- coding:utf-8 -*-
'''
__author__ = 'XD'
__mtime__ = 2021/1/22
__project__ = Pon-Sol2
Fix the Problem, Not the Blame.
'''
import os, sys
sys.path.append(os.path.abspath(".."))

import joblib
import pandas as pd
from ponsol2 import feature_extraction, config

A_LIST = ('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y')
class PonSol2:
    def __init__(self):
        """
        model
        self.Estimator: model
        self.kwargs: args of model
        self.special_kind: the special classes that divide the first layer
        self.fs1: feature of layer1
        self.fs2: feature of layer2

        """
        self.model_path = config.model_path
        self.model = joblib.load(self.model_path)
        self.fs1 = self.model.fs1
        self.fs2 = self.model.fs2

    def check_X(self, X):
        if not isinstance(X, pd.DataFrame):
            raise RuntimeError("The input is not the object of pandas.DataFrame")
        all_features = set(self.fs1.to_list() + self.fs2.to_list())
        input_data_features = set(X.columns.to_list())
        reduce_features = all_features - input_data_features
        if len(reduce_features) > 0:
            raise RuntimeError("lack feature:%s" % reduce_features)
        return True

    def predict(self, seq, aa):
        """
        预测
        :param seq: FASTA sequence without name
        :param aa: e.g. A1B
        :return: result
        """
        all_features = feature_extraction.get_all_features(seq, aa)
        if aa[0] == aa[-1]:
            return 0
        pred = self._predict(all_features)
        return pred


    def _predict(self, X):
        self.check_X(X)
        pred = self.model.predict(X)
        return pred
