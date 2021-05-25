# coding: utf-8

import pandas as pd
import numpy as np
import os
pd.options.display.max_rows = 10 
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, multilabel_confusion_matrix

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0) 

def ponsol_metrics(y_true, y_pre, balance=False, k=3):
    label = pd.DataFrame({'true': y_true, 'pre': y_pre})

    unique_state = label.true.unique()
    targets = {}  
    ii = io = id_ = oi = oo = od = di = do = dd = 0
    tpi = tni = fpi = fni = tpd = tnd = fpd = fnd = tpo = tno = fpo = fno = 0
    for i, (t, p) in label.iterrows():
      
        if t == -1 and p == -1:
            dd += 1
        if t == -1 and p == 0:
            do += 1
        if t == -1 and p == 1:
            di += 1
        if t == 0 and p == -1:
            od += 1
        if t == 0 and p == 0:
            oo += 1
        if t == 0 and p == 1:
            oi += 1
        if t == 1 and p == -1:
            id_ += 1
        if t == 1 and p == 0:
            io += 1
        if t == 1 and p == 1:
            ii += 1

    alli = ii + io + id_
    alld = di + do + dd
    allo = oi + oo + od

  
    if balance:
        ii = ii * allo / alli
        io = io * allo / alli
        id_ = id_ * allo / alli
        di = di * allo / alld
        do = do * allo / alld
        dd = dd * allo / alld

    
    acc = (ii + oo + dd) / (ii + io + id_ + oi + oo + od + di + do + dd)
    N = ii + io + id_ + oi + oo + od + di + do + dd

 
    eii = (ii + io + id_) * (ii + oi + di) / N
    eio = (ii + io + id_) * (io + oo + do) / N
    eid_ = (ii + io + id_) * (id_ + od + dd) / N

    eoi = (oi + oo + od) * (ii + oi + di) / N
    eoo = (oi + oo + od) * (io + oo + do) / N
    eod = (oi + oo + od) * (id_ + od + dd) / N

    edi = (di + do + dd) * (ii + oi + di) / N
    edo = (di + do + dd) * (io + oo + do) / N
    edd = (di + do + dd) * (id_ + od + dd) / N

   
    gc2 = None
    if 0 not in [eii, eio, eid_, eoi, eoo, eod, edi, edo, edd]:
        gc2 = (((ii - eii) * (ii - eii) / eii) + ((io - eio) * (io - eio) / eio) +
               ((id_ - eid_) * (id_ - eid_) / eid_) + ((oi - eoi) * (oi - eoi) / eoi) +
               ((oo - eoo) * (oo - eoo) / eoo) + ((od - eod) * (od - eod) / eod) +
               ((di - edi) * (di - edi) / edi) + ((do - edo) * (do - edo) / edo) +
               ((dd - edd) * (dd - edd) / edd)) / ((k - 1) * N)

  
    seni = ii / (ii + io + id_)
    send = dd / (di + do + dd)
    seno = oo / (oi + oo + od)

  
    spei = (dd + do + od + oo) / (dd + do + od + oo + di + oi)
    sped = (ii + io + oi + oo) / (ii + io + oi + oo + id_ + od)
    speo = (dd + di + id_ + ii) / (dd + di + id_ + ii + do + io)

   
    ppvi = ppvd = ppvo = None
    if ii + oi + di != 0:
        ppvi = ii / (ii + oi + di)
    if id_ + dd + od != 0:
        ppvd = dd / (id_ + dd + od)
    if io + do + oo != 0:
        ppvo = oo / (io + do + oo)

    
    npvi = npvd = npvo = None
    if dd + do + od + oo + io + id_ != 0:
        npvi = (dd + do + od + oo) / (dd + do + od + oo + io + id_)
    if ii + io + oi + oo + di + do != 0:
        npvd = (ii + io + oi + oo) / (ii + io + oi + oo + di + do)
    if dd + di + id_ + ii + od + oi != 0:
        npvo = (dd + di + id_ + ii) / (dd + di + id_ + ii + od + oi)

    
    tpi = ii
    tni = oo + od + do + dd
    fpi = oi + di
    fni = io + id_

    tpd = dd
    fnd = di + do
    fpd = id_ + od
    tnd = ii + io + oi + oo

    tpo = oo
    fno = oi + od
    fpo = io + do
    tno = ii + id_ + di + dd
    columns = ['tp', 'tn', 'fp', 'fn', 'ppv', 'npv', 'tpr', 'tnr']
    res2 = pd.DataFrame(
        [
            [tpd, tnd, fpd, fnd, ppvd, npvd, send, sped],
            [tpo, tno, fpo, fno, ppvo, npvo, seno, speo],
            [tpi, tni, fpi, fni, ppvi, npvi, seni, spei]
        ],
        columns=columns,
        index=[-1, 0, 1]
    )
    return acc, gc2, res2


def ponsol_metrics_new(y_true, y_pred, labels=[-1, 0, 1], verbose=0, balance=False):
    N = len(y_true)
    K = len(labels)
    mcms = multilabel_confusion_matrix(y_true, y_pred, labels=labels)
    res = pd.DataFrame([i.ravel() for i in mcms], columns="tn fp fn tp".split(), index=labels)
    res = res.loc[:, "tp  tn  fp  fn".split()]
    # PPV NPV TPR TNR
    res["ppv"] = res.tp / (res.tp + res.fp)
    res["npv"] = res.tn / (res.tn + res.fn)
    res["sensitivity"] = res.tp / (res.tp + res.fn)
    res["specificity"] = res.tn / (res.tn + res.fp)
    if verbose > 4:
        print("MCM = ")
        print(res)
        

    # gcc
    # zij, xi, yi
    zij = confusion_matrix(y_true, y_pred, labels=labels).astype(np.float)
    #     if balance:
    #         zij = zij * zij.sum(axis=1)[0] / np.tile(zij.sum(axis=1), (K, 1)).T

    xi = np.sum(zij, axis=1)  
    yi = np.sum(zij, axis=0)
    # xij, yij
    xij = xi.reshape(xi.shape[0], 1)
    xij = np.repeat(xij, xi.shape[0]).reshape((-1, xi.shape[0]))
    yij = np.array([yi for _ in range(yi.shape[0])])
    # zij
    eij = xij * yij / N
    gcc = np.sum((zij - eij) ** 2 / eij) / (N * (K - 1))
    acc = accuracy_score(y_true, y_pred)
    if verbose > 4:
        print("zij = \n%s" % zij)
        print("xi = \n%s" % xi)
        print("yi = \n%s" % yi)
        print("xij = \n%s" % xij)
        print("yij = \n%s" % yij)
        print("xij * yij = \n%s" % (xij * yij))
        print("eij = \n%s" % eij)
        print("gcc = %s" % gcc)
    if verbose > 0:
        print("acc:", acc)
        print("gcc:", gcc)
        print("res:")
        print(res)
    return acc, gcc, res


def format_metrics(metrics, name=None):
    """
    :param metrics: (accuracy, gc2, other)
    :param name: string
    :reutrn : 
             tag     
        fn   -1          28.500000
             0           19.500000
             1           26.600000
        fp   -1          32.000000
             0            9.900000
                           ...    
        tpr  -1           0.546416
             0            0.204500
             1            0.583755
        all  accuracy     0.506627
             gc2          0.061903
        Length: 26, dtype: float64
    """
    acc, gc2, other = metrics
    other = other.sort_index().unstack()
    other = other.append(pd.Series([acc, gc2], index=[['all', 'all'], ['accuracy', 'gc2']]))
    other.name = name
    return other



class CVUtil:
    def __init__(self, rfc, name, cv_method, n_cv=10, feature_select=None):
        self.rfc = rfc
        self.name = name
        self.cv_method = cv_method
        self.n_cv = n_cv
        self.feature_select = feature_select

    def set_data(self, X_train, y_train, g_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.g_train = np.array(g_train)

        if self.feature_select:
            self.X_train = self.feature_select.transform(self.X_train)
        return self

    def fit(self):
        self.normal, self.balance = self._cv()
        return self

    def get_res(self, kind):
        # nomal
        acc, gc2, metr = self.normal
        acc = np.mean(acc)
        gc2 = np.mean(gc2)
        metr = metr.groupby('tag').mean()
        res = metr.unstack()
        res1 = res.append(pd.Series([acc, gc2], index=[['all', 'all'], ['accuracy', 'gc2']]))
        # res1 = res1.map(lambda x: round(x, 3))
        res1.name = self.name

        # balance
        acc, gc2, metr = self.balance
        acc = np.mean(acc)
        gc2 = np.mean(gc2)
        metr = metr.groupby('tag').mean()
        res = metr.unstack()
        res2 = res.append(pd.Series([acc, gc2], index=[['all', 'all'], ['accuracy', 'gc2']]))
        # res2 = res2.map(lambda x: round(x, 3))
        res2.name = self.name

        res3 = []
        for i in range(len(res2)):
            if '%.3f' % res1.iloc[i] == '%.3f' % res2.iloc[i]:
                res3.append('%.3f' % res1.iloc[i])
            else:
                res3.append('%.3f/%.3f' % (res1.iloc[i], res2.iloc[i]))
        res3 = pd.Series(res3, index=res2.index)
        res3.name = self.name
        if kind == 1:
            return res1
        elif kind == 2:
            return res2
        elif kind == 3:
            return res3
        else:
            raise RuntimeError("kind in [1, 2, 3]")

    def _cv(self):

        _value_counts = pd.Series(self.y_train).value_counts().reindex([-1, 0, 1])
        # result
        ## normal
        cv_accuracy = [] 
        cv_gc2 = [] 
        cv_metrics = [] 
        ## balance
        cv_accuracy_balance = []  
        cv_gc2_balance = [] 
        cv_metrics_balance = [] 
        for (index, (train_index, test_index)) in enumerate(
                self.cv_method.split(self.X_train, self.y_train, groups=self.g_train)):
            cv_X_train, cv_X_test = self.X_train[train_index], self.X_train[test_index]
            cv_y_train, cv_y_test = self.y_train[train_index], self.y_train[test_index]

            _value_counts = pd.Series(cv_y_train).value_counts().reindex([-1, 0, 1])

            _value_counts = pd.Series(cv_y_test).value_counts().reindex([-1, 0, 1])
            _rfc = self.rfc.fit(cv_X_train, cv_y_train)
            p_test = _rfc.predict(cv_X_test)
            ## normal
            acc, gc2, metr = ponsol_metrics(cv_y_test, p_test)
            cv_accuracy.append(acc)
            cv_gc2.append(gc2)
            metr['tag'] = metr.index
            metr['cv'] = 'cv%s' % (index + 1)
            cv_metrics.append(metr)
            ## balance
            acc, gc2, metr = ponsol_metrics(cv_y_test, p_test, balance=True)
            cv_accuracy_balance.append(acc)
            cv_gc2_balance.append(gc2)
            metr['tag'] = metr.index
            metr['cv'] = 'cv%s' % (index + 1)
            cv_metrics_balance.append(metr)
        ## normal
        res_metr = pd.concat(cv_metrics)
        res_metr.name = self.name
        res_acc = cv_accuracy
        res_gc2 = cv_gc2
        ## balance
        res_metr_balance = pd.concat(cv_metrics_balance)
        res_metr_balance.name = self.name
        res_acc_balance = cv_accuracy_balance
        res_gc2_balance = cv_gc2_balance
        return [[res_acc, res_gc2, res_metr], [res_acc_balance, res_gc2_balance, res_metr_balance]]



class BlindTestUtil:

    def __init__(self, rfc, name='', feature_select=None):
        self.rfc = rfc
        self.name = name

        self.feature_select = feature_select

    def set_data(self, X_train, y_train, X_test1, y_test1, X_test2, y_test2):

        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_test1 = np.array(X_test1)
        self.y_test1 = np.array(y_test1)
        self.X_test2 = np.array(X_test2)
        self.y_test2 = np.array(y_test2)

        if self.feature_select:
            self.X_train = self.feature_select.transform(self.X_train)
            self.X_test1 = self.feature_select.transform(self.X_test1)
            self.X_test2 = self.feature_select.transform(self.X_test2)

        return self

    def fit(self):
        self.test1, self.test1_balance, self.test2, self.test2_balance = self._test(
        )
        return self

    def get_res(self, kind):
        # test1
        acc, gc2, metr = self.test1
        res = metr.unstack()
        res1 = res.append(
            pd.Series([acc, gc2], index=[['all', 'all'], ['accuracy', 'gc2']]))
        res1.name = self.name + '_test1'
        acc, gc2, metr = self.test1_balance
        res = metr.unstack()
        res1_balance = res.append(
            pd.Series([acc, gc2], index=[['all', 'all'], ['accuracy', 'gc2']]))
        res1_balance.name = self.name + '_test1_balance'
        res1_all = pd.concat([res1, res1_balance], axis=1)

        # test2
        acc, gc2, metr = self.test2
        res = metr.unstack()
        res2 = res.append(
            pd.Series([acc, gc2], index=[['all', 'all'], ['accuracy', 'gc2']]))
        res2.name = self.name + '_test2'
        acc, gc2, metr = self.test2_balance
        res = metr.unstack()
        res2_balance = res.append(
            pd.Series([acc, gc2], index=[['all', 'all'], ['accuracy', 'gc2']]))
        res2_balance.name = self.name + '_test2_balance'
        res2_all = pd.concat([res2, res2_balance], axis=1)

        ## test1
        res3_1 = []
        for i in range(len(res1)):
            if '%.3f' % res1.iloc[i] == '%.3f' % res2.iloc[i]:
                res3_1.append('%.3f' % res1.iloc[i])
            else:
                res3_1.append('%.3f/%.3f' %
                              (res1.iloc[i], res1_balance.iloc[i]))
        res3_1 = pd.Series(res3_1, index=res1.index)
        res3_1.name = self.name + '_test1'
        ## test2
        res3_2 = []
        for i in range(len(res2)):
            if res2.iloc[i] == res2_balance.iloc[i]:
                res3_2.append('%.3f' % res2.iloc[i])
            else:
                res3_2.append('%.3f/%.3f' %
                              (res2.iloc[i], res2_balance.iloc[i]))
        res3_2 = pd.Series(res3_2, index=res2.index)
        res3_2.name = self.name + '_test2'
        res3_all = pd.concat([res3_1, res3_2], axis=1)

        if kind == 1:
            return res1_all
        elif kind == 2:
            return res2_all
        elif kind == 3:
            return res3_all
        else:
            raise RuntimeError("kind in [1, 2, 3]")

    def _test(self):
        _rfc = self.rfc.fit(self.X_train, self.y_train)
        p_test1 = _rfc.predict(self.X_test1)
        p_test2 = _rfc.predict(self.X_test2)
        # test1
        res1 = ponsol_metrics(self.y_test1, p_test1)
        res1[2].name = self.name + '_test1'
        # test1 balance
        res1_balance = ponsol_metrics(self.y_test1, p_test1, balance=True)
        res1_balance[2].name = self.name + '_test1_balance'
        # test2
        res2 = ponsol_metrics(self.y_test2, p_test2)
        res2[2].name = self.name + '_test2'
        # test2 balca
        res2_balance = ponsol_metrics(self.y_test2, p_test2, balance=True)
        res2_balance[2].name = self.name + '_test2_balance'
        return res1, res1_balance, res2, res2_balance


class PonsolLayerEstimator:
    def __init__(self, estimator, feature_selected=None, special_kind=-1, kwargs=None):
        self.Estimator = estimator
        self.feature_selected = feature_selected
        self.kwargs = kwargs
        self.special_kind = special_kind

    def fit(self, X, y):
        y = np.array(y)
        if self.feature_selected is not None:
            self.fs1 = self.feature_selected[0]
            self.fs2 = self.feature_selected[1]
        else:
            self.fs1 = X.columns.values
            self.fs2 = X.columns.values
        index_layer_1 = (y == self.special_kind)
        index_layer_2 = (y != self.special_kind)
        X1 = X.loc[:, self.fs1]
        y1 = np.array(y)
        y1[index_layer_2] = self.special_kind + 1
        X2 = X.loc[index_layer_2, self.fs2]
        y2 = y[index_layer_2]
        estimator1 = self.Estimator(**self.kwargs)
        self.estimator1 = estimator1.fit(X1, y1)
        estimator2 = self.Estimator(**self.kwargs)
        self.estimator2 = estimator2.fit(X2, y2)
        return self

    def predict(self, X):
        X1 = X.loc[:, self.fs1]
        p_layer1 = self.estimator1.predict(X1)
        p_layer1 = np.array(p_layer1)

        index_layer1_is_special = p_layer1 == self.special_kind
        index_layer1_not_special = p_layer1 != self.special_kind

        X2 = X.loc[index_layer1_not_special, self.fs2]

        p_layer2 = self.estimator2.predict(X2)

        p_all = np.full(X.shape[0], None)
        p_all[index_layer1_is_special] = p_layer1[index_layer1_is_special]
        p_all[index_layer1_not_special] = p_layer2

        return p_all



class CVUtilLayer2:
    def __init__(self, Estimator, cvs_method, name="", n_cv=10, kwargs=None):

        self.Estimator = Estimator
        self.kwargs = kwargs
        self.cvs_method = cvs_method
        self.name = name
        self.n_cv = n_cv

    def set_data(self, X_train, y_train, g_train, special_layer_kind, features_for_layer_1=None,
                 features_for_layer_2=None):

        self.X_train = X_train
        self.y_train = y_train
        self.g_train = g_train
        self.special_layer_kind = special_layer_kind
        self.features_for_layer_1 = features_for_layer_1
        self.features_for_layer_2 = features_for_layer_2
        return self

    def fit(self):

        self.normal, self.balance = self._cv(
            self.special_layer_kind,
            self.X_train,
            self.y_train,
            self.cvs_method,
            self.features_for_layer_1,
            self.features_for_layer_2
        )
        return self

    def get_res(self, kind):

        # nomal
        acc, gc2, metr = self.normal
        acc = np.mean(acc)
        gc2 = np.mean(gc2)
        metr = metr.groupby('tag').mean()
        res = metr.unstack()
        res1 = res.append(pd.Series([acc, gc2], index=[['all', 'all'], ['accuracy', 'gc2']]))
        # res1 = res1.map(lambda x: round(x, 3))
        res1.name = self.name

        # balance
        acc, gc2, metr = self.balance
        acc = np.mean(acc)
        gc2 = np.mean(gc2)
        metr = metr.groupby('tag').mean()
        res = metr.unstack()
        res2 = res.append(pd.Series([acc, gc2], index=[['all', 'all'], ['accuracy', 'gc2']]))
        # res2 = res2.map(lambda x: round(x, 3))
        res2.name = self.name

        res3 = []
        for i in range(len(res2)):
            if '%.3f' % res1.iloc[i] == '%.3f' % res2.iloc[i]:
                res3.append('%.3f' % res1.iloc[i])
            else:
                res3.append('%.3f/%.3f' % (res1.iloc[i], res2.iloc[i]))
        res3 = pd.Series(res3, index=res2.index)
        res3.name = self.name

        if kind == 1:
            return res1
        elif kind == 2:
            return res2
        elif kind == 3:
            return res3
        else:
            raise RuntimeError("kind in [1, 2, 3]")

    def _cv(self, special_layer_kind,
            X_train,
            y_train,
            cvs_method,
            features_for_layer_1=None,
            features_for_layer_2=None,
            **args):


        if (features_for_layer_1 is None):
            features_for_layer_1 = X_train.columns.values
        if (features_for_layer_2 is None):
            features_for_layer_2 = X_train.columns.values
        layer_name = {
            0: special_layer_kind,
            1: [i for i in [-1, 0, 1] if i != special_layer_kind]
        }

        cv_accuracy = [] 
        cv_gc2 = [] 
        cv_metrics = []  

        cv_accuracy_balance = []  
        cv_gc2_balance = [] 
        cv_metrics_balance = []

        for (index, (train_index,
                     test_index)) in enumerate(cvs_method.split(X_train, y_train)):
            cv_X_train, cv_X_validate = X_train.iloc[train_index, :], X_train.iloc[
                                                                      test_index, :]
            cv_y_train, cv_y_validate = y_train.iloc[train_index], y_train.iloc[
                test_index]

            Estimator = self.Estimator
            if self.kwargs is None:
                kwargs = {"random_state": 0, }
            else:
                kwargs = self.kwargs
            layer_estimator = PonsolLayerEstimator(Estimator, kwargs=kwargs, special_kind=special_layer_kind,
                                                   feature_selected=[features_for_layer_1, features_for_layer_2])
            layer_estimator.fit(cv_X_train, cv_y_train)
            cv_y_pred = layer_estimator.predict(cv_X_validate)
            cv_y_true = cv_y_validate

            ## normal
            acc, gc2, metr = ponsol_metrics(cv_y_true, cv_y_pred)
            cv_accuracy.append(acc)
            cv_gc2.append(gc2)
            metr['tag'] = metr.index
            metr['cv'] = 'cv%s' % (index + 1)
            cv_metrics.append(metr)
            ## balance
            acc, gc2, metr = ponsol_metrics(cv_y_true, cv_y_pred, True)
            cv_accuracy_balance.append(acc)
            cv_gc2_balance.append(gc2)
            metr['tag'] = metr.index
            metr['cv'] = 'cv%s' % (index + 1)
            cv_metrics_balance.append(metr)

        ## normal
        res_metr = pd.concat(cv_metrics)
        res_metr.name = self.name
        res_acc = cv_accuracy
        res_gc2 = cv_gc2
        ## balance
        res_metr_balance = pd.concat(cv_metrics_balance)
        res_metr_balance.name = self.name
        res_acc_balance = cv_accuracy_balance
        res_gc2_balance = cv_gc2_balance
        return [[res_acc, res_gc2, res_metr], [res_acc_balance, res_gc2_balance, res_metr_balance]]



class BlindTestUtilLayer2:
    def __init__(self, Estimator, name="", kwargs=None):
        self.Estimator = Estimator
        self.kwargs = kwargs
        self.name = name

    def set_data(self,
                 X_train,
                 y_train,
                 X_test1,
                 y_test1,
                 X_test2,
                 y_test2,
                 special_layer_kind,
                 features_for_layer_1=None,
                 features_for_layer_2=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test1 = X_test1
        self.y_test1 = y_test1
        self.X_test2 = X_test2
        self.y_test2 = y_test2
        self.special_layer_kind = special_layer_kind
        self.features_for_layer_1 = features_for_layer_1
        self.features_for_layer_2 = features_for_layer_2
        return self

    def fit(self):
        self.test1, self.test1_balance, self.test2, self.test2_balance = self._test(
            self.special_layer_kind, self.X_train, self.y_train,
            self.X_test1, self.y_test1, self.X_test2, self.y_test2,
            self.features_for_layer_1,
            self.features_for_layer_2)
        return self

    def get_res(self, kind):
        # test1
        acc, gc2, metr = self.test1
        res = metr.unstack()
        res1 = res.append(
            pd.Series([acc, gc2], index=[['all', 'all'], ['accuracy', 'gc2']]))
        res1.name = self.name + '_test1'
        acc, gc2, metr = self.test1_balance
        res = metr.unstack()
        res1_balance = res.append(
            pd.Series([acc, gc2], index=[['all', 'all'], ['accuracy', 'gc2']]))
        res1_balance.name = self.name + '_test1_balance'
        res1_all = pd.concat([res1, res1_balance], axis=1)

        # test2
        acc, gc2, metr = self.test2
        res = metr.unstack()
        res2 = res.append(
            pd.Series([acc, gc2], index=[['all', 'all'], ['accuracy', 'gc2']]))
        res2.name = self.name + '_test2'
        acc, gc2, metr = self.test2_balance
        res = metr.unstack()
        res2_balance = res.append(
            pd.Series([acc, gc2], index=[['all', 'all'], ['accuracy', 'gc2']]))
        res2_balance.name = self.name + '_test2_balance'
        res2_all = pd.concat([res2, res2_balance], axis=1)

        ## test1
        res3_1 = []
        for i in range(len(res1)):
            if '%.3f' % res1.iloc[i] == '%.3f' % res2.iloc[i]:
                res3_1.append('%.3f' % res1.iloc[i])
            else:
                res3_1.append('%.3f/%.3f' %
                              (res1.iloc[i], res1_balance.iloc[i]))
        res3_1 = pd.Series(res3_1, index=res1.index)
        res3_1.name = self.name + '_test1'
        ## test2
        res3_2 = []
        for i in range(len(res2)):
            if res2.iloc[i] == res2_balance.iloc[i]:
                res3_2.append('%.3f' % res2.iloc[i])
            else:
                res3_2.append('%.3f/%.3f' %
                              (res2.iloc[i], res2_balance.iloc[i]))
        res3_2 = pd.Series(res3_2, index=res2.index)
        res3_2.name = self.name + '_test2'
        res3_all = pd.concat([res3_1, res3_2], axis=1)

        if kind == 1:
            return res1_all
        elif kind == 2:
            return res2_all
        elif kind == 3:
            return res3_all
        else:
            raise RuntimeError("kind in [1, 2, 3]")

    def _test(self,
              special_layer_kind,
              X_train,
              y_train,
              X_test1,
              y_test1,
              X_test2,
              y_test2,
              features_for_layer_1=None,
              features_for_layer_2=None,
              **args):

        if (features_for_layer_1 is None):
            features_for_layer_1 = X_train.columns
        if (features_for_layer_2 is None):
            features_for_layer_2 = X_train.columns

        layer_name = {
            0: special_layer_kind,
            1: [i for i in [-1, 0, 1] if i != special_layer_kind]
        }
        res = []
        y_train_blind, y_test1_blind, y_test2_blind = y_train, y_test1, y_test2
        if self.kwargs is None:
            kwargs = {"random_state": 0, }
        else:
            kwargs = self.kwargs
        Estimator = self.Estimator
        layer_estimator = PonsolLayerEstimator(Estimator, kwargs=kwargs, special_kind=special_layer_kind,
                                               feature_selected=[features_for_layer_1, features_for_layer_2])
        layer_estimator.fit(X_train, y_train)

        # test1
        y_true_test1 = y_test1
        y_pred_test1 = layer_estimator.predict(X_test1)


        res1 = ponsol_metrics(y_true_test1,
                              y_pred_test1)
        res1[2].name = self.name + '_test1'
        # test1 balance
        res1_balance = ponsol_metrics(y_true_test1,
                                      y_pred_test1, True)
        res1_balance[2].name = self.name + '_test1_balance'

        acc, gc2, metr = ponsol_metrics(y_true_test1,
                                        y_pred_test1)
        res.append((acc, gc2, metr))

        # test2
        y_true_test2 = y_test2
        y_pred_test2 = layer_estimator.predict(X_test2)


        res2 = ponsol_metrics(y_true_test2,
                              y_pred_test2)
        res2[2].name = self.name + '_test2'
        # test1 balance
        res2_balance = ponsol_metrics(y_true_test2,
                                      y_pred_test2, True)
        res2_balance[2].name = self.name + '_test2_balance'


        acc, gc2, metr = ponsol_metrics(y_true_test2,
                                        y_pred_test2, )

        return res1, res1_balance, res2, res2_balance


def _result_output(res_cv, res_blind, names, path, filename):
    _res_all = []
    _res_all.append([
        *[res_cv[i].get_res(1)[("all", "accuracy")] for i in range(len(names))],  # acc
        *[res_cv[i].get_res(1)[("all", "gc2")] for i in range(len(names))],  # gc2
    ])
    _res_all.append([
        *[res_blind[i].get_res(1).iloc[:, 0][("all", "accuracy")] for i in range(len(names))],
        *[res_blind[i].get_res(1).iloc[:, 0][("all", "gc2")] for i in range(len(names))]
    ])
    _res_all.append([
        *[res_blind[i].get_res(2).iloc[:, 0][("all", "accuracy")] for i in range(len(names))],
        *[res_blind[i].get_res(2).iloc[:, 0][("all", "gc2")] for i in range(len(names))]
    ])
    dp_overall = pd.DataFrame(_res_all, columns=pd.MultiIndex.from_product([["acc", "gc2"], names]),
                              index=["10cv", "test1", "test2"])

    # acc
    dp_acc_detail = pd.concat(
        [
            pd.DataFrame([
                *res_cv[i].normal[0],  # 10cv
                res_blind[i].get_res(1).iloc[:, 0][("all", "accuracy")],  # test1
                res_blind[i].get_res(2).iloc[:, 0][("all", "accuracy")],  # test2
            ]) for i in range(len(names))
        ],
        axis=1, )
    dp_acc_detail.index = [*["cv{}".format(i) for i in range(1, 11)], "test1", "test2"]
    dp_acc_detail.columns = names

    # gc2
    dp_gc2_detail = pd.concat(
        [
            pd.DataFrame([
                *res_cv[i].normal[1],  # 10cv
                res_blind[i].get_res(1).iloc[:, 0][("all", "gc2")],  # test1
                res_blind[i].get_res(2).iloc[:, 0][("all", "gc2")],  # test2
            ]) for i in range(len(names))
        ],
        axis=1, )
    dp_gc2_detail.index = [*["cv{}".format(i) for i in range(1, 11)], "test1", "test2"]
    dp_gc2_detail.columns = names

    dp_metr_detail = pd.concat([pd.concat(
        [
            res_cv[i].get_res(1).unstack().unstack().unstack().loc[
                [-1, 0, 1], "tp	tn	fp	fn	ppv	npv	tpr	tnr".split()],
            res_blind[i].get_res(1).iloc[:, 0].unstack().unstack().unstack().loc[
                [-1, 0, 1], "tp	tn	fp	fn	ppv	npv	tpr	tnr".split()],
            res_blind[i].get_res(2).iloc[:, 0].unstack().unstack().unstack().loc[
                [-1, 0, 1], "tp	tn	fp	fn	ppv	npv	tpr	tnr".split()]
        ],
        keys=["10cv", "test1", "test2"]
    ) for i in range(len(names))], keys=names)

    _path = os.path.join(path, "{}_res.xlsx".format(filename))
    with pd.ExcelWriter(_path) as writer:
        dp_overall.to_excel(writer, sheet_name="overall", float_format="%.3f")
        dp_acc_detail.to_excel(writer, sheet_name="acc_detail", float_format="%.3f")
        dp_gc2_detail.to_excel(writer, sheet_name="acc_gc2", float_format="%.3f")
        dp_metr_detail.to_excel(writer, sheet_name="metr_detail", float_format="%.3f")
    return dp_overall, dp_acc_detail, dp_gc2_detail, dp_metr_detail


def _result_balance_output(res_cv, res_blind, names, path, filename):
    _res_all = []
    _res_all.append([
        *[res_cv[i].get_res(2)[("all", "accuracy")] for i in range(len(names))],  # acc
        *[res_cv[i].get_res(2)[("all", "gc2")] for i in range(len(names))],  # gc2
    ])
    _res_all.append([
        *[res_blind[i].get_res(1).iloc[:, 1][("all", "accuracy")] for i in range(len(names))],
        *[res_blind[i].get_res(1).iloc[:, 1][("all", "gc2")] for i in range(len(names))]
    ])
    _res_all.append([
        *[res_blind[i].get_res(2).iloc[:, 1][("all", "accuracy")] for i in range(len(names))],
        *[res_blind[i].get_res(2).iloc[:, 1][("all", "gc2")] for i in range(len(names))]
    ])
    dp_overall_balance = pd.DataFrame(_res_all, columns=pd.MultiIndex.from_product([["acc", "gc2"], names]),
                                      index=["10cv", "test1", "test2"])

    # acc
    dp_acc_detail_balance = pd.concat(
        [
            pd.DataFrame([
                *res_cv[i].balance[0],  # 10cv
                res_blind[i].get_res(1).iloc[:, 1][("all", "accuracy")],  # test1
                res_blind[i].get_res(2).iloc[:, 1][("all", "accuracy")],  # test2
            ]) for i in range(len(names))
        ],
        axis=1, )
    dp_acc_detail_balance.index = [*["cv{}".format(i) for i in range(1, 11)], "test1", "test2"]
    dp_acc_detail_balance.columns = names

    # gc2
    dp_gc2_detail_balance = pd.concat(
        [
            pd.DataFrame([
                *res_cv[i].balance[1],  # 10cv
                res_blind[i].get_res(1).iloc[:, 1][("all", "gc2")],  # test1
                res_blind[i].get_res(2).iloc[:, 1][("all", "gc2")],  # test2
            ]) for i in range(len(names))
        ],
        axis=1, )
    dp_gc2_detail_balance.index = [*["cv{}".format(i) for i in range(1, 11)], "test1", "test2"]
    dp_gc2_detail_balance.columns = names

    dp_metr_detail_balance = pd.concat([pd.concat(
        [
            res_cv[i].get_res(2).unstack().unstack().unstack().loc[
                [-1, 0, 1], "tp	tn	fp	fn	ppv	npv	tpr	tnr".split()],
            res_blind[i].get_res(1).iloc[:, 1].unstack().unstack().unstack().loc[
                [-1, 0, 1], "tp	tn	fp	fn	ppv	npv	tpr	tnr".split()],
            res_blind[i].get_res(2).iloc[:, 1].unstack().unstack().unstack().loc[
                [-1, 0, 1], "tp	tn	fp	fn	ppv	npv	tpr	tnr".split()]
        ],
        keys=["10cv", "test1", "test2"]
    ) for i in range(len(names))], keys=names)

    _path = os.path.join(path, "{}_res_balanced.xlsx".format(filename))
    with pd.ExcelWriter(_path) as writer:
        dp_overall_balance.to_excel(writer, sheet_name="overall", float_format="%.3f")
        dp_acc_detail_balance.to_excel(writer, sheet_name="acc_detail", float_format="%.3f")
        dp_gc2_detail_balance.to_excel(writer, sheet_name="acc_gc2", float_format="%.3f")
        dp_metr_detail_balance.to_excel(writer, sheet_name="metr_detail", float_format="%.3f")
    return dp_overall_balance, dp_acc_detail_balance, dp_gc2_detail_balance, dp_metr_detail_balance


def result_output(res_cv, res_blind, names, path, filename, is_res_balance=False):
    if is_res_balance:
        return _result_balance_output(res_cv, res_blind, names, path, filename)
    else:
        return _result_output(res_cv, res_blind, names, path, filename)

def solubility_distribute(y, i=0, verb=0):
    _vc = pd.value_counts(y)
    _index = _vc.index.sort_values()
    _vc = _vc[_index]
    _str = ": ".join(map(str, _index)) + " = " + ": ".join(map(str, _vc.values)) + "= " + ": ".join(
        map(lambda x: "%.2f" % x, (_vc / _vc.iloc[i]).values))
    if (verb > 0):
        print(_str)
    return _str
