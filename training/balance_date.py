import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import re
import json
from pprint import pprint
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold, GroupShuffleSplit
from sklearn.model_selection import train_test_split
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)  
gss = GroupShuffleSplit(n_splits=10, random_state=0)  

with open("./config.json") as f:
    config = json.loads(f.read())
print("config")
pprint(config)


from imblearn import over_sampling
from imblearn import under_sampling
from collections import Counter

feature_path = config["all_feature"]  
feature = pd.read_csv(feature_path, index_col=0)


feature.insert(loc=0, column="index", value=feature.index)  


new_gi = ['213133708', '190610225', '118137700', '1236048426', '40789249', '364505958'] 
large_gi = ['190610225', '213133708']  
new_not_large_gi = []
for i in new_gi:
    if i not in large_gi:
        new_not_large_gi.append(i)

old_gi = [i for i in feature.gi if i not in new_gi]  
old_gi_test = ['145563', '730948', '1684918', '2668592', '40796138', '67471959', '82581658', '116666949', '326936885']  
old_gi_train = []
for i in feature.gi.drop_duplicates().values:
    i = str(i)
    if (i not in large_gi) and (i not in old_gi_test):
        old_gi_train.append(i)
feature



def solubility_distribute(y, i=0,verb=0):
    _vc = pd.value_counts(y)
    _index = _vc.index.sort_values()
    _vc = _vc[_index]
    _str = ": ".join(map(str, _index)) + " = " + ": ".join(map(str, _vc.values)) + "= " + ": ".join(map(lambda x: "%.2f" % x, (_vc / _vc.iloc[i]).values))
    if (verb > 0):
        print(_str)
    return _str





feature_of_lgk = feature[feature.gi == int(large_gi[0])]
feature_of_tem = feature[feature.gi == int(large_gi[1])]
feature_of_not_lgk_tem = feature[(feature.gi != int(large_gi[0])) & (feature.gi != int(large_gi[1]))]



yeast_lgk = pd.read_csv(config['yeast_lgk'], index_col=0)
index_reduce_lgk = ((yeast_lgk.Sol_Score <= -1.0) | (yeast_lgk.Sol_Score >= -0.15))
reduce_yeast_lgk = yeast_lgk[index_reduce_lgk]


yeast_tem = pd.read_csv(config['yeast_tem'], index_col=0)
index_reduce_tem = ((yeast_tem.Sol_Score <= -1.0) | (yeast_tem.Sol_Score >= -0.15))
reduce_yeast_tem = yeast_tem[index_reduce_tem]



feature = pd.concat([
    feature_of_lgk.loc[index_reduce_lgk.values, :],
    feature_of_tem.loc[index_reduce_tem.values, :],
    feature_of_not_lgk_tem,
], axis=0)
feature = feature[feature.mut_from != feature.mut_to]  


old_test_feature = feature[feature.gi.apply(lambda x: str(x) in old_gi_test)]  
feature_not_old_test = feature.loc[feature.index.difference(old_test_feature.index)]


X_test1 = old_test_feature
y_test1 = old_test_feature.solubility
y_test1 = list(y_test1)

X_train1 = feature_not_old_test
y_train1 = feature_not_old_test.solubility


X_test2_1 = X_train1[X_train1.mut_residue.apply(lambda x: x in  set(X_test1.mut_residue.to_list()))]
X_test2 = pd.concat([X_test2_1, X_test1], axis=0)
y_test2 = X_test2.solubility
X_train2 = X_train1[X_train1.mut_residue.apply(lambda x: x not in  set(X_test1.mut_residue.to_list()))]
y_train2 = X_train2.solubility


X_train = X_train2
y_train = y_train2
y_train = list(y_train)


class BalanceDate():
    X_test1 = pd.DataFrame(X_test1)
    X_test2 = pd.DataFrame(X_test2)
    X_train = pd.DataFrame(X_train)
    y_test1 = pd.Series(y_test1)
    y_test2 = pd.Series(y_test2)
    y_train = pd.Series(y_train)
    FEATURE = pd.Series(X_test1.columns[10:])
    
    @classmethod
    def get_feature(cls):
        return cls.FEATURE
    @classmethod
    def get_test(cls, kind=0):
        if kind == 0:
            return cls.X_test1, cls.y_test1, cls.X_test2, cls.y_test2
        elif kind == 1:
            return cls.X_test1, cls.y_test1
        elif kind == 2:
            return cls.X_test2, cls.y_test2

    @classmethod
    def get_test_values(cls, kind=0):
        X_test1 = cls.X_test1.iloc[:, 10:]
        y_test1 = cls.y_test1
        X_test2 = cls.X_test2.iloc[:, 10:]
        y_test2 = cls.y_test2

        if kind == 0:
            return X_test1, y_test1, X_test2, y_test2
        elif kind == 1:
            return X_test1, y_test1
        elif kind == 2:
            return X_test2, y_test2

    @classmethod
    def get_train(cls, kind=1, method=None):
        X_train_res = None
        y_train_res = None
        sampling_method = method  

        X_train = cls.X_train
        y_train = cls.y_train
        X_train_values = cls.X_train.iloc[:, 10:]
        if kind not in [1, 2, 3, 0]:

            kind = 1
        if kind == 0 and method == None:
            kind = 1
        if method != None:
            kind = 0
        
        if kind == 1:
            X_train_res = X_train
            y_train_res = y_train
        else:
            if kind == 2:
                sampling_method = over_sampling.RandomOverSampler(
                    random_state=0)
            elif kind == 3:
                sampling_method = under_sampling.RandomUnderSampler(
                    random_state=0)
            elif kind == 0 and method == None:
                sampling_method = over_sampling.RandomOverSampler(
                    random_state=0)

            X_train_res, y_train_res = sampling_method.fit_sample(
                X_train, y_train)
        X_train_res = pd.DataFrame(X_train_res, columns=cls.X_train.columns)
        y_train_res = pd.Series(y_train_res)
        return X_train_res, y_train_res

    @classmethod
    def get_train_values(cls, kind):
        X_train_res, y_train_res = cls.get_train(kind)
        group = X_train_res.mut_residue
        X_train_res = X_train_res.iloc[:, 10:]
        return X_train_res, y_train_res, group

    @classmethod
    def split_cv(cls,
                 kind=1,
                 train_set_kind=1,
                 train_set_mehtod=None,
                 method=None,
                 n_cv=10,
                 return_kind=None
                ):
        if kind not in [0, 1, 2]:
            kind = 1
        if method != None:
            kind = 0
        # 获取数据
        X_train, y_train = cls.get_train(kind=train_set_kind,
                                                 method=train_set_mehtod)
        group = X_train.mut_residue

        if kind == 1:
          
            method = skf
        elif kind == 2:
            method = gss
        elif kind == 0:
            print("0")
        split_index = method.split(
            X_train, y_train, groups=group)  
        if return_kind==1:
            return split_index
        if return_kind==2:
            return method
            
        split_index = list(split_index)
        # [print(i[0].shape, i[1].shape) for i in split_index]
        cvs = [(X_train.iloc[i[1], :], y_train.iloc[i[1]])
               for i in split_index]

        tb = pd.DataFrame(
            columns=["variations", "decrease", "no-change", "increase"])
        for i, (X, y) in enumerate(cvs):
            tb.loc["train" + str(i + 1)] = [
                y.shape[0], *list(pd.value_counts(y).sort_index())
            ]

        print(tb)
        return cvs
    @classmethod
    def plot_split_cv(cls, train_set_kind, split_cv_kind, ax, n_splits=10, lw=10):
        X, y =  cls.get_train(kind=train_set_kind)
        group = X.mut_residue
        
        if split_cv_kind == 1:
            cv = skf
        elif split_cv_kind  == 2:
            cv = gss
        
        
        cmap_data = plt.cm.Paired
        cmap_cv = plt.cm.coolwarm
        # Generate the training/testing visualizations for each CV split
        for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
            # Fill in indices with the training/test groups
            indices = np.array([np.nan] * len(X))
            indices[tt] = 1
            indices[tr] = 0

            # Visualize the results
            ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                       c=indices,
                       marker='_',
                       lw=lw,
                       cmap=cmap_cv,
                       vmin=-.2,
                       vmax=1.2)

        # Plot the data classes and groups at the end
        ax.scatter(range(len(X)), [ii + 1.5] * len(X),
                   c=y,
                   marker='_',
                   lw=lw,
                   cmap=cmap_data)

        ax.scatter(range(len(X)), [ii + 2.5] * len(X),
                   c=group,
                   marker='_',
                   lw=lw,
                   cmap=cmap_data)

        # Formatting
        yticklabels = list(range(n_splits)) + ['class', 'group']
        ax.set(yticks=np.arange(n_splits + 2) + .5,
               yticklabels=yticklabels,
               xlabel='',
               ylabel="CV iteration",
               ylim=[n_splits + 2.2, -.2],
               xlim=[0, 100])
        ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
        return ax