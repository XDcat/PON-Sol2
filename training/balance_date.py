#!/usr/bin/env python
# coding: utf-8

# # 大蛋白质只以 YSD 为指标

# In[55]:


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
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)  # 10 层交叉验证（随机打乱）（训练/测试集数据分布与原数据分布一致）
gss = GroupShuffleSplit(n_splits=10, random_state=0)  # 10cv, 完全随机，按照group

with open("./config.json") as f:
    config = json.loads(f.read())
print("配置文件")
pprint(config)


# In[56]:


# 过采样以及欠采样
from imblearn import over_sampling
from imblearn import under_sampling
from collections import Counter


# # 读取数据

# In[57]:


# 读取文件
feature_path = config["all_feature"]  # 特征文件路径
feature = pd.read_csv(feature_path, index_col=0)

# 查看数据
feature.insert(loc=0, column="index", value=feature.index)  # 添加索引字段

# 对gi进行分类
new_gi = ['213133708', '190610225', '118137700', '1236048426', '40789249', '364505958'] # 新蛋白质的gi
large_gi = ['190610225', '213133708']  # 最大的两个蛋白质
new_not_large_gi = []
for i in new_gi:
    if i not in large_gi:
        new_not_large_gi.append(i)

old_gi = [i for i in feature.gi if i not in new_gi]  # 旧蛋白质的
old_gi_test = ['145563', '730948', '1684918', '2668592', '40796138', '67471959', '82581658', '116666949', '326936885']  # 旧的测试集
old_gi_train = []
for i in feature.gi.drop_duplicates().values:
    i = str(i)
    if (i not in large_gi) and (i not in old_gi_test):
        old_gi_train.append(i)
feature


# # 查看比例的函数

# In[58]:


def solubility_distribute(y, i=0,verb=0):
    _vc = pd.value_counts(y)
    _index = _vc.index.sort_values()
    _vc = _vc[_index]
    _str = ": ".join(map(str, _index)) + " = " + ": ".join(map(str, _vc.values)) + "= " + ": ".join(map(lambda x: "%.2f" % x, (_vc / _vc.iloc[i]).values))
    if (verb > 0):
        print(_str)
    return _str


# In[59]:


print("初始数据比例:", solubility_distribute(feature.solubility))
feature_of_lgk = feature[feature.gi == int(large_gi[0])]
feature_of_tem = feature[feature.gi == int(large_gi[1])]
feature_of_not_lgk_tem = feature[(feature.gi != int(large_gi[0])) & (feature.gi != int(large_gi[1]))]
print("lgk 分布:", solubility_distribute(feature_of_lgk.solubility))
print("tem 分布:", solubility_distribute(feature_of_tem.solubility))
print("其他 分布:", solubility_distribute(feature_of_not_lgk_tem.solubility))


# In[60]:


yeast_lgk = pd.read_csv(config['yeast_lgk'], index_col=0)
# pd.cut(yeast_lgk.Sol_Score, np.arange(-3, 3, 0.1), ).value_counts().sort_index().plot(kind="bar", figsize=(20, 10))
index_reduce_lgk = ((yeast_lgk.Sol_Score <= -1.0) | (yeast_lgk.Sol_Score >= -0.15))
reduce_yeast_lgk = yeast_lgk[index_reduce_lgk]
# print("lgk 筛除部分后:", solubility_distribute(reduce_yeast_lgk.Sol))


# In[61]:


yeast_tem = pd.read_csv(config['yeast_tem'], index_col=0)
# pd.cut(yeast_tem.Sol_Score, np.arange(-3, 3, 0.1), ).value_counts().sort_index().plot(kind="bar", figsize=(20, 10))
index_reduce_tem = ((yeast_tem.Sol_Score <= -1.0) | (yeast_tem.Sol_Score >= -0.15))
reduce_yeast_tem = yeast_tem[index_reduce_tem]
# print("tem 筛除部分后", solubility_distribute(reduce_yeast_tem.Sol))


# In[66]:


feature = pd.concat([
    feature_of_lgk.loc[index_reduce_lgk.values, :],
    feature_of_tem.loc[index_reduce_tem.values, :],
    feature_of_not_lgk_tem,
], axis=0)
feature = feature[feature.mut_from != feature.mut_to]  # 进行校验
print("经过两大蛋白质筛除后:", solubility_distribute(feature.solubility))


# # 划分训练（交叉验证）/测试集
# 测试集分为两种：
# 1. 测试集为以前的测试集
# 2. 测试集为以前的测试集加上新的数据，测试集的数目占10%; 然后使用过采样和欠采样，让比例为 -: 0: + = 2: 1: 1(其中基准为0)
# 
# 而训练集只有一种为：`所有数据 - 测试集2`
# 
# NOTE:
# 1. X 为数据 DataFrame
# 2. y 为分类的标签 list
# 3. index 为在元数据中的索引 list

# In[17]:


# 对选择出的特征进行分类
old_test_feature = feature[feature.gi.apply(lambda x: str(x) in old_gi_test)]  # 旧的测试集
feature_not_old_test = feature.loc[feature.index.difference(old_test_feature.index)]# 已选择的，非测试集

# 1. 测试集为以前的测试集
# 测试集
X_test1 = old_test_feature
y_test1 = old_test_feature.solubility
y_test1 = list(y_test1)
# 训练集(不包含以前的数据)
X_train1 = feature_not_old_test
y_train1 = feature_not_old_test.solubility


# In[18]:


# # 2. 测试集为以前的测试集加上新的数据，测试集的数目占10%, 且比例为 -1: 0: 1 = 1: 0.5: 0.5
# # 确定数目
# count_test2 = feature.solubility.value_counts() * 0.1
# count_test1 = pd.value_counts(y_test1)
# count_test2_new = count_test2 - count_test1
# count_test2_new = count_test2_new.astype(int)

# # 对每一类进行分割(数据源是 feature_not_old_test)
# X_train2s = []
# X_test2s = []

# _feature_1 = feature_not_old_test[feature_not_old_test.gi.apply(lambda x: str(x) in new_gi)]  # 在新的大蛋白质中寻找 test2
# _feature_2 = feature_not_old_test[feature_not_old_test.gi.apply(lambda x: str(x) not in new_gi)] # old train 仍然作为 train
# for i, v in count_test2_new.items():
#     feature_1_i = _feature_1[_feature_1.solubility == i]
#     X_i, y_i = feature_1_i, feature_1_i.solubility
#     X_train_i, X_test_i, y_train_i, y_test_i =  train_test_split(X_i, y_i, test_size=v, random_state=0)
#     X_train2s.append(X_train_i)
#     X_test2s.append(X_test_i)
    
# X_test2 = pd.concat(X_test2s + [X_test1, ], axis=0)
# y_test2 = list(X_test2.solubility)
# X_train2 = pd.concat(X_train2s + [_feature_2, ], axis=0)
# y_train2 = list(X_train2.solubility)


# In[19]:


# 2. 新的测试集包含旧的测试集，并且按照位点进行划分，将位点相同的数据同时放在训练集或者测试集
X_test2_1 = X_train1[X_train1.mut_residue.apply(lambda x: x in  set(X_test1.mut_residue.to_list()))]
X_test2 = pd.concat([X_test2_1, X_test1], axis=0)
y_test2 = X_test2.solubility
X_train2 = X_train1[X_train1.mut_residue.apply(lambda x: x not in  set(X_test1.mut_residue.to_list()))]
y_train2 = X_train2.solubility


# In[20]:


# 训练集
X_train = X_train2
y_train = y_train2
y_train = list(y_train)

print("数据总数目：", feature.shape[0],"\t", solubility_distribute(feature["solubility"]))
print("test1数目：", X_test1.shape[0], "\t\t", solubility_distribute(y_test1))
print("test2数目：", X_test2.shape[0], "\t", solubility_distribute(y_test2))
print("train数目：", X_train.shape[0], "\t", solubility_distribute(y_train))


# # 构建BalanceDate类，存储数据

# In[21]:


class BalanceDate():
    """
    将外部的数据，都分装到一个类中
    X - DateFrame
    y - Series
    FEATURE - Series
    """
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
        """ 获取测试集数据
        :param kind: 0-全部，1-test1，2-test2
        """
        if kind == 0:
            return cls.X_test1, cls.y_test1, cls.X_test2, cls.y_test2
        elif kind == 1:
            return cls.X_test1, cls.y_test1
        elif kind == 2:
            return cls.X_test2, cls.y_test2

    @classmethod
    def get_test_values(cls, kind=0):
        """ 获取测试集数据（只有特征的数据，没有溶解度等多余数据）
        :param kind: 0-全部，1-test1，2-test2
        """
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
        """
        根据不同的方式，平衡测试集。
        :param kind: 数据类型
            0: 自定义
            1: 比例 -1: 0: 1 = 2: 1: 1
            2: 过采样——增加少的
            3：欠采样——减少多的
        :param method: 自定义划分方法
        """
        X_train_res = None
        y_train_res = None
        sampling_method = method  # imblearn 中的对象
        # 引用数据
        X_train = cls.X_train
        y_train = cls.y_train
        X_train_values = cls.X_train.iloc[:, 10:]
        if kind not in [1, 2, 3, 0]:
            # 如果为空，默认为 1
            kind = 1
        if kind == 0 and method == None:
            kind = 1
        if method != None:
            kind = 0
        
        if kind == 1:
            """策略1—— 比例 -1: 0: 1 = 2: 1: 1
            """
            print("划分测试/训练集方法——1. 按照原始比例")
            X_train_res = X_train
            y_train_res = y_train
        else:
            # 使用 imblearn 的方法
            if kind == 2:
                """策略2——过采样
                通过增加 no effect 和 increase 的方式，使三种数据平衡。
                """
                print("划分测试/训练集方法——2. 过采样") 
                sampling_method = over_sampling.RandomOverSampler(
                    random_state=0)
            elif kind == 3:
                """ 策略3——下采样
                通过减少decrease的方式，使三种数据平衡
                """
                print("划分测试/训练集方法——3. 下采样") 
                sampling_method = under_sampling.RandomUnderSampler(
                    random_state=0)
            elif kind == 0 and method == None:
                """默认过采样"""
                sampling_method = over_sampling.RandomOverSampler(
                    random_state=0)

            X_train_res, y_train_res = sampling_method.fit_sample(
                X_train, y_train)
        # 格式化数据
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
        """
        划分n-cv
        :param kind: 类型
            0: 自定义方法
            1: 默认 随机取，每种（+ - =）取原比例的
            2: 利用group，将位点相同的变异放在起
        :param train_set_kind: 数据类型
            0: 自定义
            1: 比例 -1: 0: 1 = 2: 1: 1
            2: 过采样——增加少的
            3：欠采样——减少多的
        :param method: 其他的划分方法
        :param n_cv: cv数目
        :param return_kind: 返回的类型
            默认返回每一个验证集
            1. 返回迭代器
            2. 直接返回使用的方法
        """
        # 容错：初始化 kind
        if kind not in [0, 1, 2]:
            kind = 1
        if method != None:
            kind = 0
        # 获取数据
        X_train, y_train = cls.get_train(kind=train_set_kind,
                                                 method=train_set_mehtod)
        group = X_train.mut_residue

        print("划分{}cv".format(n_cv))
        print("训练集数目:", X_train.shape)
        print("划分cv方法——", end=" ")
        if kind == 1:
            print("1: 默认 随机取，每种（+ - =）取原比例的")
            method = skf
        elif kind == 2:
            print("2: 利用group，将位点相同的变异放在起")
            method = gss
        elif kind == 0:
            print("0: 自定义方法")
        split_index = method.split(
            X_train, y_train, groups=group)  # 划分10cv，返回10个（验证集index，其他index）
        if return_kind==1:
            print("返回可迭代对象")
            return split_index
        if return_kind==2:
            print("返回使用的cv方法对象")
            return method
            
        split_index = list(split_index)
        # [print(i[0].shape, i[1].shape) for i in split_index]
        cvs = [(X_train.iloc[i[1], :], y_train.iloc[i[1]])
               for i in split_index]
        # 绘制表格
        tb = pd.DataFrame(
            columns=["variations", "decrease", "no-change", "increase"])
        for i, (X, y) in enumerate(cvs):
            tb.loc["train" + str(i + 1)] = [
                y.shape[0], *list(pd.value_counts(y).sort_index())
            ]
        print("返回所有验证集")
        print(tb)
        return cvs
    @classmethod
    def plot_split_cv(cls, train_set_kind, split_cv_kind, ax, n_splits=10, lw=10):
        """
        可视化cv的数据

        """
        # 获取数据
        X, y =  cls.get_train(kind=train_set_kind)
        group = X.mut_residue
        
        print("划分cv方法——", end=" ")
        if split_cv_kind == 1:
            print("1: 默认 随机取，每种（+ - =）取原比例的")
            cv = skf
        elif split_cv_kind  == 2:
            print("2: 利用group，将位点相同的变异放在起")
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