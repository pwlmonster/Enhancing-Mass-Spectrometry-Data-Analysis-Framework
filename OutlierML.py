import numpy as np
import pandas as pd
import torch
import math

from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans


def stand(x):
    x = (x-x.mean(axis=0))/x.std(axis=0)
    return x


# 基于密度的LOF算法
def LOF(traindataset, testdataset, outlier_ratio):

    final_ratio = math.floor(outlier_ratio * traindataset.shape[0])

    rest_pd = traindataset.iloc[:, :3]
    data_pd = traindataset.iloc[:, 3:]
    # data_pd = stand(data_pd)
    # data_pd = data_pd.fillna(0)

    X = data_pd.values
    lof = LocalOutlierFactor(n_neighbors=1000)
    lof.fit_predict(X)
    outfactor = lof.negative_outlier_factor_
    radius = (outfactor.max() - outfactor) / (outfactor.max() - outfactor.min())
    data_pd["score"] = radius
    data_result = pd.concat([rest_pd, data_pd], axis=1)
    outlier_result = data_result.sort_values("score", ascending=False).head(final_ratio)
    # outlier_score = data_result.sort_values("score", ascending=False).head(50)

    mask = ~data_result.isin(outlier_result)
    data_new = data_result[mask]
    data_new = data_new.dropna(how='all')

    # 输出离群点列表
    outlier_name_list = outlier_result.iloc[:, 0].tolist()

    f = open("./resultLog.txt", "a")
    print('LOF, %.2f, outlier:' % (outlier_ratio), outlier_name_list, file=f)
    f.close()

    # f = open("./resultLog.txt", "a")
    # print(outlier_score, file=f)
    # f.close()

    return outlier_name_list

#     # 生成去除离群点后的数据集
#     # 去除无关信息
#     traindataset_new = data_new.iloc[:, 2:]
#     traindataset_new = traindataset_new.iloc[:, :-1]
#     testdataset_new = testdataset.iloc[:, 2:]
#     # 转换数据类型
#     traindataset_new = np.array(traindataset_new)
#     testdataset_new = np.array(testdataset_new)
#     traindataset_new = traindataset_new.astype(np.float)
#     testdataset_new = testdataset_new.astype(np.float)
#     # 分割数据与标签
#     TrainLabel, TrainData = np.split(traindataset_new, (1,), axis=1)
#     TestLabel, TestData = np.split(testdataset_new, (1,), axis=1)
#     # 转换为张量
#     TrainData = torch.FloatTensor(TrainData)
#     TestData = torch.FloatTensor(TestData)
#     TrainLabel = torch.IntTensor(TrainLabel)
#     TestLabel = torch.IntTensor(TestLabel)

#     return TrainData, TrainLabel, TestData, TestLabel


# 孤立森林算法
def iForest(traindataset, testdataset, outlier_ratio):

    final_ratio = math.floor(outlier_ratio * traindataset.shape[0])

    rest_pd = traindataset.iloc[:, :3]
    data_pd = traindataset.iloc[:, 3:]
    # data_pd = stand(data_pd)
    # data_pd = data_pd.fillna(0)

# 定义孤立森林模型
    model = IsolationForest(n_estimators=300,
                            max_samples=2000)

    model.fit(data_pd)
    # 预测 decision_function 可以得出异常评分
    data_pd['scores'] = model.decision_function(data_pd)
    data_result = pd.concat([rest_pd, data_pd], axis=1)
    outlier_result = data_result.sort_values("scores", ascending=True).head(final_ratio)
    # outlier_score = data_result.sort_values("scores", ascending=False).head(50)

    mask = ~data_result.isin(outlier_result)
    data_new = data_result[mask]
    data_new = data_new.dropna(how='all')

    # 输出离群点列表
    outlier_name_list = outlier_result.iloc[:, 0].tolist()

    f = open("./resultLog.txt", "a")
    print('iForest, %.2f, outlier:' % (outlier_ratio), outlier_name_list, file=f)
    f.close()

    # f = open("./resultLog.txt", "a")
    # print(outlier_score, file=f)
    # f.close()

    return outlier_name_list

#     # 生成去除离群点后的数据集
#     # 去除无关信息
#     traindataset_new = data_new.iloc[:, 2:]
#     traindataset_new = traindataset_new.iloc[:, :-1]
#     testdataset_new = testdataset.iloc[:, 2:]
#     # 转换数据类型
#     traindataset_new = np.array(traindataset_new)
#     testdataset_new = np.array(testdataset_new)
#     traindataset_new = traindataset_new.astype(np.float)
#     testdataset_new = testdataset_new.astype(np.float)
#     # 分割数据与标签
#     TrainLabel, TrainData = np.split(traindataset_new, (1,), axis=1)
#     TestLabel, TestData = np.split(testdataset_new, (1,), axis=1)
#     # 转换为张量
#     TrainData = torch.FloatTensor(TrainData)
#     TestData = torch.FloatTensor(TestData)
#     TrainLabel = torch.IntTensor(TrainLabel)
#     TestLabel = torch.IntTensor(TestLabel)

#     return TrainData, TrainLabel, TestData, TestLabel


# Kmeans聚类算法
def Kmeans(traindataset, testdataset, outlier_ratio):

    final_ratio = math.floor(outlier_ratio * traindataset.shape[0])

    rest_pd = traindataset.iloc[:, :3]
    data_pd = traindataset.iloc[:, 3:]
    # data_pd = stand(data_pd)
    # data_pd = data_pd.fillna(0)

    # 定义Kmeans模型
    model = KMeans(n_clusters=2, max_iter=1000, random_state=2023)
    # 模型训练
    model.fit(data_pd)
    # 计算每个数据点到所在集群质心的距离
    distances = np.min(model.transform(data_pd), axis=1)
    data_pd['distances'] = pd.DataFrame(distances)
    data_result = pd.concat([rest_pd, data_pd], axis=1)
    outlier_result = data_result.sort_values("distances", ascending=False).head(final_ratio)
    # outlier_score = data_result.sort_values("distances", ascending=False).head(50)

    mask = ~data_result.isin(outlier_result)
    data_new = data_result[mask]
    data_new = data_new.dropna(how='all')

    # 输出离群点列表
    outlier_name_list = outlier_result.iloc[:, 0].tolist()

    f = open("./resultLog.txt", "a")
    print('Kmeans, %.2f, outlier:' % (outlier_ratio), outlier_name_list, file=f)
    f.close()

    # f = open("./resultLog.txt", "a")
    # print(outlier_score, file=f)
    # f.close()

    return outlier_name_list

#     # 生成去除离群点后的数据集
#     # 去除无关信息
#     traindataset_new = data_new.iloc[:, 2:]
#     traindataset_new = traindataset_new.iloc[:, :-1]
#     testdataset_new = testdataset.iloc[:, 2:]
#     # 转换数据类型
#     traindataset_new = np.array(traindataset_new)
#     testdataset_new = np.array(testdataset_new)
#     traindataset_new = traindataset_new.astype(np.float)
#     testdataset_new = testdataset_new.astype(np.float)
#     # 分割数据与标签
#     TrainLabel, TrainData = np.split(traindataset_new, (1,), axis=1)
#     TestLabel, TestData = np.split(testdataset_new, (1,), axis=1)
#     # 转换为张量
#     TrainData = torch.FloatTensor(TrainData)
#     TestData = torch.FloatTensor(TestData)
#     TrainLabel = torch.IntTensor(TrainLabel)
#     TestLabel = torch.IntTensor(TestLabel)

#     return TrainData, TrainLabel, TestData, TestLabel