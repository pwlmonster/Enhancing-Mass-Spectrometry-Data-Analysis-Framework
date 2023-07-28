import gc

import numpy as np
import pandas as pd
import torch

from Dataloading import Task_Dataset, DataLoader
from OutlierML import LOF, iForest, Kmeans
from AE_train import AE_model_train
from AE_test import AE_model_test
from Batch_AE_train import Batch_AE_model_train
from Batch_AE_test import Batch_AE_model_test
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# 所需要执行的任务列表
task_list = ['Task1', 'Task2', 'Task3', 'Task4']

# 所需要筛选的离群点比例
outlier_ratio_list = [0.01, 0.04]

# AutoEncoder的网络参数
# CHD数据集
hyp_AE = {
    'input_dim': 199,
    'hidden_dim': 128,
    'batch_size': 64,
    'lr': 1e-4,
    'epochs': 200,
}
# # MI数据集
# hyp_AE = {
#     'input_dim': 204,
#     'hidden_dim': 128,
#     'batch_size': 64,
#     'lr': 1e-4,
#     'epochs': 200,
# }


# 离群点筛选函数
def select_outlier(TrainData_ML, TestData_ML, TrainData_DL, TestData_DL, lab_Dataset_train, lab_Dataset_test, outlier_ratio):

    all_list = []

    # LOF算法
    outlier_LOF = LOF(TrainData_ML, TestData_ML, outlier_ratio)
    # iForest算法
    outlier_iForest = iForest(TrainData_ML, TestData_ML, outlier_ratio)
    # Kmeans算法
    outlier_Kmeans = Kmeans(TrainData_ML, TestData_ML, outlier_ratio)
    # AutoEncoder模型混合筛离群点
    AE_model_path_mix = AE_model_train(hyp_AE, TrainData_DL)
    outlier_AE_mix = AE_model_test(hyp_AE, AE_model_path_mix, TrainData_DL, TestData_DL, outlier_ratio)

    # # AutoEncoder模型分类别筛选离群点
    # TrainData_DL_control = TrainData_DL[TrainData_DL[:, 2] == '0']
    # TrainData_DL_CHD = TrainData_DL[TrainData_DL[:, 2] == '1']
    # # AutoEncoder模型训练
    # AE_model_path_control = AE_model_train(hyp_AE, TrainData_DL_control)
    # outlier_AE_control = AE_model_test(hyp_AE, AE_model_path_control, TrainData_DL_control, TestData_DL, outlier_ratio)
    #
    # AE_model_path_CHD = AE_model_train(hyp_AE, TrainData_DL_CHD)
    # outlier_AE_CHD = AE_model_test(hyp_AE, AE_model_path_CHD, TrainData_DL_CHD, TestData_DL, outlier_ratio)
    #
    # outlier_AE = outlier_AE_control + outlier_AE_CHD

    outlier_all = outlier_LOF + outlier_iForest + outlier_Kmeans + outlier_AE_mix

    del TrainData_ML, TestData_ML, TrainData_DL, TestData_DL
    gc.collect()

    all_list.append(outlier_LOF)
    all_list.append(outlier_iForest)
    all_list.append(outlier_Kmeans)
    all_list.append(outlier_AE_mix)

    for list in all_list:
        # 构建用于分类任务的训练集和测试集
        traindataset_new = pd.DataFrame(lab_Dataset_train)
        traindataset_new = traindataset_new[~traindataset_new[0].isin(list)]
        traindataset_new = traindataset_new.iloc[:, 2:]
        traindataset_new = np.array(traindataset_new)
        traindataset_new = traindataset_new.astype(np.float)

        testdataset_new = pd.DataFrame(lab_Dataset_test)
        testdataset_new = testdataset_new.iloc[:, 2:]
        testdataset_new = np.array(testdataset_new)
        testdataset_new = testdataset_new.astype(np.float)

        # 分割数据与标签
        Dataset_train_label, Dataset_train_data = np.split(traindataset_new, (1,), axis=1)
        Dataset_test_label, Dataset_test_data = np.split(testdataset_new, (1,), axis=1)

        del traindataset_new, testdataset_new
        gc.collect()

        # # 支持向量机线性核函数模型
        # clf = svm.SVC(kernel='linear', C=0.8, coef0=0.0, gamma='auto', degree=3, decision_function_shape='ovo', max_iter=5000)
        # # 随机森林模型
        # clf = RandomForestClassifier(random_state=2023)
        # 逻辑回归模型
        clf = LogisticRegression()

        clf.fit(Dataset_train_data, Dataset_train_label)

        f = open("./resultLog.txt", "a")
        print('测试集准确率：', clf.score(Dataset_test_data, Dataset_test_label), file=f)
        f.close()

        del Dataset_train_data, Dataset_train_label, Dataset_test_data, Dataset_test_label
        gc.collect()

    # 投票出现了三次的离群点
    count_dict_3_4 = {}
    for item in set(outlier_all):
        count_dict_3_4[item] = outlier_all.count(item)
    result_ensemble_3_4 = [k for k, v in count_dict_3_4.items() if v >= 3]
    # 投票出现了四次的离群点
    count_dict_4_4 = {}
    for item in set(outlier_all):
        count_dict_4_4[item] = outlier_all.count(item)
    result_ensemble_4_4 = [k for k, v in count_dict_4_4.items() if v >= 4]

    # 记录投票产生的最终离群点
    f = open("./resultLog.txt", "a")
    print('**********ensemble 3/4：', result_ensemble_3_4, file=f)
    f.close()

    f = open("./resultLog.txt", "a")
    print('**********ensemble 4/4：', result_ensemble_4_4, file=f)
    f.close()

    # 构建用于分类任务的训练集和测试集
    traindataset_new = pd.DataFrame(lab_Dataset_train)

    # 在训练集中去除3/4ensemble策略离群点
    traindataset_new_3_4 = traindataset_new[~traindataset_new[0].isin(result_ensemble_3_4)]
    traindataset_new_3_4 = traindataset_new_3_4.iloc[:, 2:]
    traindataset_new_3_4 = np.array(traindataset_new_3_4)
    traindataset_new_3_4 = traindataset_new_3_4.astype(np.float)
    # 分割数据与标签
    Dataset_train_label_3_4, Dataset_train_data_3_4 = np.split(traindataset_new_3_4, (1,), axis=1)

    # 在训练集中去除4/4ensemble策略离群点
    traindataset_new_4_4 = traindataset_new[~traindataset_new[0].isin(result_ensemble_4_4)]
    traindataset_new_4_4 = traindataset_new_4_4.iloc[:, 2:]
    traindataset_new_4_4 = np.array(traindataset_new_4_4)
    traindataset_new_4_4 = traindataset_new_4_4.astype(np.float)
    # 分割数据与标签
    Dataset_train_label_4_4, Dataset_train_data_4_4 = np.split(traindataset_new_4_4, (1,), axis=1)

    del traindataset_new, traindataset_new_3_4, traindataset_new_4_4
    gc.collect()

    testdataset_new = pd.DataFrame(lab_Dataset_test)
    testdataset_new = testdataset_new.iloc[:, 2:]
    testdataset_new = np.array(testdataset_new)
    testdataset_new = testdataset_new.astype(np.float)
    # 分割数据与标签
    Dataset_test_label, Dataset_test_data = np.split(testdataset_new, (1,), axis=1)

    del testdataset_new
    gc.collect()

    # # 支持向量机线性核函数模型
    # clf = svm.SVC(kernel='linear', C=0.8, coef0=0.0, gamma='auto', degree=3, decision_function_shape='ovo', max_iter=5000)
    # # 随机森林模型
    # clf = RandomForestClassifier(random_state=2023)
    # 逻辑回归模型
    clf = LogisticRegression()

    clf.fit(Dataset_train_data_3_4, Dataset_train_label_3_4)

    f = open("./resultLog.txt", "a")
    print('测试集准确率：', clf.score(Dataset_test_data, Dataset_test_label), file=f)
    f.close()

    # # 支持向量机线性核函数模型
    # clf = svm.SVC(kernel='linear', C=0.8, coef0=0.0, gamma='auto', degree=3, decision_function_shape='ovo', max_iter=5000)
    # # 随机森林模型
    # clf = RandomForestClassifier(random_state=2023)
    # 逻辑回归模型
    clf = LogisticRegression()

    clf.fit(Dataset_train_data_4_4, Dataset_train_label_4_4)

    f = open("./resultLog.txt", "a")
    print('测试集准确率：', clf.score(Dataset_test_data, Dataset_test_label), file=f)
    f.close()

    del Dataset_train_data_3_4, Dataset_train_label_3_4, Dataset_train_data_4_4, Dataset_train_label_4_4, Dataset_test_data, Dataset_test_label
    gc.collect()

    del lab_Dataset_train, lab_Dataset_test


def batch_off(TrainData_DL, TestData_DL):

    # AutoEncoder模型训练
    Batch_AE_model_path = Batch_AE_model_train(hyp_AE, TrainData_DL)
    # 得到去除批次效应的数据集
    Batch_lab_Dataset_train, Batch_lab_Dataset_test = Batch_AE_model_test(hyp_AE, Batch_AE_model_path, TrainData_DL, TestData_DL)

    Batch_lab_Dataset_train_info, Batch_lab_Dataset_train_use = np.split(Batch_lab_Dataset_train, (2,), axis=1)
    Batch_lab_Dataset_train_label, Batch_lab_Dataset_train_data = np.split(Batch_lab_Dataset_train_use, (1,), axis=1)
    Batch_lab_Dataset_test_info, Batch_lab_Dataset_test_use = np.split(Batch_lab_Dataset_test, (2,), axis=1)
    Batch_lab_Dataset_test_label, Batch_lab_Dataset_test_data = np.split(Batch_lab_Dataset_test_use, (1,), axis=1)

    del Batch_lab_Dataset_train_info, Batch_lab_Dataset_train_use, Batch_lab_Dataset_test_info, Batch_lab_Dataset_test_use
    gc.collect()

    # # 支持向量机线性核函数模型
    # clf = svm.SVC(kernel='linear', C=0.8, coef0=0.0, gamma='auto', degree=3, decision_function_shape='ovo', max_iter=5000)
    # # 随机森林模型
    # clf = RandomForestClassifier(random_state=2023)
    # 逻辑回归模型
    clf = LogisticRegression()

    Batch_lab_Dataset_train_data = Batch_lab_Dataset_train_data.astype(np.float)
    Batch_lab_Dataset_train_label = Batch_lab_Dataset_train_label.astype(np.float)
    Batch_lab_Dataset_test_data = Batch_lab_Dataset_test_data.astype(np.float)
    Batch_lab_Dataset_test_label = Batch_lab_Dataset_test_label.astype(np.float)

    # 得到去除批次效应后的分类效果
    f = open("./resultLog.txt", "a")
    print('********************去除批次效应后的分类效果', file=f)
    f.close()

    clf.fit(Batch_lab_Dataset_train_data, Batch_lab_Dataset_train_label)

    f = open("./resultLog.txt", "a")
    print('测试集准确率：', clf.score(Batch_lab_Dataset_test_data, Batch_lab_Dataset_test_label), file=f)
    f.close()

    del Batch_lab_Dataset_train_data, Batch_lab_Dataset_train_label, Batch_lab_Dataset_test_data, Batch_lab_Dataset_test_label
    gc.collect()

    return Batch_lab_Dataset_train, Batch_lab_Dataset_test


# 运行不同的分类任务
for task in task_list:

    f = open("./resultLog.txt", "a")
    print('********************数据集任务：', task, file=f)
    f.close()

    lab_Dataset_train, lab_Dataset_test = Task_Dataset(task)
    lab_Dataset_train_info, lab_Dataset_train_use = np.split(lab_Dataset_train, (2,), axis=1)
    lab_Dataset_train_label, lab_Dataset_train_data = np.split(lab_Dataset_train_use, (1,), axis=1)
    lab_Dataset_test_info, lab_Dataset_test_use = np.split(lab_Dataset_test, (2,), axis=1)
    lab_Dataset_test_label, lab_Dataset_test_data = np.split(lab_Dataset_test_use, (1,), axis=1)

    TrainData_ML, TestData_ML = DataLoader(lab_Dataset_train, lab_Dataset_test, 'ML')
    TrainData_DL, TestData_DL = DataLoader(lab_Dataset_train, lab_Dataset_test, 'DL')

    del lab_Dataset_train_info, lab_Dataset_test_info, lab_Dataset_train_use, lab_Dataset_test_use
    gc.collect()

    # # 支持向量机线性核函数模型
    # clf = svm.SVC(kernel='linear', C=0.8, coef0=0.0, gamma='auto', degree=3, decision_function_shape='ovo', max_iter=5000)
    # # 随机森林模型
    # clf = RandomForestClassifier(random_state=2023)
    # 逻辑回归模型
    clf = LogisticRegression()

    lab_Dataset_train_data = lab_Dataset_train_data.astype(np.float)
    lab_Dataset_train_label = lab_Dataset_train_label.astype(np.float)
    lab_Dataset_test_data = lab_Dataset_test_data.astype(np.float)
    lab_Dataset_test_label = lab_Dataset_test_label.astype(np.float)

    # 得到分类任务的基线
    f = open("./resultLog.txt", "a")
    print('********************分类基线', file=f)
    f.close()

    clf.fit(lab_Dataset_train_data, lab_Dataset_train_label)

    f = open("./resultLog.txt", "a")
    print('测试集准确率：', clf.score(lab_Dataset_test_data, lab_Dataset_test_label), file=f)
    f.close()

    del lab_Dataset_train_data, lab_Dataset_train_label, lab_Dataset_test_data, lab_Dataset_test_label
    gc.collect()

    # 去除数据的批次效应
    f = open("./resultLog.txt", "a")
    print('********************去除数据间批次效应', file=f)
    f.close()

    Batch_lab_Dataset_train, Batch_lab_Dataset_test = batch_off(TrainData_DL, TestData_DL)
    Batch_TrainData_ML, Batch_TestData_ML = DataLoader(Batch_lab_Dataset_train, Batch_lab_Dataset_test, 'ML')
    Batch_TrainData_DL, Batch_TestData_DL = DataLoader(Batch_lab_Dataset_train, Batch_lab_Dataset_test, 'DL')

    # # 根据不同比例筛选离群点
    # for outlier_ratio in outlier_ratio_list:
    #     f = open("./resultLog.txt", "a")
    #     print('********************筛选离群点比例：', outlier_ratio, file=f)
    #     f.close()
    #     select_outlier(TrainData_ML, TestData_ML, TrainData_DL, TestData_DL, lab_Dataset_train, lab_Dataset_test, outlier_ratio)

    # 根据不同比例筛选去除批次效应后的离群点
    for outlier_ratio in outlier_ratio_list:
        f = open("./resultLog.txt", "a")
        print('********************筛选去除批次效应后离群点比例：', outlier_ratio, file=f)
        f.close()
        select_outlier(Batch_TrainData_ML, Batch_TestData_ML, Batch_TrainData_DL, Batch_TestData_DL, Batch_lab_Dataset_train, Batch_lab_Dataset_test, outlier_ratio)

    del TrainData_ML, TestData_ML, TrainData_DL, TestData_DL
    gc.collect()

    del Batch_TrainData_ML, Batch_TestData_ML, Batch_TrainData_DL, Batch_TestData_DL
    gc.collect()

    del lab_Dataset_train, lab_Dataset_test, Batch_lab_Dataset_train, Batch_lab_Dataset_test
    gc.collect()