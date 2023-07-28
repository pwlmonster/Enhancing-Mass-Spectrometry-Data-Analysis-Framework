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
from Classfication import Classfication
from Classfication_transformer import Classfication_transformer


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

# Classfication的网络参数
# CHD二分类
hyp_Classfication = {
    'input_dim': 199,
    'hidden_dim': 512,
    'output_dim': 2,
    'drop_out': 0.5,
    'batch_size': 128,
    'lr': 0.0001,
    'epochs': 300,
}
# # MI二分类
# hyp_Classfication = {
#     'input_dim': 204,
#     'hidden_dim': 512,
#     'output_dim': 2,
#     'drop_out': 0.5,
#     'batch_size': 128,
#     'lr': 0.0001,
#     'epochs': 300,
# }
# Transformer
hyp_Classfication_transformer = {
    'num_layers': 2,
    'attention_heads': 8,
    'dropout': 0.5,
    'batch_size': 128,
    'lr': 0.0001,
    'epochs': 200,
}


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

    # # AutoEncoder模型筛选离群点
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
        # 分割数据与标签
        TrainLabel, TrainData = np.split(traindataset_new, (1,), axis=1)
        # 转换为张量
        TrainData = torch.FloatTensor(TrainData)
        TrainLabel = torch.IntTensor(TrainLabel)

        del traindataset_new
        gc.collect()

        testdataset_new = pd.DataFrame(lab_Dataset_test)
        testdataset_new = testdataset_new.iloc[:, 2:]
        testdataset_new = np.array(testdataset_new)
        testdataset_new = testdataset_new.astype(np.float)
        # 分割数据与标签
        TestLabel, TestData = np.split(testdataset_new, (1,), axis=1)
        # 转换为张量
        TestData = torch.FloatTensor(TestData)
        TestLabel = torch.IntTensor(TestLabel)

        del testdataset_new
        gc.collect()

        Classfication(hyp_Classfication, TrainData, TrainLabel, TestData, TestLabel)

        del TrainData, TrainLabel, TestData, TestLabel
        gc.collect()

    # 投票出现了三次的离群点
    count_dict_3_4 = {}
    for item in set(outlier_all):
        count_dict_3_4[item] = outlier_all.count(item)
    result_ensemble_3_4 = [k for k, v in count_dict_3_4.items() if v >= 3]
    # # 投票出现了四次的离群点
    # count_dict_4_4 = {}
    # for item in set(outlier_all):
    #     count_dict_4_4[item] = outlier_all.count(item)
    # result_ensemble_4_4 = [k for k, v in count_dict_4_4.items() if v >= 4]

    # 记录投票产生的最终离群点
    f = open("./resultLog.txt", "a")
    print('**********ensemble 3/4：', result_ensemble_3_4, file=f)
    f.close()

    # f = open("./resultLog.txt", "a")
    # print('**********ensemble 4/4：', result_ensemble_4_4, file=f)
    # f.close()

    # 构建用于分类任务的训练集和测试集
    traindataset_new = pd.DataFrame(lab_Dataset_train)

    # 在训练集中去除3/4ensemble策略离群点
    traindataset_new_3_4 = traindataset_new[~traindataset_new[0].isin(result_ensemble_3_4)]
    traindataset_new_3_4 = traindataset_new_3_4.iloc[:, 2:]
    traindataset_new_3_4 = np.array(traindataset_new_3_4)
    traindataset_new_3_4 = traindataset_new_3_4.astype(np.float)
    # 分割数据与标签
    TrainLabel_3_4, TrainData_3_4 = np.split(traindataset_new_3_4, (1,), axis=1)
    # 转换为张量
    TrainData_3_4 = torch.FloatTensor(TrainData_3_4)
    TrainLabel_3_4 = torch.IntTensor(TrainLabel_3_4)

    # # 在训练集中去除4/4ensemble策略离群点
    # traindataset_new_4_4 = traindataset_new[~traindataset_new[0].isin(result_ensemble_4_4)]
    # traindataset_new_4_4 = traindataset_new_4_4.iloc[:, 2:]
    # traindataset_new_4_4 = np.array(traindataset_new_4_4)
    # traindataset_new_4_4 = traindataset_new_4_4.astype(np.float)
    # # 分割数据与标签
    # TrainLabel_4_4, TrainData_4_4 = np.split(traindataset_new_4_4, (1,), axis=1)
    # # 转换为张量
    # TrainData_4_4 = torch.FloatTensor(TrainData_4_4)
    # TrainLabel_4_4 = torch.IntTensor(TrainLabel_4_4)

    # del traindataset_new, traindataset_new_3_4, traindataset_new_4_4
    del traindataset_new, traindataset_new_3_4
    gc.collect()

    testdataset_new = pd.DataFrame(lab_Dataset_test)
    testdataset_new = testdataset_new.iloc[:, 2:]
    testdataset_new = np.array(testdataset_new)
    testdataset_new = testdataset_new.astype(np.float)
    # 分割数据与标签
    TestLabel, TestData = np.split(testdataset_new, (1,), axis=1)
    # 转换为张量
    TestData = torch.FloatTensor(TestData)
    TestLabel = torch.IntTensor(TestLabel)

    del testdataset_new
    gc.collect()

    Classfication(hyp_Classfication, TrainData_3_4, TrainLabel_3_4, TestData, TestLabel)
    # Classfication(hyp_Classfication, TrainData_4_4, TrainLabel_4_4, TestData, TestLabel)
    Classfication_transformer(hyp_Classfication_transformer, TrainData_3_4, TrainLabel_3_4, TestData, TestLabel)

    # del TrainData_3_4, TrainLabel_3_4, TrainData_4_4, TrainLabel_4_4, TestData, TestLabel
    del TrainData_3_4, TrainLabel_3_4, TestData, TestLabel
    gc.collect()

    del lab_Dataset_train, lab_Dataset_test
    gc.collect()


def batch_off(TrainData_DL, TestData_DL):

    # AutoEncoder模型训练
    Batch_AE_model_path = Batch_AE_model_train(hyp_AE, TrainData_DL)
    # 得到去除批次效应的数据集
    Batch_lab_Dataset_train, Batch_lab_Dataset_test = Batch_AE_model_test(hyp_AE, Batch_AE_model_path, TrainData_DL, TestData_DL)
    Batch_TrainData_CL, Batch_TrainLabel_CL, Batch_TestData_CL, Batch_TestLabel_CL = DataLoader(Batch_lab_Dataset_train, Batch_lab_Dataset_test, 'CL')

    # 得到去除批次效应后的分类效果
    f = open("resultLog.txt", "a")
    print('********************去除批次效应后的分类效果', file=f)
    f.close()
    Classfication(hyp_Classfication, Batch_TrainData_CL, Batch_TrainLabel_CL, Batch_TestData_CL, Batch_TestLabel_CL)

    del TrainData_DL, TestData_DL, Batch_TrainData_CL, Batch_TrainLabel_CL, Batch_TestData_CL, Batch_TestLabel_CL
    gc.collect()

    return Batch_lab_Dataset_train, Batch_lab_Dataset_test


# 运行不同的分类任务
for task in task_list:

    f = open("resultLog.txt", "a")
    print('********************数据集任务：', task, file=f)
    f.close()

    lab_Dataset_train, lab_Dataset_test = Task_Dataset(task)
    TrainData_CL, TrainLabel_CL, TestData_CL, TestLabel_CL = DataLoader(lab_Dataset_train, lab_Dataset_test, 'CL')
    TrainData_ML, TestData_ML = DataLoader(lab_Dataset_train, lab_Dataset_test, 'ML')
    TrainData_DL, TestData_DL = DataLoader(lab_Dataset_train, lab_Dataset_test, 'DL')

    # 得到分类任务的基线
    f = open("resultLog.txt", "a")
    print('********************分类基线', file=f)
    f.close()
    Classfication(hyp_Classfication, TrainData_CL, TrainLabel_CL, TestData_CL, TestLabel_CL)

    del TrainData_CL, TrainLabel_CL, TestData_CL, TestLabel_CL
    gc.collect()

    # 去除数据的批次效应
    f = open("resultLog.txt", "a")
    print('********************去除数据间批次效应', file=f)
    f.close()

    Batch_lab_Dataset_train, Batch_lab_Dataset_test = batch_off(TrainData_DL, TestData_DL)
    Batch_TrainData_ML, Batch_TestData_ML = DataLoader(Batch_lab_Dataset_train, Batch_lab_Dataset_test, 'ML')
    Batch_TrainData_DL, Batch_TestData_DL = DataLoader(Batch_lab_Dataset_train, Batch_lab_Dataset_test, 'DL')

    # # 根据不同比例筛选离群点
    # for outlier_ratio in outlier_ratio_list:
    #     f = open("resultLog.txt", "a")
    #     print('********************筛选离群点比例：', outlier_ratio, file=f)
    #     f.close()
    #     select_outlier(TrainData_ML, TestData_ML, TrainData_DL, TestData_DL, lab_Dataset_train, lab_Dataset_test, outlier_ratio)

    # 根据不同比例筛选去除批次效应后的离群点
    for outlier_ratio in outlier_ratio_list:
        f = open("resultLog.txt", "a")
        print('********************筛选去除批次效应后离群点比例：', outlier_ratio, file=f)
        f.close()
        select_outlier(Batch_TrainData_ML, Batch_TestData_ML, Batch_TrainData_DL, Batch_TestData_DL, Batch_lab_Dataset_train, Batch_lab_Dataset_test, outlier_ratio)

    del TrainData_ML, TestData_ML, TrainData_DL, TestData_DL
    gc.collect()

    del Batch_TrainData_ML, Batch_TestData_ML, Batch_TrainData_DL, Batch_TestData_DL
    gc.collect()

    del lab_Dataset_train, lab_Dataset_test, Batch_lab_Dataset_train, Batch_lab_Dataset_test
    gc.collect()