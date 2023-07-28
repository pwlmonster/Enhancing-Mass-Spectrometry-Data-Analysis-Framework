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


# 所需要执行的任务列表
task_list = ['Task20']

# 所需要筛选的离群点比例
outlier_ratio_list = [0.04]

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
def select_outlier(TrainData_ML, TestData_ML, TrainData_DL, TestData_DL, outlier_ratio):

    # LOF算法
    outlier_LOF = LOF(TrainData_ML, TestData_ML, outlier_ratio)
    # iForest算法
    outlier_iForest = iForest(TrainData_ML, TestData_ML, outlier_ratio)
    # Kmeans算法
    outlier_Kmeans = Kmeans(TrainData_ML, TestData_ML, outlier_ratio)
    # AutoEncoder模型混合筛离群点
    AE_model_path_mix = AE_model_train(hyp_AE, TrainData_DL)
    outlier_AE_mix = AE_model_test(hyp_AE, AE_model_path_mix, TrainData_DL, TestData_DL, outlier_ratio)

    outlier_all = outlier_LOF + outlier_iForest + outlier_Kmeans + outlier_AE_mix

    del TrainData_ML, TestData_ML, TrainData_DL, TestData_DL
    gc.collect()

    # 投票出现了三次的离群点
    count_dict_3_4 = {}
    for item in set(outlier_all):
        count_dict_3_4[item] = outlier_all.count(item)
    result_ensemble_3_4 = [k for k, v in count_dict_3_4.items() if v >= 3]

    # 记录投票产生的最终离群点
    f = open("./resultLog.txt", "a")
    print('**********ensemble 3/4：', result_ensemble_3_4, file=f)
    f.close()


# 运行不同的分类任务
for task in task_list:

    lab_Dataset_train, lab_Dataset_test = Task_Dataset(task)
    TrainData_ML, TestData_ML = DataLoader(lab_Dataset_train, lab_Dataset_test, 'ML')
    TrainData_DL, TestData_DL = DataLoader(lab_Dataset_train, lab_Dataset_test, 'DL')

    # 根据不同比例筛选离群点
    for outlier_ratio in outlier_ratio_list:
        f = open("./resultLog.txt", "a")
        print('********************筛选离群点比例：', outlier_ratio, file=f)
        f.close()
        select_outlier(TrainData_ML, TestData_ML, TrainData_DL, TestData_DL, outlier_ratio)

    del TrainData_ML, TestData_ML, TrainData_DL, TestData_DL
    gc.collect()

    del lab_Dataset_train, lab_Dataset_test
    gc.collect()