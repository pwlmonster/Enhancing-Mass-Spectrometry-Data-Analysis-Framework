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
from Classfication_transformer import Classfication_transformer


# 所需要执行的任务列表
task_list = ['Task1', 'Task2', 'Task3', 'Task4']

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

# Classfication_transformer的网络参数
# 二分类
hyp_Classfication_transformer = {
    'num_layers': 2,
    'attention_heads': 8,
    'dropout': 0.5,
    'batch_size': 128,
    'lr': 0.0001,
    'epochs': 200,
}


def batch_off(TrainData_DL, TestData_DL):

    # AutoEncoder模型训练
    Batch_AE_model_path = Batch_AE_model_train(hyp_AE, TrainData_DL)
    # 得到去除批次效应的数据集
    Batch_lab_Dataset_train, Batch_lab_Dataset_test = Batch_AE_model_test(hyp_AE, Batch_AE_model_path, TrainData_DL, TestData_DL)

    del TrainData_DL, TestData_DL
    gc.collect()

    return Batch_lab_Dataset_train, Batch_lab_Dataset_test


# 运行不同的分类任务
for task in task_list:

    f = open("./resultLog.txt", "a")
    print('********************数据集任务：', task, file=f)
    f.close()

    # 定义离群点
    outlier_list = []
    if task == 'Task17':
        outlier_list = ['pt2034', 'pt2271', 'pt2455', 'pt90', 'pt1776', 'pt1415', 'pt1215', 'pt2219', 'pt2270', 'pt1346', 'pt1773', 'pt2509', 'pt647', 'pt2454', 'pt2036', 'pt2268', 'pt648', 'pt646', 'pt2269', 'pt2452', 'pt2453', 'pt650', 'pt2033', 'pt87', 'pt2192', 'pt1155', 'pt6', 'pt2158', 'pt649', 'pt2117', 'pt2508', 'pt2221', 'pt770', 'pt1964', 'pt769', 'pt2195', 'pt1774', 'pt1854', 'pt89', 'pt2507', 'pt2119', 'pt2267', 'pt2121', 'pt1416', 'pt2032', 'pt2456', 'pt2035', 'pt2026', 'pt720', 'pt86', 'pt1775', 'pt768', 'pt2120', 'pt2224', 'pt2510', 'pt1412', 'pt88', 'pt1414', 'pt2118', 'pt2223']
    if task == 'Task18':
        outlier_list = ['pt3646', 'pt3647', 'pt1148', 'pt2845', 'pt788', 'pt3649', 'pt3415', 'pt3366', 'pt2674', 'pt2844', 'pt538', 'pt3411', 'pt1271', 'pt150', 'pt647', 'pt3851', 'pt71', 'pt3412', 'pt3369', 'pt786', 'pt1200', 'pt1274', 'pt789', 'pt1146', 'pt3414', 'pt2671', 'pt1150', 'pt2713', 'pt453', 'pt3304', 'pt73', 'pt1147', 'pt648', 'pt646', 'pt3650', 'pt2711', 'pt3301', 'pt650', 'pt3413', 'pt2673', 'pt3367', 'pt2714', 'pt1056', 'pt2743', 'pt2741', 'pt4', 'pt649', 'pt2842', 'pt2712', 'pt3302', 'pt231', 'pt769', 'pt3305', 'pt3303', 'pt2672', 'pt1272', 'pt3855', 'pt3368', 'pt1149', 'pt787', 'pt2742', 'pt3852', 'pt3853', 'pt233', 'pt3648', 'pt2715', 'pt768', 'pt3684', 'pt767', 'pt2744', 'pt790']
    if task == 'Task19':
        outlier_list = ['pt3646', 'pt3647', 'pt2334', 'pt2034', 'pt2845', 'pt2271', 'pt2455', 'pt3415', 'pt1415', 'pt1776', 'pt3366', 'pt2674', 'pt2270', 'pt3411', 'pt1346', 'pt2377', 'pt1773', 'pt2509', 'pt3851', 'pt2454', 'pt3412', 'pt3369', 'pt3414', 'pt2036', 'pt2671', 'pt2268', 'pt3370', 'pt3304', 'pt2269', 'pt3650', 'pt2711', 'pt2452', 'pt2453', 'pt3301', 'pt2033', 'pt3413', 'pt2335', 'pt2333', 'pt2592', 'pt3367', 'pt2714', 'pt2743', 'pt2741', 'pt2158', 'pt2591', 'pt2842', 'pt2712', 'pt2117', 'pt2508', 'pt3302', 'pt1774', 'pt1854', 'pt2507', 'pt3305', 'pt3303', 'pt2672', 'pt2119', 'pt2267', 'pt1772', 'pt2121', 'pt3368', 'pt2745', 'pt1416', 'pt2032', 'pt2742', 'pt2456', 'pt2332', 'pt3853', 'pt1775', 'pt2715', 'pt2120', 'pt2510', 'pt1414', 'pt2744', 'pt2336', 'pt2118']
    if task == 'Task1':
        outlier_list = ['pt1726', 'pt2334', 'pt2630', 'pt3157', 'pt780', 'pt111', 'pt3723', 'pt3911', 'pt2912', 'pt912', 'pt2306', 'pt114', 'pt2937', 'pt2853', 'pt2631', 'pt3155', 'pt2923', 'pt1736', 'pt1167', 'pt112', 'pt37', 'pt3160', 'pt2619', 'pt64', 'pt2616', 'pt2935', 'pt62', 'pt2685', 'pt1732', 'pt1025', 'pt2335', 'pt3153', 'pt38', 'pt2333', 'pt2936', 'pt782', 'pt783', 'pt3159', 'pt3724', 'pt1730', 'pt2689', 'pt2850', 'pt2686', 'pt2632', 'pt2934', 'pt2690', 'pt2852', 'pt2851', 'pt2337', 'pt2634', 'pt2336', 'pt2909', 'pt1728', 'pt2693', 'pt2849', 'pt2922', 'pt1735', 'pt2913', 'pt1734', 'pt2622', 'pt3020', 'pt3908', 'pt1737', 'pt1738', 'pt3022', 'pt3154', 'pt1725', 'pt2617', 'pt113', 'pt784', 'pt3907', 'pt2911', 'pt3725', 'pt1731', 'pt61', 'pt2688', 'pt3021', 'pt3910', 'pt1733', 'pt39', 'pt2618', 'pt1727', 'pt40', 'pt3161', 'pt36', 'pt2938', 'pt3726', 'pt786', 'pt3727', 'pt3156', 'pt3158', 'pt2687', 'pt3909', 'pt4479', 'pt2633', 'pt781', 'pt1729', 'pt2910', 'pt3018', 'pt4141', 'pt1724', 'pt3019', 'pt115']
    if task == 'Task2':
        outlier_list = ['pt1726', 'pt1968', 'pt2334', 'pt2630', 'pt780', 'pt111', 'pt2912', 'pt5068', 'pt2306', 'pt114', 'pt2937', 'pt2853', 'pt4678', 'pt2113', 'pt1736', 'pt1165', 'pt1167', 'pt112', 'pt37', 'pt990', 'pt2619', 'pt2616', 'pt5066', 'pt2935', 'pt1168', 'pt2685', 'pt1732', 'pt1025', 'pt2335', 'pt38', 'pt2333', 'pt2936', 'pt782', 'pt783', 'pt1730', 'pt2689', 'pt2850', 'pt2686', 'pt991', 'pt2934', 'pt2690', 'pt2852', 'pt4677', 'pt2851', 'pt2337', 'pt5077', 'pt2336', 'pt2909', 'pt2112', 'pt1728', 'pt2693', 'pt2849', 'pt5065', 'pt1735', 'pt2913', 'pt1734', 'pt2580', 'pt2622', 'pt3020', 'pt1737', 'pt3022', 'pt1738', 'pt5064', 'pt5075', 'pt1725', 'pt2617', 'pt113', 'pt784', 'pt5074', 'pt2911', 'pt1166', 'pt1731', 'pt1432', 'pt61', 'pt3021', 'pt2688', 'pt1733', 'pt39', 'pt2618', 'pt1727', 'pt40', 'pt840', 'pt1169', 'pt1433', 'pt36', 'pt2938', 'pt2579', 'pt2687', 'pt186', 'pt1430', 'pt781', 'pt1729', 'pt2910', 'pt1724', 'pt115', 'pt5067']
    if task == 'Task3':
        outlier_list = ['pt1428', 'pt199', 'pt5076', 'pt780', 'pt111', 'pt5078', 'pt693', 'pt196', 'pt3911', 'pt5068', 'pt1405', 'pt1407', 'pt1426', 'pt200', 'pt114', 'pt926', 'pt1429', 'pt1165', 'pt112', 'pt37', 'pt811', 'pt990', 'pt1408', 'pt812', 'pt941', 'pt1117', 'pt5066', 'pt943', 'pt814', 'pt38', 'pt1434', 'pt782', 'pt783', 'pt3159', 'pt3724', 'pt929', 'pt1409', 'pt991', 'pt690', 'pt944', 'pt5077', 'pt197', 'pt1118', 'pt190', 'pt189', 'pt940', 'pt1145', 'pt1119', 'pt5065', 'pt188', 'pt928', 'pt3908', 'pt5064', 'pt5075', 'pt113', 'pt784', 'pt5074', 'pt3907', 'pt1166', 'pt1431', 'pt1432', 'pt3910', 'pt39', 'pt198', 'pt1123', 'pt40', 'pt840', 'pt3161', 'pt1433', 'pt36', 'pt691', 'pt820', 'pt3909', 'pt1427', 'pt1116', 'pt186', 'pt1430', 'pt942', 'pt781', 'pt1425', 'pt813', 'pt1406', 'pt187', 'pt115', 'pt821', 'pt5067', 'pt810', 'pt925']
    if task == 'Task4':
        outlier_list = ['pt1726', 'pt2334', 'pt5076', 'pt2630', 'pt3157', 'pt2925', 'pt2983', 'pt2928', 'pt5078', 'pt2926', 'pt3911', 'pt2912', 'pt5068', 'pt2669', 'pt1792', 'pt2306', 'pt2646', 'pt2937', 'pt2853', 'pt2927', 'pt2113', 'pt2578', 'pt3155', 'pt1736', 'pt2978', 'pt2649', 'pt2619', 'pt2616', 'pt5066', 'pt2935', 'pt2685', 'pt1732', 'pt2335', 'pt3153', 'pt2647', 'pt2333', 'pt2936', 'pt3724', 'pt1730', 'pt2689', 'pt2850', 'pt2686', 'pt5016', 'pt1790', 'pt2934', 'pt2690', 'pt2852', 'pt2851', 'pt2337', 'pt5077', 'pt2336', 'pt2909', 'pt2112', 'pt2986', 'pt1728', 'pt2693', 'pt2577', 'pt2849', 'pt5065', 'pt1735', 'pt2913', 'pt1734', 'pt2580', 'pt2622', 'pt2984', 'pt3020', 'pt3908', 'pt1737', 'pt3022', 'pt1738', 'pt3154', 'pt5064', 'pt5075', 'pt1725', 'pt2985', 'pt2617', 'pt5074', 'pt3907', 'pt2911', 'pt1731', 'pt3021', 'pt2688', 'pt3910', 'pt1733', 'pt2618', 'pt1727', 'pt2938', 'pt3726', 'pt2576', 'pt2579', 'pt3727', 'pt3156', 'pt2687', 'pt3909', 'pt1729', 'pt2910', 'pt4141', 'pt1724', 'pt5067', 'pt2648']

    lab_Dataset_train, lab_Dataset_test = Task_Dataset(task)

    names = lab_Dataset_train[:, 0]
    remain = ~np.isin(names, outlier_list)
    lab_Dataset_train = lab_Dataset_train[remain]

    TrainData_DL, TestData_DL = DataLoader(lab_Dataset_train, lab_Dataset_test, 'DL')

    Batch_lab_Dataset_train, Batch_lab_Dataset_test = batch_off(TrainData_DL, TestData_DL)
    Batch_TrainData_DL, Batch_TestData_DL = DataLoader(Batch_lab_Dataset_train, Batch_lab_Dataset_test, 'DL')

    Batch_TrainData_CL, Batch_TrainLabel_CL, Batch_TestData_CL, Batch_TestLabel_CL = DataLoader(Batch_lab_Dataset_train, Batch_lab_Dataset_test, 'CL')

    Classfication_transformer(hyp_Classfication_transformer, Batch_TrainData_CL, Batch_TrainLabel_CL, Batch_TestData_CL, Batch_TestLabel_CL)

    del TrainData_DL, TestData_DL, Batch_TrainData_CL, Batch_TrainLabel_CL, Batch_TestData_CL, Batch_TestLabel_CL
    gc.collect()

    del Batch_TrainData_DL, Batch_TestData_DL
    gc.collect()

    del lab_Dataset_train, lab_Dataset_test, Batch_lab_Dataset_train, Batch_lab_Dataset_test
    gc.collect()