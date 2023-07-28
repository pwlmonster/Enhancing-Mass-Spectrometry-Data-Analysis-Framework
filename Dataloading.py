import csv
import numpy as np
import pandas as pd
import torch


# 数据标准化
def stand(x):
    x = (x-x.mean(axis=0))/x.std(axis=0)
    return x


# 读取单个CSV文件下的数据
def read_path_data(file_pathname):

    data = []

    with open(file_pathname, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            data.append(row)

    data = np.array(data)

    # 将质谱数据进行数据标准化,针对transformer
    data_arr_rest = data[:, :3]
    data_arr_new = data[:, 3:]
    data_arr_new = data_arr_new.astype(np.float)
    data_arr_new = stand(data_arr_new)
    data_arr_new = pd.DataFrame(data_arr_new)
    data_arr_new = data_arr_new.fillna(0)
    data_arr_new = np.array(data_arr_new)

    # 获取目标矩阵
    result = np.concatenate((data_arr_rest, data_arr_new), axis=1)

    return result


# 定义某一具体分类任务所需的数据集：
def Task_Dataset(Task_name):

    if Task_name == 'Task1':
        train_1_arr = read_path_data('./CHD/1.csv')
        train_2_arr = read_path_data('./CHD/2.csv')
        train_3_arr = read_path_data('./CHD/3.csv')

        result_train = np.concatenate((train_1_arr, train_2_arr, train_3_arr), axis=0)

        test_arr = read_path_data('./CHD/4.csv')

        result_test = test_arr

        return result_train, result_test

    if Task_name == 'Task2':
        train_1_arr = read_path_data('./CHD/1.csv')
        train_2_arr = read_path_data('./CHD/2.csv')
        train_3_arr = read_path_data('./CHD/4.csv')

        result_train = np.concatenate((train_1_arr, train_2_arr, train_3_arr), axis=0)

        test_arr = read_path_data('./CHD/3.csv')

        result_test = test_arr

        return result_train, result_test

    if Task_name == 'Task3':
        train_1_arr = read_path_data('./CHD/1.csv')
        train_2_arr = read_path_data('./CHD/3.csv')
        train_3_arr = read_path_data('./CHD/4.csv')

        result_train = np.concatenate((train_1_arr, train_2_arr, train_3_arr), axis=0)

        test_arr = read_path_data('./CHD/2.csv')

        result_test = test_arr

        return result_train, result_test

    if Task_name == 'Task4':
        train_1_arr = read_path_data('./CHD/2.csv')
        train_2_arr = read_path_data('./CHD/3.csv')
        train_3_arr = read_path_data('./CHD/4.csv')

        result_train = np.concatenate((train_1_arr, train_2_arr, train_3_arr), axis=0)

        test_arr = read_path_data('./CHD/1.csv')

        result_test = test_arr

        return result_train, result_test

    if Task_name == 'Task5':
        train_1_arr = read_path_data('./CHD/1.csv')
        train_2_arr = read_path_data('./CHD/2.csv')

        result_train = np.concatenate((train_1_arr, train_2_arr), axis=0)

        test_arr = read_path_data('./CHD/3.csv')

        result_test = test_arr

        return result_train, result_test

    if Task_name == 'Task6':
        train_1_arr = read_path_data('./CHD/1.csv')
        train_2_arr = read_path_data('./CHD/2.csv')

        result_train = np.concatenate((train_1_arr, train_2_arr), axis=0)

        test_arr = read_path_data('./CHD/4.csv')

        result_test = test_arr

        return result_train, result_test

    if Task_name == 'Task7':
        train_1_arr = read_path_data('./CHD/1.csv')
        train_2_arr = read_path_data('./CHD/3.csv')

        result_train = np.concatenate((train_1_arr, train_2_arr), axis=0)

        test_arr = read_path_data('./CHD/2.csv')

        result_test = test_arr

        return result_train, result_test

    if Task_name == 'Task8':
        train_1_arr = read_path_data('./CHD/1.csv')
        train_2_arr = read_path_data('./CHD/3.csv')

        result_train = np.concatenate((train_1_arr, train_2_arr), axis=0)

        test_arr = read_path_data('./CHD/4.csv')

        result_test = test_arr

        return result_train, result_test

    if Task_name == 'Task9':
        train_1_arr = read_path_data('./CHD/1.csv')
        train_2_arr = read_path_data('./CHD/4.csv')

        result_train = np.concatenate((train_1_arr, train_2_arr), axis=0)

        test_arr = read_path_data('./CHD/2.csv')

        result_test = test_arr

        return result_train, result_test

    if Task_name == 'Task10':
        train_1_arr = read_path_data('./CHD/1.csv')
        train_2_arr = read_path_data('./CHD/4.csv')

        result_train = np.concatenate((train_1_arr, train_2_arr), axis=0)

        test_arr = read_path_data('./CHD/3.csv')

        result_test = test_arr

        return result_train, result_test

    if Task_name == 'Task11':
        train_1_arr = read_path_data('./CHD/2.csv')
        train_2_arr = read_path_data('./CHD/3.csv')

        result_train = np.concatenate((train_1_arr, train_2_arr), axis=0)

        test_arr = read_path_data('./CHD/1.csv')

        result_test = test_arr

        return result_train, result_test

    if Task_name == 'Task12':
        train_1_arr = read_path_data('./CHD/2.csv')
        train_2_arr = read_path_data('./CHD/3.csv')

        result_train = np.concatenate((train_1_arr, train_2_arr), axis=0)

        test_arr = read_path_data('./CHD/4.csv')

        result_test = test_arr

        return result_train, result_test

    if Task_name == 'Task13':
        train_1_arr = read_path_data('./CHD/2.csv')
        train_2_arr = read_path_data('./CHD/4.csv')

        result_train = np.concatenate((train_1_arr, train_2_arr), axis=0)

        test_arr = read_path_data('./CHD/1.csv')

        result_test = test_arr

        return result_train, result_test

    if Task_name == 'Task14':
        train_1_arr = read_path_data('./CHD/2.csv')
        train_2_arr = read_path_data('./CHD/4.csv')

        result_train = np.concatenate((train_1_arr, train_2_arr), axis=0)

        test_arr = read_path_data('./CHD/3.csv')

        result_test = test_arr

        return result_train, result_test

    if Task_name == 'Task15':
        train_1_arr = read_path_data('./CHD/3.csv')
        train_2_arr = read_path_data('./CHD/4.csv')

        result_train = np.concatenate((train_1_arr, train_2_arr), axis=0)

        test_arr = read_path_data('./CHD/1.csv')

        result_test = test_arr

        return result_train, result_test

    if Task_name == 'Task16':
        train_1_arr = read_path_data('./CHD/3.csv')
        train_2_arr = read_path_data('./CHD/4.csv')

        result_train = np.concatenate((train_1_arr, train_2_arr), axis=0)

        test_arr = read_path_data('./CHD/2.csv')

        result_test = test_arr

        return result_train, result_test

    if Task_name == 'Task17':
        train_1_arr = read_path_data('./MI/1.csv')
        train_2_arr = read_path_data('./MI/2.csv')

        result_train = np.concatenate((train_1_arr, train_2_arr), axis=0)

        test_arr = read_path_data('./MI/3.csv')

        result_test = test_arr

        return result_train, result_test

    if Task_name == 'Task18':
        train_1_arr = read_path_data('./MI/1.csv')
        train_2_arr = read_path_data('./MI/3.csv')

        result_train = np.concatenate((train_1_arr, train_2_arr), axis=0)

        test_arr = read_path_data('./MI/2.csv')

        result_test = test_arr

        return result_train, result_test

    if Task_name == 'Task19':
        train_1_arr = read_path_data('./MI/2.csv')
        train_2_arr = read_path_data('./MI/3.csv')

        result_train = np.concatenate((train_1_arr, train_2_arr), axis=0)

        test_arr = read_path_data('./MI/1.csv')

        result_test = test_arr

        return result_train, result_test

    if Task_name == 'Task20':
        train_1_arr = read_path_data('./CHD/1.csv')
        train_2_arr = read_path_data('./CHD/2.csv')
        train_3_arr = read_path_data('./CHD/3.csv')
        train_4_arr = read_path_data('./CHD/4.csv')

        result_train = np.concatenate((train_1_arr, train_2_arr, train_3_arr, train_4_arr), axis=0)

        test_arr = np.empty([2, 202])

        result_test = test_arr

        return result_train, result_test

    if Task_name == 'Task21':
        train_1_arr = read_path_data('./MI/1.csv')
        train_2_arr = read_path_data('./MI/2.csv')
        train_3_arr = read_path_data('./MI/3.csv')

        result_train = np.concatenate((train_1_arr, train_2_arr, train_3_arr), axis=0)

        test_arr = np.empty([2, 207])

        result_test = test_arr

        return result_train, result_test


def DataLoader(traindataset, testdataset, Task_type):

    if Task_type == 'ML':
        traindataset_new = pd.DataFrame(traindataset)
        traindataset_new.columns = traindataset_new.columns.astype(str)

        testdataset_new = pd.DataFrame(testdataset)
        testdataset_new.columns = testdataset_new.columns.astype(str)

        return traindataset_new, testdataset_new

    if Task_type == 'DL':

        return traindataset, testdataset

    if Task_type == 'CL':

        traindataset_info, traindataset_data = np.split(traindataset, (2,), axis=1)
        traindataset_data = traindataset_data.astype(np.float)

        testdataset_info, testdataset_data = np.split(testdataset, (2,), axis=1)
        testdataset_data = testdataset_data.astype(np.float)

        # 分割数据与标签
        TrainLabel, TrainData = np.split(traindataset_data, (1,), axis=1)
        TestLabel, TestData = np.split(testdataset_data, (1,), axis=1)

        # 转换为张量
        TrainData = torch.FloatTensor(TrainData)
        TestData = torch.FloatTensor(TestData)
        TrainLabel = torch.IntTensor(TrainLabel)
        TestLabel = torch.IntTensor(TestLabel)

        return TrainData, TrainLabel, TestData, TestLabel