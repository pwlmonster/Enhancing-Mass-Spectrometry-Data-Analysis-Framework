import numpy as np
import pandas as pd
import math

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def stand(x):
    x = (x-x.mean(axis=0))/x.std(axis=0)
    return x


class AutoEncoder(nn.Module):
    def __init__(self, hyp):
        super(AutoEncoder, self).__init__()

        layers = []
        layers += [nn.Linear(hyp['input_dim'], hyp['hidden_dim'])]
        layers += [nn.Tanh()]

        self.encoder = nn.Sequential(*layers)

        layers = []
        layers += [nn.Linear(hyp['hidden_dim'], hyp['input_dim'])]
        layers += [nn.Tanh()]

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)

        return enc, dec


def load_model(hyp, model_path):

    model_loading = AutoEncoder(hyp)
    model_loading.load_state_dict(torch.load(model_path))
    print('success to load AutoEncoder model')

    return model_loading


def AE_model_test(hyp, model_path, traindataset, testdataset, outlier_ratio):

    # 组织传入数据集
    traindataset_info, traindataset_data = np.split(traindataset, (3,), axis=1)
    traindataset_data = traindataset_data.astype(np.float)
    traindataset_data = stand(traindataset_data)
    traindataset_data = pd.DataFrame(traindataset_data)
    traindataset_data = traindataset_data.fillna(0)
    traindataset_data = np.array(traindataset_data)
    traindataset_data = traindataset_data.astype(np.float)
    traindataset_data = torch.FloatTensor(traindataset_data)
    train_dataset = TensorDataset(traindataset_data)
    trainDataLoader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False, drop_last=False)

    # 定义重构误差列表
    se_total = []

    model = load_model(hyp, model_path)
    model.eval()

    with torch.no_grad():
        for _, x in enumerate(trainDataLoader):

            input_data = x[0]
            _, decodedTestdata = model(input_data)

            criterion = nn.MSELoss()
            se = criterion(decodedTestdata, input_data)
            se = se.item()
            se_total.append(se)

        se_arr = np.array(se_total)
        se_arr = se_arr.reshape(se_arr.shape[0], 1)

    # 筛选离群点
    final_ratio = math.floor(outlier_ratio * traindataset.shape[0])
    traindataset_withoutlier = np.concatenate((traindataset, se_arr), axis=1)
    traindataset_withoutlier = pd.DataFrame(traindataset_withoutlier)
    # CHD数据集
    outlier_result = traindataset_withoutlier.sort_values(202, ascending=False).head(final_ratio)
    outlier_score = traindataset_withoutlier.sort_values(202, ascending=False).head(50)
    # # MI数据集
    # outlier_result = traindataset_withoutlier.sort_values(207, ascending=False).head(final_ratio)
    # outlier_score = traindataset_withoutlier.sort_values(207, ascending=False).head(50)

    mask = ~traindataset_withoutlier.isin(outlier_result)
    data_new = traindataset_withoutlier[mask]
    data_new = data_new.dropna(how='all')

    # 输出离群点列表
    outlier_name_list = outlier_result.iloc[:, 0].tolist()

    f = open("./resultLog.txt", "a")
    print('AutoEncoder, %.2f, outlier:' % (outlier_ratio), outlier_name_list, file=f)
    f.close()

    # f = open("./resultLog.txt", "a")
    # print(outlier_score, file=f)
    # f.close()

    return outlier_name_list

#     # 生成去除离群点后的数据集
#     # 去除无关信息
#     traindataset_new = data_new.iloc[:, 2:]
#     traindataset_new = traindataset_new.iloc[:, :-1]
#     testdataset = testdataset.reshape(5 * testdataset.shape[0], -1)
#     testdataset_new = pd.DataFrame(testdataset)
#     testdataset_new = testdataset_new.iloc[:, 2:]

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