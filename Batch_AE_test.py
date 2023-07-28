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
    print('success to load Batch_AutoEncoder model')

    return model_loading


def Batch_AE_model_test(hyp, model_path, traindataset, testdataset):

    # 组织传入训练数据集
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

    model = load_model(hyp, model_path)
    model.eval()

    with torch.no_grad():

        all_tensor_train = torch.empty(0,0)

        for _, x in enumerate(trainDataLoader):

            input_data = x[0]
            _, decodedTestdata = model(input_data)
            # CHD数据集
            all_tensor_train = torch.cat((all_tensor_train.view(-1, 199), decodedTestdata.view(-1, 199)), dim=0)
            # # MI数据集
            # all_tensor_train = torch.cat((all_tensor_train.view(-1, 204), decodedTestdata.view(-1, 204)), dim=0)

    Batch_traindataset_data = all_tensor_train.numpy()
    Batch_traindataset = np.concatenate((traindataset_info, Batch_traindataset_data), axis=1)


    # 组织传入测试数据集
    testdataset_info, testdataset_data = np.split(testdataset, (3,), axis=1)
    testdataset_data = testdataset_data.astype(np.float)
    testdataset_data = stand(testdataset_data)
    testdataset_data = pd.DataFrame(testdataset_data)
    testdataset_data = testdataset_data.fillna(0)
    testdataset_data = np.array(testdataset_data)
    testdataset_data = testdataset_data.astype(np.float)
    testdataset_data = torch.FloatTensor(testdataset_data)
    test_dataset = TensorDataset(testdataset_data)
    testDataLoader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, drop_last=False)


    model = load_model(hyp, model_path)
    model.eval()

    with torch.no_grad():

        all_tensor_test = torch.empty(0,0)

        for _, x in enumerate(testDataLoader):

            input_data = x[0]
            _, decodedTestdata = model(input_data)
            # CHD数据集
            all_tensor_test = torch.cat((all_tensor_test.view(-1, 199), decodedTestdata.view(-1, 199)), dim=0)
            # # MI数据集
            # all_tensor_test = torch.cat((all_tensor_test.view(-1, 204), decodedTestdata.view(-1, 204)), dim=0)

    Batch_testdataset_data = all_tensor_test.numpy()
    Batch_testdataset = np.concatenate((testdataset_info, Batch_testdataset_data), axis=1)


    return Batch_traindataset, Batch_testdataset