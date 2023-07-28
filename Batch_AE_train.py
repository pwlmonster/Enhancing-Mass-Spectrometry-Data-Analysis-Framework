import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt


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


def Batch_AE_model_train(hyp, traindataset):

    # 组织传入数据集
    traindataset_info, traindataset_data = np.split(traindataset, (3,), axis=1)
    traindataset_data = traindataset_data.astype(np.float)
    traindataset_data = stand(traindataset_data)
    traindataset_data = pd.DataFrame(traindataset_data)
    traindataset_data = traindataset_data.fillna(0)
    traindataset_data = np.array(traindataset_data)
    traindataset_data = traindataset_data.astype(np.float)
    traindataset_data = torch.FloatTensor(traindataset_data)
    train_dataset = TensorDataset(traindataset_data, traindataset_data)
    trainDataLoader = DataLoader(dataset=train_dataset, batch_size=hyp['batch_size'], shuffle=True, drop_last=False)

    print('==> Building Batch_AutoEncoder model..')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loss_List = []

    model = AutoEncoder(hyp).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=hyp['lr'])
    loss_train = np.zeros((hyp['epochs'], 1))

    for epoch in range(hyp['epochs']):
        for batchidx, (x, _) in enumerate(trainDataLoader):

            x = x.to(device)
            encoded, decoded = model(x)

            loss = criterion(decoded, x)
            loss_save = loss.item()
            train_loss_List.append(loss_save)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_train[epoch, 0] = np.mean(train_loss_List)

    # 保存训练loss最小的模型
    best_loss = float('inf')  # 初始化一个超大的值
    best_epoch = -1  # 初始化一个不合法的值

    for epoch, loss_val in enumerate(loss_train):
        if loss_val[0] < best_loss:
            best_loss = loss_val[0]
            best_epoch = epoch

    # 保存模型
    torch.save(model.state_dict(), os.path.join('./models', '{}_{}_Batch_AE.pth'.format(best_epoch, best_loss)))
    model_path = os.path.join('./models', '{}_{}_Batch_AE.pth'.format(best_epoch, best_loss))

    f = open("./resultLog.txt", "a")
    print('Batch_AutoEncoder_model_path:', model_path, file=f)
    f.close()

    # 绘制训练的loss曲线
    fig = plt.figure(figsize=(12, 6))
    ax = plt.subplot(1, 1, 1)
    ax.plot(loss_train, color='blue', linestyle='-', linewidth=2)
    ax.set_xlabel('Epoches')
    ax.set_ylabel('Loss')
    loss_path = os.path.join('./result/{}_{}_loss_Batch_AE.png'.format(best_epoch, best_loss))
    plt.savefig(loss_path)

    f = open("./resultLog.txt", "a")
    print('Batch_AutoEncoder_loss_png:', loss_path, file=f)
    f.close()

    return model_path