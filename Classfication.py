import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
from evaluator import getACC
from sklearn.metrics import roc_auc_score, f1_score
from matplotlib import pyplot as plt


def plot_loss_and_validation(loss_list, acc_list, epoch):

    plt.clf()
    x = range(0, epoch)
    title_c = 'Train LOSS & Validation ACC'
    plt.title(title_c)
    plt.plot(x, loss_list, '-', label='Trian LOSS', color='blue')
    plt.xlabel('epoch')
    plt.plot(x, acc_list, '-', label='Validation ACC', color='red')
    plt.ylabel('Train LOSS & Validation ACC')
    plt.legend()
    save_path = os.path.join('./result', '{}_Train LOSS & Validation ACC.png'.format(datetime.datetime.now()))
    plt.savefig(save_path)

    f = open("./resultLog.txt", "a")
    print('Classfication_loss_acc:', save_path, file=f)
    f.close()


# 全连接神经网络
class Model(nn.Module):
    def __init__(self, hyp):
        super(Model, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(hyp['input_dim'], hyp['hidden_dim']), nn.BatchNorm1d(hyp['hidden_dim']), nn.ReLU(), nn.Dropout(p=hyp['drop_out']))
        self.layer2 = nn.Sequential(nn.Linear(hyp['hidden_dim'], hyp['output_dim']), nn.Softmax(dim=1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        return x


# # 一维卷积神经网络
# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.model1 = nn.Sequential(
#             #输入通道一定为1，输出通道为卷积核的个数，2为卷积核的大小（实际为一个[1,2]大小的卷积核）
#             nn.Conv1d(1, 16, 3),
#             nn.Sigmoid(),
#             nn.AvgPool1d(2),
#             nn.Conv1d(16, 32, 2),
#             nn.Sigmoid(),
#             nn.AvgPool1d(2),
#             nn.Conv1d(32, 16, 3),
#             nn.Flatten(),  # 扁平化
#         )
#         self.model2 = nn.Sequential(
#             nn.Linear(in_features=768, out_features=2, bias=True),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, input):
#         x = self.model1(input)
#         x = self.model2(x)
#         return x


def Classfication(hyp, traindataset, trainlabel, testdataset, testlabel):

    # 处理组织数据集
    TrainDataSet = TensorDataset(traindataset, trainlabel)
    TestDataSet = TensorDataset(testdataset, testlabel)
    trainDataLoader = DataLoader(dataset=TrainDataSet, batch_size=hyp['batch_size'], shuffle=True, drop_last=True)
    testDataLoader = DataLoader(dataset=TestDataSet, batch_size=hyp['batch_size'], shuffle=True, drop_last=False)

    print('==> Building Classfication model..')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loss_avg_List = []
    val_loss_avg_List = []
    val_acc_List = []
    val_auc_List = []
    val_f_score_List = []

    model = Model(hyp).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hyp['lr'])
    acc_val = np.zeros((hyp['epochs'], 1))

    for epoch in range(hyp['epochs']):

        # 训练
        train_loss_List = []

        for batchidx, (input_data, label) in enumerate(trainDataLoader):

            optimizer.zero_grad()
            model.train()

            # # 卷积网络才有这一行
            # # CHD数据集
            # input_data = input_data.reshape(-1, 1, 199)
            # # MI数据集
            # input_data = input_data.reshape(-1, 1, 204)

            outputs = model(input_data.to(device))

            # # 卷积神经网络才有这一行
            # outputs = torch.squeeze(outputs, dim=0)

            label = label.long().to(device)
            label = label.squeeze(dim=1)

            loss = criterion(outputs, label)

            loss_save = loss.item()
            train_loss_List.append(loss_save)

            loss.backward()
            optimizer.step()

        train_loss_avg_List.append(np.mean(train_loss_List))

        # 测试
        val_loss_List = []
        y_true = torch.tensor([]).to(device)
        y_score = torch.tensor([]).to(device)
        model.eval()

        with torch.no_grad():
            for batchidx, (input_data, label) in enumerate(testDataLoader):

                # # 卷积网络才有这一行
                # # CHD数据集
                # input_data = input_data.reshape(-1, 1, 199)
                # # MI数据集
                # input_data = input_data.reshape(-1, 1, 204)

                outputs = model(input_data.to(device))

                # # 卷积神经网络才有这一行
                # outputs = torch.squeeze(outputs, dim=0)

                label = label.long().to(device)
                label = label.squeeze(dim=1)

                loss = criterion(outputs, label).to(device)
                loss = loss.item()
                val_loss_List.append(loss)

                y_true = torch.cat((y_true, label), 0)
                y_score = torch.cat((y_score, outputs), 0)

            val_loss_avg_List.append(np.mean(val_loss_List))
            y_true = y_true.cpu().numpy()
            y_score = y_score.detach().cpu().numpy()
            y_pred = y_score[:, 1]

            acc = getACC(y_true, y_score)
            val_acc_List.append(acc)

            auc_score = roc_auc_score(y_true, y_pred)
            val_auc_List.append(auc_score)

            y_pred_f = (y_score[:, 1] >= 0.5).astype(int)  # 预测值二值化，用于计算F-Score
            f_score = f1_score(y_true, y_pred_f)
            val_f_score_List.append(f_score)

        acc_val[epoch, 0] = np.max(val_acc_List)

    # 保存测试acc最大的模型
    best_acc = float('-inf')  # 初始化一个超大的值
    best_epoch = -1  # 初始化一个不合法的值

    for epoch, acc in enumerate(acc_val):
        if acc[0] > best_acc:
            best_acc = acc[0]
            best_epoch = epoch

    # 保存模型
    torch.save(model.state_dict(), os.path.join('./models', '{}_{}_model.pth'.format(best_epoch, best_acc)))

    f = open("./resultLog.txt", "a")
    print('The best ACC: %.5f' % (max(val_acc_List)), file=f)
    f.close()

    f = open("./resultLog.txt", "a")
    print('The best AUC: %.5f' % (max(val_auc_List)), file=f)
    f.close()

    f = open("./resultLog.txt", "a")
    print('The best F_score: %.5f' % (max(val_f_score_List)), file=f)
    f.close()

    plot_loss_and_validation(train_loss_avg_List, val_acc_List, hyp['epochs'])