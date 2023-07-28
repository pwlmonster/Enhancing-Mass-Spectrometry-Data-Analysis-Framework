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


class TransformerModel(nn.Module):
    def __init__(self, hyp):
        super(TransformerModel, self).__init__()
        # self.fc0 = nn.Sequential(nn.Linear(1, 16))
        self.fc0 = nn.Conv1d(1, 16, 3, padding=1)
        encoder_layers = nn.TransformerEncoderLayer(16, hyp['attention_heads'])
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, hyp['num_layers'])
        # # MI数据集
        # self.BN1 = nn.BatchNorm1d(204 * 16)
        # CHD数据集
        self.BN1 = nn.BatchNorm1d(199 * 16)
        self.BN2 = nn.BatchNorm1d(512)
        self.BN3 = nn.BatchNorm1d(128)
        self.fc1 = nn.Linear(16, 128)
        self.fc2 = nn.Linear(128, 16)
        # # MI数据集
        # self.fc3 = nn.Linear(204 * 16, 512)
        # CHD数据集
        self.fc3 = nn.Linear(199 * 16, 512)
        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, 2)


    def forward(self, x):
        b, l = x.shape
        x = torch.unsqueeze(x, dim=2).permute(0, 2, 1)
        x = self.fc0(x).permute(0, 2, 1)
        x = self.transformer_encoder(x)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        # # MI数据集
        # x = x.reshape((b, 204 * 16))
        # CHD数据集
        x = x.reshape((b, 199 * 16))
        x = self.BN1(x)
        x = nn.functional.relu(self.fc3(x))
        x = self.BN2(x)
        x = nn.functional.relu(self.fc4(x))
        x = self.BN3(x)
        x = nn.functional.softmax(self.fc5(x), dim=1)

        return x


def Classfication_transformer(hyp, traindataset, trainlabel, testdataset, testlabel):

    # 处理组织数据集
    TrainDataSet = TensorDataset(traindataset, trainlabel)
    TestDataSet = TensorDataset(testdataset, testlabel)
    trainDataLoader = DataLoader(dataset=TrainDataSet, batch_size=hyp['batch_size'], shuffle=True, drop_last=True)
    testDataLoader = DataLoader(dataset=TestDataSet, batch_size=hyp['batch_size'], shuffle=True, drop_last=False)

    print('==> Building Classfication_transformer model..')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loss_avg_List = []
    val_loss_avg_List = []
    val_acc_List = []
    val_auc_List = []
    val_f_score_List = []

    model = TransformerModel(hyp).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hyp['lr'])
    acc_val = np.zeros((hyp['epochs'], 1))

    for epoch in range(hyp['epochs']):

        # 训练
        train_loss_List = []

        for batchidx, (input_data, label) in enumerate(trainDataLoader):

            optimizer.zero_grad()
            model.train()

            outputs = model(input_data.to(device))

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

                outputs = model(input_data.to(device))

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