import os
import csv
from dataset import protain_dataset
from network import simple_model
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch


def load_data():
    data_file = os.listdir("D:/PROJ/DL/kaggle/train")
    name_labels = []
    data_labels = []
    with open("D:/PROJ/DL/kaggle/labels_train.csv") as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        # header = next(csv_reader)        # 读取第一行每一列的标题
        for i, row in enumerate(csv_reader):  # 将csv 文件中的数据保存到data中
            if i != 0:
                name_labels.append(row[0])  # 选择某一列加入到data数组中
                data_labels.append(row[1])

    return data_file, name_labels, data_labels




def train():
    for epoch in range(20):
        net.train()
        epoch_loss = 0
        for i, data in enumerate(train_dataloader):
            train_data, label = data['data'], data['label']
            train_data = train_data.view(-1, train_data.size(2), train_data.size(3)).permute(1, 0, 2)
            label = label.view(-1, label.size(2))
            train_data = train_data.to(torch.float32)
            label = label.to(torch.float32)
            # train_data, label = train_data.to(torch.float32).to(device), label.to(torch.float32).to(device)


            optimizer.zero_grad()
            output = net(train_data)
            optimizer.step()
            # print('batch loss: ' + str(loss))
        print('epoch loss: ' + str(epoch_loss))


if __name__ == '__main__':
    data_file, name_labels, data_labels = load_data()

    train_dataset = protain_dataset(data_file, name_labels, data_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    net = simple_model()
    fun_loss = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), 0.01, weight_decay=0.01)

    train()
