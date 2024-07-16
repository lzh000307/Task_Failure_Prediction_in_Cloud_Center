import os
import csv

import numpy as np

from ds3 import ProteinDataset
from network7 import FCNForSeq2Seq
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
from sklearn.model_selection import StratifiedKFold

from test_dataset import protain_test_dataset
from test_dataset_2 import ProteinTestDataset


def load_data():
    data_file = os.listdir("D:/PROJ/DL/kaggle/train")
    name_labels = []
    data_labels = []
    with open("D:/PROJ/DL/kaggle/labels_train.csv") as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        for i, row in enumerate(csv_reader):  # 将csv 文件中的数据保存到data中
            if i != 0:  # 跳过标题行
                name_labels.append(row[0])
                data_labels.append(row[1])
    return data_file, name_labels, data_labels

def calculate_accuracy(output, label):
    # 获取模型预测的最可能的类别索引
    _, preds = torch.max(output, 1)
    # 计算正确预测的数量
    correct = torch.sum(preds == label).item()
    # 计算准确率
    accuracy = correct / label.size(0)
    return accuracy


def train(net, device, train_dataloader, optimizer, fun_loss):
    for epoch in range(10):
        net.train()
        epoch_loss = 0
        total_accuracy = 0
        for i, (train_data, label) in enumerate(train_dataloader):
            train_data = train_data.to(device)
            label = label.to(device)
            # label = label.view(-1, label.size(2))
            # _, label_indices = torch.max(label, 1)
            # if label_indices.max() >= 3:
            #     print("存在超出预期范围的标签索引：", label.max().item())

            optimizer.zero_grad()
            output = net(train_data)
            # 如果你的标签是one-hot编码的，找到每个标签的最大值的索引
            # _, label_indices = torch.max(output, 1)

            # 使用标签的类别索引来计算损失
            output = output.transpose(1, 2)  # 调整为 [batch_size, seq_len, num_classes]
            output = output.contiguous().view(-1, 3)  # 展平除了类别维度的所有维度
            label = label.contiguous().view(-1)  # 展平标签

            loss = fun_loss(output, label)  # 计算损失
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            total_accuracy += calculate_accuracy(output, label)

        avg_accuracy = total_accuracy / len(train_dataloader)
        print(f'Epoch [{epoch + 1}], Loss: {epoch_loss:.4f}, Accuracy: {avg_accuracy:.4f}')
        if (epoch + 1) % 5 == 0:
            torch.save(net.state_dict(), f"model_{epoch+1}.pth")
            print(f"Model saved at epoch {epoch + 1}")
            test_dataset = ProteinTestDataset("D:/PROJ/DL/kaggle/test")
            test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            output_csv = f"predictions_{epoch+1}.csv"
            predict_and_save_to_csv(net, device, test_dataloader, output_csv)


def predict_and_save_to_csv(model, device, test_dataloader, output_csv="predictions.csv"):
    model.eval()
    structure_labels = {0: 'C', 1: 'E', 2: 'H'}  # 确保这与训练时的标签对应
    with torch.no_grad(), open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ID', 'STRUCTURE'])
        for data, pdb_id, sec_num, residue_num in test_dataloader:
            data = data.to(device)  # 增加批次维度
            output = model(data)
            _, preds = torch.max(output, 1)
            preds = preds.squeeze().cpu().numpy()  # 移除批次维度并转换为numpy数组
            for i, pred in enumerate(preds):
                # 确保 pdb_id, sec_num, 和 residue_num 被正确地引用为字符串
                writer.writerow([f"{pdb_id[0]}_{sec_num[0]}_{residue_num[0]}_{i + 1}", structure_labels[pred]])

def train_and_validate(net, device, train_dataloader, val_dataloader, optimizer, fun_loss, num_fold=1, epochs=50):
    for epoch in range(epochs):
        net.train()
        train_epoch_loss = 0
        train_total_accuracy = 0

        # 训练过程
        for train_data, label in train_dataloader:
            # print(train_data.shape, label.shape)
            train_data = train_data.to(device)
            label = label.to(device)


            optimizer.zero_grad()
            output = net(train_data)
            # print(output.shape)
            output = output.transpose(1, 2)
            output = output.contiguous().view(-1, 3)
            label = label.contiguous().view(-1)
            # print(output)
            # print(label)
            # print(output.shape)
            # print(label.shape)
            loss = fun_loss(output, label)
            train_epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            train_total_accuracy += calculate_accuracy(output, label)

        train_avg_accuracy = train_total_accuracy / len(train_dataloader)

        # 验证过程
        net.eval()
        with torch.no_grad():
            val_epoch_loss = 0
            val_total_accuracy = 0

            for val_data, label in val_dataloader:
                val_data = val_data.to(device)
                label = label.to(device)

                output = net(val_data)
                output = output.transpose(1, 2).contiguous().view(-1, 3)
                label = label.contiguous().view(-1)

                loss = fun_loss(output, label)
                val_epoch_loss += loss.item()
                val_total_accuracy += calculate_accuracy(output, label)

            val_avg_accuracy = val_total_accuracy / len(val_dataloader)

        print(f'Epoch [{epoch + 1}], Train Loss: {train_epoch_loss:.4f}, Train Accuracy: {train_avg_accuracy:.4f}, Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_avg_accuracy:.4f}')

        if (epoch + 1) % 3 == 0:
            torch.save(net.state_dict(), f"model_cv_{num_fold}_{epoch+1}.pth")
            print(f"Model saved at epoch {num_fold}_{epoch + 1}")
            test_dataset = ProteinTestDataset("D:/PROJ/DL/kaggle/test")
            test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            predict_and_save_to_csv(net, device, test_dataloader, f"predictions_cv_{num_fold}_{epoch+1}.csv")



if __name__ == '__main__':
    data_file, name_labels, data_labels = load_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # 交叉验证设置
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    data_labels_np = np.array(data_labels)  # 确保data_labels是一个numpy数组，便于索引


    # 开始交叉验证的循环
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.array(data_file), data_labels_np)):
        print(f"Fold {fold + 1}/{skf.n_splits}")
        # num of fold
        num_fold = fold + 1

        # 为当前折创建训练和验证数据集
        train_dataset = ProteinDataset(data_file, name_labels, data_labels, indices=train_idx)
        val_dataset = ProteinDataset(data_file, name_labels, data_labels, indices=val_idx)

        # save idx
        np.save(f"train_idx_fold_{fold + 1}.npy", train_idx)
        np.save(f"val_idx_fold_{fold + 1}.npy", val_idx)

        # 创建数据加载器
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        net = FCNForSeq2Seq(20, 3, 0.037762).to(device)
        optimizer = optim.Adam(net.parameters(), lr=0.000196, weight_decay=0.000143)
        fun_loss = nn.CrossEntropyLoss()
        train_and_validate(net, device, train_dataloader, val_dataloader, optimizer, fun_loss, num_fold, epochs=18)

        # net = FCNForSeq2Seq(20, 3, 0.10888).to(device)
        # optimizer = optim.Adam(net.parameters(), lr=0.000116106, weight_decay=0.000103)
        # fun_loss = nn.CrossEntropyLoss()
        # train_and_validate(net, device, train_dataloader, val_dataloader, optimizer, fun_loss, num_fold, epochs=6)

        # 可以在这里保存模型，如果需要的话
        torch.save(net.state_dict(), f"model_fold_{fold + 1}.pth")

    # 保存模型
    # torch.save(net.state_dict(), "model_15.pth")
    # print("Trained model saved!")
    # net = FCNForSeq2Seq(20, 3).to(device)
    # net.load_state_dict(torch.load("model_03090200.pth"))
    # test_dataset = ProteinTestDataset("D:/PROJ/DL/kaggle/test")
    # test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    # predict_and_save_to_csv(net, device, test_dataloader, "predictions.csv")





