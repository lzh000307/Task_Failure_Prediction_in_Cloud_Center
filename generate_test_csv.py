import os
import csv
from ds3 import ProteinDataset
from network5 import FCNForSeq2Seq
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch

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
    for epoch in range(50):
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
        if (epoch + 1) % 10 == 0:
            torch.save(net.state_dict(), f"model_{epoch+1}.pth")
            print(f"Model saved at epoch {epoch + 1}")


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


if __name__ == '__main__':
    data_file, name_labels, data_labels = load_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"Using {device} device")

    net = FCNForSeq2Seq(20, 3).to(device)
    # 保存模型
    # net = FCNForSeq2Seq(20, 3).to(device)
    net.load_state_dict(torch.load("model_15.pth"))
    test_dataset = ProteinTestDataset("D:/PROJ/DL/kaggle/test")
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    predict_and_save_to_csv(net, device, test_dataloader, "predictions_15.csv")





