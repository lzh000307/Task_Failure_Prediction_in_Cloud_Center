import numpy as np
import torch
from torch.utils.data import Dataset


class ProteinDataset(Dataset):
    def __init__(self, data, name_labels, data_labels, indices=None):
        super().__init__()
        self.data = data
        self.name_labels = name_labels
        self.data_labels = data_labels
        self.indices = indices if indices is not None else list(range(len(name_labels)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        item = self.indices[idx]
        protein_data = np.loadtxt(
            open(f"D:/PROJ/DL/kaggle/train/{self.data[item]}", "rb"),
            delimiter=",", skiprows=1, usecols=range(2, 22)
        )
        # print(self.data[item])
        # 获取标签
        file_name = self.data[item].replace('_train.csv', '')
        label = self.data_labels[self.name_labels.index(file_name)]
        # print("protein_data.shape: ", protein_data.shape, "label: ", len(label))

        # 通过one-hot编码转换标签
        st = ['C', 'E', 'H']
        label_encoded = np.array([st.index(x) for x in label])

        # 调整数据形状为[channel, length]
        protein_data = np.transpose(protein_data)  # 将形状从[length, channel]转换为[channel, length]

        # 转换为Tensor
        protein_data = torch.tensor(protein_data, dtype=torch.float32)
        label_encoded = torch.tensor(label_encoded, dtype=torch.long)  # 使用long类型，适合分类任务

        return protein_data, label_encoded
        # return {'data': protein_data, 'label': label_encoded}