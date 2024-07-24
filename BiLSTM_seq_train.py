import os.path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import category_encoders as ce

from dataset import dataset
from network import BiLSTM
# Assuming 'utils.py' contains necessary custom functions such as get_df, get_dfa, etc.
from utils import *
from scipy.sparse import hstack, csr_matrix


# def load_data():
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import category_encoders as ce
import torch

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import category_encoders as ce
import torch


def load_data():
    pd.set_option('display.max_columns', None)
    DATA_DIR = 'data/'  # Adjust to your actual data directory
    df = pd.read_csv(DATA_DIR + 'integrated_instance_data.csv')

    # Sort by 'group' and 'time'
    df.sort_values(by=['group', 'time'], inplace=True)

    # Split the dataset into train and test sets
    train_df = df.iloc[10000:]
    test_df = df.iloc[:10000]

    # Group by 'group' for the training set and pad sequences
    grouped = train_df.groupby('group', group_keys=False)

    # Define a function to pad sequences
    def pad_sequence(group, size=20, pad_value=0):
        arr = group.to_numpy()
        status_value = arr[-1, df.columns.get_loc('status')]  # Get the last status
        if len(arr) < size:
            padding_size = size - len(arr)
            padding = np.full((padding_size, arr.shape[1]), pad_value, dtype=arr.dtype)
            padding[:, df.columns.get_loc('status')] = status_value  # Set all padded 'status' to the last real 'status'
            arr = np.vstack([arr, padding])
        return arr[:size]  # Return only the first 'size' elements

    # Apply padding function to each group and reset index for training set
    train_df_padded = np.array([pad_sequence(group, size=20, pad_value=0) for _, group in grouped])

    # For the test set, we take sequences as they are (length 1)
    test_df_sequences = np.array([group.values[:1] for _, group in test_df.groupby('group', group_keys=False)])

    # Feature engineering for training set
    numeric_features_train = train_df_padded[
                             :, :, [df.columns.get_loc(col) for col in
                                    ['cpu_usage', 'gpu_wrk_util', 'avg_mem', 'max_mem', 'avg_gpu_wrk_mem',
                                     'max_gpu_wrk_mem', 'read', 'write', 'time']]
                             ]
    scaler = StandardScaler()
    numeric_scaled_train = scaler.fit_transform(
        numeric_features_train.reshape(-1, numeric_features_train.shape[-1])).reshape(numeric_features_train.shape)

    # Feature engineering for test set
    numeric_features_test = test_df_sequences[
                            :, :, [test_df.columns.get_loc(col) for col in
                                   ['cpu_usage', 'gpu_wrk_util', 'avg_mem', 'max_mem', 'avg_gpu_wrk_mem',
                                    'max_gpu_wrk_mem', 'read', 'write', 'time']]
                            ]
    numeric_scaled_test = scaler.transform(numeric_features_test.reshape(-1, numeric_features_test.shape[-1])).reshape(
        numeric_features_test.shape)

    # Encode categorical features for training set
    categorical_features_train = train_df_padded[:, :,
                                 [df.columns.get_loc(col) for col in ['job_name', 'machine', 'gpu_type', 'group']]]
    encoder = ce.HashingEncoder(cols=[0, 1, 2, 3], n_components=100)
    categorical_encoded_train = encoder.fit_transform(
        categorical_features_train.reshape(-1, categorical_features_train.shape[-1])).values.reshape(
        categorical_features_train.shape[0], categorical_features_train.shape[1], -1)

    # Encode categorical features for test set
    categorical_features_test = test_df_sequences[:, :,
                                [test_df.columns.get_loc(col) for col in ['job_name', 'machine', 'gpu_type', 'group']]]
    categorical_encoded_test = encoder.transform(
        categorical_features_test.reshape(-1, categorical_features_test.shape[-1])).values.reshape(
        categorical_features_test.shape[0], categorical_features_test.shape[1], -1)

    # Combine features for training set
    X_train = np.concatenate((numeric_scaled_train, categorical_encoded_train), axis=-1)
    y_train = train_df_padded[:, :, df.columns.get_loc('status')].reshape(-1)
    y_train = np.where(y_train == 'Terminated', 1, 0)

    # Combine features for test set
    X_test = np.concatenate((numeric_scaled_test, categorical_encoded_test), axis=-1)
    y_test = test_df_sequences[:, :, test_df.columns.get_loc('status')].reshape(-1)
    y_test = np.where(y_test == 'Terminated', 1, 0)

    # shape
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    return X_train, X_test, y_train, y_test

def train_and_validate(model, train_loader, val_loader, device, lr=0.00001, num_epochs=4, batch_size=512):
    # dataloader
    train_dataloader = DataLoader(train_loader, batch_size=batch_size, shuffle=True)
    labels = train_dataloader.dataset.tensors[1]
    positive_data = labels.sum()
    negative_data = len(labels) - positive_data
    pos_weight = negative_data / positive_data
    # pos_weight = 0.21274018287658691
    pos_weight = 0.26
    # to tensor
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
    print("len(labels): ", len(labels))
    print("labels.sum(): ", labels.sum())
    print(f"Positive weight: {pos_weight}")
    print(f"Positive rate: {positive_data / len(labels)}")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    # Loss and optimizer
    # criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)


    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        for i, (features, labels) in enumerate(train_dataloader):
            features, labels = features.to(device), labels.to(device)
            # print(f"Batch {i + 1} - Features shape: {features.shape}, Labels shape: {labels.shape}")
            outputs = model(features)
            # print(f"Outputs shape before squeeze: {outputs.shape}")
            outputs = outputs.squeeze()
            # print(outputs.shape, labels.shape)
            # print(f"Outputs shape after squeeze: {outputs.shape}")

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10000 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}')

        # Validation step
        model.eval()  # Set model to evaluation mode
        val_dataloader = DataLoader(val_loader, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            val_loss = 0
            correct = 0
            total = 0
            labels_positive_rate = 0
            predicted_positive_rate = 0
            for features, labels in val_dataloader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                # print(outputs.shape, labels.shape)
                outputs = outputs.squeeze()
                # print(outputs.shape, labels.shape)

                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Calculate accuracy for validation
                predicted = (outputs > 0).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Calculate positive rate
                labels_positive_rate += labels.sum().item()
                predicted_positive_rate += predicted.sum().item()

            val_loss /= len(val_dataloader)
            val_accuracy = correct / total
            print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
            print(f'Labels Positive Rate: {labels_positive_rate / total}, Predicted Positive Rate: {predicted_positive_rate / total}')



if __name__ == '__main__':
    X_train, X_test, y_train, y_test  = load_data()
    # using gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print(f"Using {device} device")

    model = BiLSTM(input_dim=X_train.shape[1], hidden_dim=64, output_dim=1, num_layers=1).to(device)
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    # train_dataset = dataset(X_train, y_train)
    # test_dataset = dataset(X_test, y_test)
    train_and_validate(model, train_dataset, test_dataset, device, lr=0.008, num_epochs=20, batch_size=4096)
    # train(model, train_dataset, device)
    # test(model, test_dataset, device)