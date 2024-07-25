import os.path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from network import BiLSTM

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import category_encoders as ce


def pad_or_truncate_features(series, length, feature_dim):
    if len(series) >= length:
        return series[:length]
    else:
        padding = np.zeros((length - len(series), feature_dim))
        return np.vstack([series, padding])

def pad_or_truncate_labels(series, length):
    if len(series) >= length:
        return series[:length]
    else:
        return np.pad(series, (0, length - len(series)), 'constant')

def load_data():
    pd.set_option('display.max_columns', None)
    DATA_DIR = 'data/'  # Adjust to your actual data directory
    df = pd.read_csv(DATA_DIR + 'start_time_seq_job.csv')

    # Sort by 'group' and 'time'
    df.sort_values(by=['group', 'time'], inplace=True)

    select = ['job_name', 'machine', 'gpu_type', 'group', 'cpu_usage', 'gpu_wrk_util', 'avg_mem', 'max_mem',
              'avg_gpu_wrk_mem', 'max_gpu_wrk_mem', 'read', 'write', 'time', 'status']

    df = df[select]

    # select 1,000 rows for example
    # df = df[:1000]

    df.reset_index(drop=True, inplace=True)

    # Preprocess numeric features: Standardization
    numeric_features = df.select_dtypes(include=[np.number]).drop(['time'], axis=1)
    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(numeric_features)

    # Encode categorical features
    categorical_features = df.select_dtypes(include=[object])
    encoder = ce.HashingEncoder(cols=list(categorical_features.columns), n_components=10)
    categorical_encoded = encoder.fit_transform(categorical_features)

    # Combine numeric and categorical features
    features = np.hstack([numeric_scaled, categorical_encoded])

    df['status'] = (df['status'].values == 'Terminated').astype(int)

    group_data = []
    group_labels = []

    feature_dim = features.shape[1]

    # Group by 'group' and process each group's data into time series
    for group, group_df in df.groupby('group'):
        group_features = features[group_df.index]
        group_target = df.loc[group_df.index, 'status'].values

        series_features = pad_or_truncate_features(group_features, 10, feature_dim)
        series_target = pad_or_truncate_labels(group_target, 10)

        group_data.append(series_features)
        group_labels.append(series_target)

    X = np.array(group_data)
    y = np.array(group_labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")        # Should print (number_of_groups, 10, feature_dimension)
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")        # Should print (number_of_groups, 10)


    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    # shape

    return X_train, X_test, y_train, y_test

def train_and_validate(model, train_loader, val_loader, device, lr=0.00001, num_epochs=4, batch_size=512):
    # dataloader
    train_dataloader = DataLoader(train_loader, batch_size=batch_size, shuffle=True)
    labels = train_dataloader.dataset.tensors[1]
    features = train_dataloader.dataset.tensors[0]
    # positive_data = labels.sum()
    # negative_data = len(labels) - positive_data
    # pos_weight = negative_data / positive_data
    # # pos_weight = 0.21274018287658691
    pos_weight = 0.32
    # # to tensor
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
    # print("len(labels): ", len(labels))
    # print("labels.sum(): ", labels.sum())
    # print(f"Positive weight: {pos_weight}")
    # print(f"Positive rate: {positive_data / len(labels)}")

    # positive data
    # for i in range(len(labels)):
    #     non_zero_len = (features[i, :].sum(dim=1) != 0).sum().item()



    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    # criterion = nn.BCEWithLogitsLoss()

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

            outputs = outputs.squeeze(-1)
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
                outputs = outputs.squeeze(-1)
                # print(outputs.shape, labels.shape)

                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Calculate accuracy for validation
                predicted = (outputs > 0).float()
                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()
                # only check the first time series in labels and outputs
                # correct += (predicted[:, 0] == labels[:, 0]).sum().item()

                # calculate the correct number of the non-zero feature series
                for i in range(labels.size(0)):
                    non_zero_len = (features[i, :].sum(dim=1) != 0).sum().item()
                    total += non_zero_len
                    correct += (predicted[i, :non_zero_len] == labels[i, :non_zero_len]).sum().item()
                    # Calculate positive rate
                    labels_positive_rate += labels[i, :non_zero_len].sum().item()
                    predicted_positive_rate += predicted[i, :non_zero_len].sum().item()

                # Calculate positive rate
                # labels_positive_rate += labels.sum().item()
                # predicted_positive_rate += predicted.sum().item()

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

    print("input_dim: ", X_train.shape[2], X_train.shape[1])


    model = BiLSTM(input_dim=X_train.shape[2], hidden_dim=64, output_dim=1, num_layers=1).to(device)
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    # train_dataset = dataset(X_train, y_train)
    # test_dataset = dataset(X_test, y_test)
    train_and_validate(model, train_dataset, test_dataset, device, lr=0.005, num_epochs=15, batch_size=256)
    # train(model, train_dataset, device)
    # test(model, test_dataset, device)