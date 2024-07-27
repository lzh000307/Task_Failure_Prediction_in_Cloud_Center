import os.path
from datetime import time, datetime

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset
from lstm import LSTM
from network import BiLSTM
from RNN import RNN
from bincp import BiNCP


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

    select = ['job_name', 'task_name', 'machine', 'gpu_type', 'group', 'cpu_usage', 'gpu_wrk_util', 'avg_mem', 'max_mem',
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

    train_losses = []
    val_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1_scores = []


    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        for i, (features, labels) in enumerate(train_dataloader):
            features, labels = features.to(device), labels.to(device)
            # print(f"Batch {i + 1} - Features shape: {features.shape}, Labels shape: {labels.shape}")
            outputs = model(features)
            # print(f"Outputs shape before squeeze: {outputs.shape}")

            outputs = outputs.squeeze(-1)
            # outputs = outputs.squeeze(0)
            # print(outputs.shape, labels.shape)
            # print(f"Outputs shape after squeeze: {outputs.shape}")


            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10000 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}')

        train_losses.append(loss.item())


        # Validation step
        model.eval()  # Set model to evaluation mode
        val_dataloader = DataLoader(val_loader, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            val_loss = 0
            correct = 0
            total = 0
            labels_positive_rate = 0
            predicted_positive_rate = 0
            true_positives = 0
            false_positives = 0
            false_negatives = 0
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

                    # Calculate TP, FP, FN
                    true_positives += (
                                (predicted[i, :non_zero_len] == 1) & (labels[i, :non_zero_len] == 1)).sum().item()
                    false_positives += (
                                (predicted[i, :non_zero_len] == 1) & (labels[i, :non_zero_len] == 0)).sum().item()
                    false_negatives += (
                                (predicted[i, :non_zero_len] == 0) & (labels[i, :non_zero_len] == 1)).sum().item()

                # Calculate positive rate
                # labels_positive_rate += labels.sum().item()
                # predicted_positive_rate += predicted.sum().item()

            val_loss /= len(val_dataloader)
            val_accuracy = correct / total
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            val_precisions.append(precision)
            val_recalls.append(recall)
            val_f1_scores.append(f1_score)

            print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
            print(f'Labels Positive Rate: {labels_positive_rate / total}, Predicted Positive Rate: {predicted_positive_rate / total}')
            print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}')

    # # Plot the metrics
    # epochs = range(1, num_epochs + 1)
    #
    # plt.figure(figsize=(12, 10))
    #
    # plt.subplot(2, 2, 1)
    # plt.plot(epochs, train_losses, label='Train Loss')
    # plt.plot(epochs, val_losses, label='Validation Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.title('Loss')
    #
    # plt.subplot(2, 2, 2)
    # plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.title('Accuracy')
    #
    # plt.subplot(2, 2, 3)
    # plt.plot(epochs, val_precisions, label='Validation Precision')
    # plt.xlabel('Epochs')
    # plt.ylabel('Precision')
    # plt.legend()
    # plt.title('Precision')
    #
    # plt.subplot(2, 2, 4)
    # plt.plot(epochs, val_recalls, label='Validation Recall')
    # plt.xlabel('Epochs')
    # plt.ylabel('Recall')
    # plt.legend()
    # plt.title('Recall')
    #
    # plt.figure(figsize=(6, 5))
    # plt.plot(epochs, val_f1_scores, label='Validation F1 Score')
    # plt.xlabel('Epochs')
    # plt.ylabel('F1 Score')
    # plt.legend()
    # plt.title('F1 Score')
    #
    # plt.show()

    # save the model, named by timestamp
    model_name = 'model_' + str(int(datetime.now().timestamp())) + '.pt'
    torch.save(model.state_dict(), model_name)

    return train_losses, val_losses, val_accuracies, val_precisions, val_recalls, val_f1_scores

def plot_model_comparisons(results, metric_names):
    model_names = ["RNN", "LSTM", "BiLSTM"]
    epochs = results.shape[2]
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))  # Adjust for five validation metrics

    for i, ax in enumerate(axes.flatten()):
        if i >= len(metric_names):  # We have more subplots than metrics
            ax.axis('off')  # Hide unused subplot
            continue
        for j in range(results.shape[0]):
            ax.plot(range(epochs), results[j, i, :], label=f"{model_names[j]}")
        ax.set_title(metric_names[i])
        ax.set_xlabel('Epochs')
        ax.set_ylabel(metric_names[i])
        ax.legend()

    plt.tight_layout()
    plt.show()

def plot_model_comparisons_bar(results, metric_names):
    model_names = ["RNN", "LSTM", "BiLSTM"]
    epochs = results.shape[2]
    num_metrics = len(metric_names)
    num_models = len(model_names)

    fig, axes = plt.subplots(nrows=num_metrics, figsize=(10, 2 * num_metrics), sharex=True)
    bar_width = 0.25
    index = np.arange(epochs)

    for metric_idx, ax in enumerate(axes):
        min_val = np.min(results[:, metric_idx, :])  # Find the minimum value for this metric
        max_val = np.max(results[:, metric_idx, :])  # Find the maximum value for this metric
        range_val = max_val - min_val
        margin = 0.1 * range_val  # 10% margin on each side

        for model_idx, model_name in enumerate(model_names):
            metric_data = results[model_idx, metric_idx, :]
            pos = index + model_idx * bar_width
            ax.bar(pos, metric_data, bar_width, label=model_name, alpha=0.7)

        ax.set_ylim(min_val - margin, max_val + margin)  # Set y-axis limits with margin
        ax.set_ylabel(metric_names[metric_idx])
        ax.set_title(f'Epoch Comparison for {metric_names[metric_idx]}')
        ax.set_xticks(index + bar_width / 2 * (num_models-1))
        ax.set_xticklabels([f'Epoch {i+1}' for i in range(epochs)])
        ax.legend()

    plt.tight_layout()
    plt.show()





if __name__ == '__main__':
    # X_train, X_test, y_train, y_test  = load_data()
    # # using gpu
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # # device = torch.device('cpu')
    # print(f"Using {device} device")
    #
    # print("input_dim: ", X_train.shape[2], X_train.shape[1])
    #
    #
    # models = [RNN(input_dim=X_train.shape[2], hidden_dim=64, output_dim=1, num_layers=1),
    #           LSTM(input_dim=X_train.shape[2], hidden_dim=64, output_dim=1, num_layers=1),
    #           BiLSTM(input_dim=X_train.shape[2], hidden_dim=64, output_dim=1, num_layers=1)]
    # train_dataset = TensorDataset(X_train, y_train)
    # test_dataset = TensorDataset(X_test, y_test)
    #
    # num_epochs = 10
    # num_metrics = 6  # Including train_losses and five validation metrics
    # results = np.zeros((3, num_metrics, num_epochs))  # 3 models x 6 metrics x num_epochs
    #
    # for idx, model in enumerate(models):
    #     model.to(device)
    #     metrics = train_and_validate(model, train_dataset, test_dataset, device, lr=0.002, num_epochs=num_epochs, batch_size=1024)
    #     for metric_idx, metric in enumerate(metrics):  # Including train_losses
    #         results[idx, metric_idx, :] = metric
    #
    # # save the results
    # np.save('results.npy', results)

    # 加载保存的结果数据
    results = np.load('results.npy')
    print(results.shape)

    # Plot the model comparisons for validation metrics
    metric_names = ["Validation Loss", "Accuracy", "Precision", "Recall", "F1 Score"]
    plot_model_comparisons_bar(results[:, 1:, :], metric_names)  # Skip train_losses for plotting