import os.path
import pickle
from datetime import time, datetime

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve
from torch.utils.data import Dataset, DataLoader, TensorDataset

from s2s_lstm import s2s_lstm
from lstm import LSTM
from network import BiLSTM
from RNN import RNN
from bi_ncp_model import BiNCPModel
from ncp_model import NCPModel


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import category_encoders as ce

from s2s_bi_ncp import s2s_bi_ncp
from s2s_bi_lstm import s2s_bi_lstm
from s2s_ncp import s2s_ncp

def train_and_validate(model, train_loader, val_loader, device, lr=0.00001, num_epochs=4, batch_size=1024, confidence=0.8):
    # dataloader
    train_dataloader = DataLoader(train_loader, batch_size=batch_size, shuffle=True)
    # labels = train_dataloader.dataset.tensors[1]
    # features = train_dataloader.dataset.tensors[0]
    # positive_data = labels.sum()
    # negative_data = len(labels) - positive_data
    # pos_weight = negative_data / positive_data
    # # pos_weight = 0.21274018287658691
    # pos_weight = 0.32
    # pos_weight = 0.35
    # # to tensor
    # pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
    # print("len(labels): ", len(labels))
    # print("labels.sum(): ", labels.sum())
    # print(f"Positive weight: {pos_weight}")
    # print(f"Positive rate: {positive_data / len(labels)}")

    # positive data
    # for i in range(len(labels)):
    #     non_zero_len = (features[i, :].sum(dim=1) != 0).sum().item()



    # Loss and optimizer
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    criterion = nn.BCEWithLogitsLoss()

    # Loss and optimizer
    # criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    train_losses = []
    val_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1_scores = []
    roc_data = []


    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        for i, (features, labels) in enumerate(train_dataloader):
            features, labels = features.to(device), labels.to(device)
            # print(f"Batch {i + 1} - Features shape: {features.shape}, Labels shape: {labels.shape}")
            outputs = model(features)
            # print(f"Outputs shape before squeeze: {outputs.shape}")
            # print(outputs.shape, labels.shape)


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
            epoch_labels = []
            epoch_outputs = []
            for features, labels in val_dataloader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                # print(outputs.shape, labels.shape)
                outputs = outputs.squeeze(-1)
                # print(outputs.shape, labels.shape)

                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Calculate accuracy for validation
                outputs = torch.sigmoid(outputs)
                predicted = (outputs > confidence).float()
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

                if epoch == num_epochs - 1:
                    epoch_labels.append(labels.cpu().numpy())
                    epoch_outputs.append(outputs.cpu().numpy())

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

            print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, ', end="")
            print(f'Labels Positive Rate: {labels_positive_rate / total}, Predicted Positive Rate: {predicted_positive_rate / total}, ', end="")
            print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}, Total: {total}')
        # scheduler.step()


    model_name = 'model_' + str(int(datetime.now().timestamp())) + '.pt'
    torch.save(model.state_dict(), model_name)

    return train_losses, val_losses, val_accuracies, val_precisions, val_recalls, val_f1_scores, epoch_labels, epoch_outputs

def plot_model_comparisons(results, metric_names, model_names):
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

def create_models(input_dim):
    hidden_dim = 32
    return [
        # s2s_lstm(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=1, num_layers=1),
        # s2s_bi_lstm(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=1, num_layers=1),
        NCPModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=1, num_layers=1),
        BiNCPModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=1, num_layers=1),
        BiLSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=1, num_layers=1)
        # RNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=1, num_layers=1),
        # LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=1, num_layers=1)
    ]



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")
    num_epochs = 5
    num_metrics = 6  # Including train_losses and five validation metrics
    results = []
    n_folds = 4

    for fold in range(n_folds):
        X_train = np.load(f'X_train_fold{fold + 1}.npy')
        X_test = np.load(f'X_test_fold{fold + 1}.npy')
        y_train = np.load(f'y_train_fold{fold + 1}.npy')
        y_test = np.load(f'y_test_fold{fold + 1}.npy')

        # to tensor
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        # 使用gpu
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 打印数据维度信息
        print(f"Fold {fold + 1}, Using {device} device")
        print(f"input_dim: {X_train.shape[2]}, sequence_length: {X_train.shape[1]}")

        models = create_models(X_train.shape[2])
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        fold_results = np.zeros((len(models), num_metrics, num_epochs))

        for idx, model in enumerate(models):
            model.to(device)
            metrics = train_and_validate(model, train_dataset, test_dataset, device, lr=0.0015, num_epochs=num_epochs,
                                         batch_size=2048, confidence=0.7)
            for metric_idx, metric in enumerate(metrics[:-2]):
                fold_results[idx, metric_idx, :] = metric

        results.append(fold_results)
        np.save(f'results_fold{fold + 1}.npy', fold_results)

    # save the results
    np.save('results_k_fold.npy', np.array(results))  # 多个fold的结果保存到一个文件中