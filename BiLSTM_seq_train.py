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
def load_data():
    pd.set_option('display.max_columns', None)
    DATA_DIR = 'data/'  # Adjust to your actual data directory
    df = pd.read_csv(DATA_DIR + 'integrated_instance_data.csv')

    # Sort by 'group' and 'time'
    df.sort_values(by=['group', 'time'], inplace=True)

    # Group by 'group' and pad sequences
    grouped = df.groupby('group')

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

    # Apply padding function to each group
    df_padded = np.vstack(grouped.apply(lambda g: pad_sequence(g, size=20, pad_value=0)))

    # Create a DataFrame from the padded data
    df_padded = pd.DataFrame(df_padded, columns=df.columns)

    # Feature engineering
    numeric_features = df_padded[
        ['cpu_usage', 'gpu_wrk_util', 'avg_mem', 'max_mem', 'avg_gpu_wrk_mem', 'max_gpu_wrk_mem', 'read', 'write',
         'time']]
    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(numeric_features)

    # Encode categorical features
    categorical_features = df_padded[['job_name', 'machine', 'gpu_type', 'group']]
    encoder = ce.HashingEncoder(cols=['job_name', 'machine', 'gpu_type', 'group'], n_components=100)
    categorical_encoded = encoder.fit_transform(categorical_features)

    # Combine features
    X = np.hstack((numeric_scaled, categorical_encoded))
    y = df_padded['status'].apply(lambda x: 1 if x == 'Terminated' else 0).values

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)
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