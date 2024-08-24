import os.path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import category_encoders as ce

from cnn_model import SimpleCNN
from dataset import dataset
from network import BiLSTM
# Assuming 'utils.py' contains necessary custom functions such as get_df, get_dfa, etc.
from utils import *
from scipy.sparse import hstack, csr_matrix


def load_data():
    pd.set_option('display.max_colwidth', 1000)
    pd.set_option('display.max_columns', 100)
    matplotlib.rcParams.update({"font.size": 16, 'lines.linewidth': 2.5})

    DATA_DIR = os.path.join(os.path.dirname(__file__), 'data\\')
    df = pd.read_csv(DATA_DIR + 'integrated_instance_data.csv')
    df = df.drop('Unnamed: 0', axis=1)
    y = df['status'].apply(lambda x: 1 if x == 'Terminated' else 0).values

    numeric_features = df[['cpu_usage', 'gpu_wrk_util', 'avg_mem', 'max_mem', 'avg_gpu_wrk_mem', 'max_gpu_wrk_mem', 'read', 'write']]
    categorical_features = df[['job_name', 'machine', 'gpu_type', 'group']]

    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(numeric_features)

    encoder = ce.HashingEncoder(cols=['job_name', 'machine', 'gpu_type', 'group'], n_components=100)
    categorical_encoded = encoder.fit_transform(categorical_features)

    X = np.hstack((numeric_scaled, categorical_encoded))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Reshape for CNN: (batch_size, input_channels, sequence_length)
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test


def train_and_validate(model, train_loader, val_loader, device, lr=0.0003, num_epochs=10, batch_size=4096, confidence=0.8):
    train_dataloader = DataLoader(train_loader, batch_size=batch_size, shuffle=True)
    # labels = train_dataloader.dataset.tensors[1]
    # pos_weight = (len(labels) - labels.sum()) / labels.sum()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        for i, (features, labels) in enumerate(train_dataloader):
            features, labels = features.to(device), labels.to(device)
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10000 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}')

        model.eval()
        val_dataloader = DataLoader(val_loader, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            val_loss = 0
            correct = 0
            total = 0
            tp = 0
            fp = 0
            fn = 0
            for features, labels in val_dataloader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                outputs = torch.sigmoid(outputs)
                predicted = (outputs > confidence).float()

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Calculate TP, FP, TN, FN
                tp += (predicted * labels).sum().item()
                fp += (predicted * (1 - labels)).sum().item()
                fn += ((1 - predicted) * labels).sum().item()

            val_loss /= len(val_dataloader)
            val_accuracy = correct / total
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            scheduler.step(val_loss)

            print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
            print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}')

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")

    model = SimpleCNN(input_channels=1, num_classes=1).to(device)
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_and_validate(model, train_dataset, test_dataset, device, lr=0.001, num_epochs=10, batch_size=4096, confidence=0.7)
