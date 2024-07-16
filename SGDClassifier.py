import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
import category_encoders as ce
import numpy as np
import os


def load_and_preprocess_data():
    pd.set_option('display.max_colwidth', 1000)
    pd.set_option('display.max_columns', 100)

    DATA_DIR = os.path.join(os.path.dirname(__file__), 'data\\')
    df = pd.read_csv(DATA_DIR + 'integrated_instance_data.csv')
    df = df.drop('Unnamed: 0', axis=1)
    y = df['status'].apply(lambda x: 1 if x == 'Terminated' else 0).values

    numeric_features = df[
        ['cpu_usage', 'gpu_wrk_util', 'avg_mem', 'max_mem', 'avg_gpu_wrk_mem', 'max_gpu_wrk_mem', 'read', 'write']]
    categorical_features = df[['job_name', 'machine', 'gpu_type', 'group']]

    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(numeric_features)

    encoder = ce.HashingEncoder(cols=['job_name', 'machine', 'gpu_type', 'group'], n_components=100)
    categorical_encoded = encoder.fit_transform(categorical_features)

    X = np.hstack((numeric_scaled, categorical_encoded))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def train_and_evaluate_sgd(X_train, X_test, y_train, y_test):
    # Calculate class weights
    pos_weight = 1 / 0.8245789408683777
    neg_weight = 1 / 0.1754210591316223
    class_weight = {0: neg_weight, 1: pos_weight}

    sgd = SGDClassifier(loss='hinge', class_weight=class_weight, max_iter=1000, tol=1e-3, n_jobs=-1)
    sgd.fit(X_train, y_train)

    y_pred_train = sgd.predict(X_train)
    y_pred_test = sgd.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    print("Training Accuracy: ", train_accuracy)
    print("Test Accuracy: ", test_accuracy)
    print("\nClassification Report on Test Data:\n", classification_report(y_test, y_pred_test))


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    train_and_evaluate_sgd(X_train, X_test, y_train, y_test)
