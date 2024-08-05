import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import category_encoders as ce


def pad_or_truncate_features(series, group_df, length, feature_dim):
    if len(series) >= length:
        valid_series = series[:length]
        valid_indices = group_df.index[:length]
        return valid_series, valid_indices
    else:
        padding = np.zeros((length - len(series), feature_dim))
        valid_series = np.vstack([series, padding])
        valid_indices = group_df.index
        return valid_series, valid_indices


def pad_or_truncate_labels(series, length):
    if len(series) >= length:
        return series[:length]
    else:
        return np.pad(series, (0, length - len(series)), 'constant')


def load_data(group_size=10):
    pd.set_option('display.max_columns', None)
    DATA_DIR = 'data/'  # Adjust to your actual data directory
    df = pd.read_csv(DATA_DIR + 'start_time_seq_job.csv')

    # Sort by 'group' and 'time'
    df.sort_values(by=['group', 'time'], inplace=True)

    select = ['job_name', 'task_name', 'machine', 'gpu_type', 'group', 'cpu_usage', 'gpu_wrk_util', 'avg_mem',
              'max_mem',
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
    encoder = ce.HashingEncoder(cols=list(categorical_features.columns), n_components=40)
    categorical_encoded = encoder.fit_transform(categorical_features)

    # Combine numeric and categorical features
    features = np.hstack([numeric_scaled, categorical_encoded])

    df['status'] = (df['status'].values == 'Terminated').astype(int)

    group_data = []
    group_labels = []

    feature_dim = features.shape[1]

    # Initialize counters
    job_set = set()
    task_set = set()
    group_count = df['group'].nunique()
    instance_count = 0

    # Group by 'group' and process each group's data into time series
    for group, group_df in df.groupby('group'):
        group_features = features[group_df.index]
        group_target = df.loc[group_df.index, 'status'].values

        series_features, valid_indices = pad_or_truncate_features(group_features, group_df, group_size, feature_dim)
        series_target = pad_or_truncate_labels(group_target, group_size)

        # Update counts using the valid instance count
        instance_count += len(valid_indices)
        job_set.update(df.loc[valid_indices, 'job_name'])
        task_set.update(df.loc[valid_indices, 'task_name'])

        group_data.append(series_features)
        group_labels.append(series_target)

    X = np.array(group_data)
    y = np.array(group_labels)

    # Print counts
    job_count = len(job_set)
    task_count = len(task_set)

    print("Job count:", job_count)
    print("Task count:", task_count)
    print("Group count:", group_count)
    print("Instance count:", instance_count)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")        # Should print (number_of_groups, 10, feature_dimension)
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")        # Should print (number_of_groups, 10)

    # write back
    np.save('X_train.npy', X_train)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    load_data(group_size=50)
