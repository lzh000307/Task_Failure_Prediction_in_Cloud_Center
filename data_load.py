import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import category_encoders as ce


def pad_or_truncate_features(series, group_df, length, feature_dim):
    num_full_chunks = len(series) // length
    remainder = len(series) % length

    # Initialize lists to hold chunks and indices
    chunks = []
    indices_list = []
    count = len(series)

    # Extract full chunks
    for i in range(num_full_chunks):
        start_idx = i * length
        end_idx = start_idx + length
        chunks.append(series[start_idx:end_idx])
        indices_list.append(group_df.index[start_idx:end_idx])

    # Handle the remainder
    if remainder > 0:
        padding = np.zeros((length - remainder, feature_dim))
        last_chunk = np.vstack([series[-remainder:], padding])
        last_indices = group_df.index[-remainder:]
        chunks.append(last_chunk)
        indices_list.append(last_indices)

    return chunks, indices_list, count


def pad_or_truncate_labels(labels, length):
    num_full_chunks = len(labels) // length
    remainder = len(labels) % length
    label_chunks = []

    # Full chunks
    for i in range(num_full_chunks):
        label_chunks.append(labels[i * length:(i + 1) * length])

    # Padding for remainder
    if remainder > 0:
        padded_labels = np.pad(labels[-remainder:], (0, length - remainder), 'constant')
        label_chunks.append(padded_labels)

    return label_chunks


def load_data(group_size=10):
    """
    Loads and preprocesses data from a CSV file, standardizes numeric features, encodes categorical features,
    and prepares the data for machine learning model training.

    Parameters:
    ----------
    group_size : int, optional
        The number of instances per group after padding or truncating. Default is 10.

    Returns:
    -------
    None
        The function does not return any value but saves preprocessed training and testing datasets for each fold.

    Steps:
    ------
    1. Load the dataset from a specified directory.
    2. Sort the data by 'group' and 'time' columns.
    3. Select specific columns for processing.
    4. Preprocess the numeric features by standardizing them using `StandardScaler`.
    5. Encode the categorical features using `HashingEncoder`.
    6. Combine numeric and categorical features into a single feature set.
    7. Convert the 'status' column into a binary format where 'Terminated' is 1, and others are 0.
    8. Process the data by grouping them based on the 'group' column and ensure each group has a fixed size of `group_size`.
       - Pad or truncate the feature sequences to match the group size.
       - Update instance counts and job names encountered in the dataset.
    9. Print out summary statistics, including the job count, group count, and instance count.
    10. Split the data into training and testing sets in a 70-30 ratio.
    11. Split the data into training and testing sets using K-Folds cross-validation (with 4 splits).
    12. Save the training and testing sets for each fold as `.npy` files.

    Notes:
    ------
    - The function assumes the presence of a CSV file named 'start_time_seq_job.csv' in the 'data' directory.
    - The preprocessing includes handling both numeric and categorical data.
    - Data is split into training and testing sets using KFold cross-validation with 4 splits.
    """
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
    group_count = df['group'].nunique()
    instance_count = 0
    counts = 0


    # Group by 'group' and process
    for group, group_df in df.groupby('group'):
        group_features = features[group_df.index]
        group_target = df.loc[group_df.index, 'status'].values

        series_features_chunks, valid_indices_chunks, count = pad_or_truncate_features(group_features, group_df, group_size,
                                                                                feature_dim)
        series_target_chunks = pad_or_truncate_labels(group_target, group_size)

        counts += count

        for chunk, indices in zip(series_features_chunks, valid_indices_chunks):
            group_data.append(chunk)
            instance_count += len(indices)
            job_set.update(df.loc[indices, 'job_name'])

        for labels in series_target_chunks:
            group_labels.append(labels)

    X = np.array(group_data)
    y = np.array(group_labels)

    # Print counts
    job_count = len(job_set)

    print("Job count:", job_count)
    print("Group count:", group_count)
    print("Instance count:", instance_count)
    print("Instance count 2:", counts)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    #
    # print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")        # Should print (number_of_groups, 10, feature_dimension)
    # print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")        # Should print (number_of_groups, 10)
    #
    # # # write back
    # np.save('X_train2.npy', X_train)
    # np.save('X_test2.npy', X_test)
    # np.save('y_train2.npy', y_train)
    # np.save('y_test2.npy', y_test)

    folds = KFold(n_splits=4, shuffle=True, random_state=42)

    for i, (train_idx, test_idx) in enumerate(folds.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        np.save(f'X_train_fold{i+1}.npy', X_train)
        np.save(f'X_test_fold{i+1}.npy', X_test)
        np.save(f'y_train_fold{i+1}.npy', y_train)
        np.save(f'y_test_fold{i+1}.npy', y_test)

    print("Data saved for each fold.")

if __name__ == '__main__':
    load_data(group_size=10)
