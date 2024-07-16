import os.path
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from dataset import dataset
from network import BiLSTM
from utils import *

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data\\')


def load_data():
    # Directory where data files are stored
    # get the absolute path of the current working directory

    # Load all data tables
    dft = get_df(DATA_DIR + 'pai_task_table.csv')
    dfi = get_df(DATA_DIR + 'pai_instance_table.csv')
    dfs = get_df(DATA_DIR + 'pai_sensor_table.csv')
    dfg = get_df(DATA_DIR + 'pai_group_tag_table.csv')

    # We only need failed and terminated instances
    dfi = dfi[dfi['status'].isin(['Failed', 'Terminated'])]

    # Select needed columns
    dfi_n = ['job_name', 'task_name', 'inst_name', 'status', 'inst_id', 'worker_name', 'machine', 'start_time', 'end_time']
    dft_n = ['job_name', 'task_name', 'inst_num', 'gpu_type']
    dfs_n = ['job_name', 'worker_name', 'cpu_usage', 'gpu_wrk_util', 'avg_mem', 'max_mem', 'avg_gpu_wrk_mem', 'max_gpu_wrk_mem', 'read', 'write']
    dfg_n = ['inst_id', 'group', 'workload']

    dfi = dfi[dfi_n]
    dft = dft[dft_n]
    dfs = dfs[dfs_n]
    dfg = dfg[dfg_n]

    # Merge instance table with task and sensor tables
    dfi = dfi.merge(dft, on=['job_name', 'task_name'], how='left')
    dfi = dfi.merge(dfs, on=['job_name', 'worker_name'], how='left')
    dfi = dfi.merge(dfg, on='inst_id', how='left')

    # Drop rows with NaN in the selected sensor columns
    dfi.dropna(subset=dfs_n, inplace=True)

    # Convert timestamps
    dfi['start_time'] = pd.to_datetime(dfi['start_time'], unit='s', origin='unix', utc=True).dt.tz_convert('Asia/Shanghai')
    dfi['end_time'] = pd.to_datetime(dfi['end_time'], unit='s', origin='unix', utc=True).dt.tz_convert('Asia/Shanghai')
    dfi['time'] = (dfi['end_time'] - dfi['start_time']).dt.total_seconds()

    # Sort by group and start_time
    dfi = dfi.sort_values(by=['group', 'time'])

    # Export to CSV
    dfi.to_csv(DATA_DIR + 'start_time_seq.csv', index=True)

    print("数据已整合并保存至 'start_time_seq.csv'.")


load_data()
