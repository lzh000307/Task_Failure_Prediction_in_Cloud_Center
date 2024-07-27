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
    dfj = get_df(DATA_DIR + 'pai_job_table.csv')
    dfm = get_df(DATA_DIR + 'pai_machine_metric.csv')

    # We only need failed and terminated instances
    dfi = dfi[dfi['status'].isin(['Failed', 'Terminated'])]

    # Select needed columns
    dfi_n = ['job_name', 'task_name', 'inst_name', 'status', 'inst_id', 'worker_name',
             'machine', 'start_time', 'end_time']
    dft_n = ['job_name', 'task_name', 'inst_num', 'gpu_type', 'start_time', 'end_time']
    dfs_n = ['job_name', 'worker_name', 'cpu_usage', 'gpu_wrk_util', 'avg_mem', 'max_mem',
             'avg_gpu_wrk_mem', 'max_gpu_wrk_mem', 'read', 'write']
    dfg_n = ['inst_id', 'group', 'workload']
    dfj_n = ['job_name', 'inst_id', 'start_time', 'end_time', 'user']
    dfm_n = ['worker_name', 'machine', 'machine_cpu', 'machine_cpu_kernel', 'machine_gpu',
                'machine_load_1', 'machine_num_worker']

    time_n = ['start_time', 'end_time']

    # Select columns
    dfi = dfi[dfi_n]
    dft = dft[dft_n]
    dfs = dfs[dfs_n]
    dfg = dfg[dfg_n]
    dfj = dfj[dfj_n]
    dfm = dfm[dfm_n]


    # change the column names
    dfi.columns = ['job_name', 'task_name', 'inst_name', 'status', 'inst_id',
                   'worker_name', 'machine', 'start_time', 'end_time']
    dft.columns = ['job_name', 'task_name', 'inst_num', 'gpu_type', 'start_time_task',
                   'end_time_task']
    dfj.columns = ['job_name', 'inst_id', 'start_time_job', 'end_time_job', 'user']



    # Merge instance table with task and sensor tables
    dfi = dfi.merge(dft, on=['job_name', 'task_name'], how='left')
    dfi = dfi.merge(dfs, on=['job_name', 'worker_name'], how='left')
    dfi = dfi.merge(dfj, on=['job_name', 'inst_id'], how='left')
    dfi = dfi.merge(dfm, on=['worker_name', 'machine'], how='left')
    dfi = dfi.merge(dfg, on='inst_id', how='left')


    # if start_time and start_time_task and start_time_job are all NaN, drop the row
    dfi.dropna(subset=['start_time', 'start_time_task', 'start_time_job'], how='all', inplace=True)
    # if end_time and end_time_task and end_time_job are all NaN, drop the row
    dfi.dropna(subset=['end_time', 'end_time_task', 'end_time_job'], how='all', inplace=True)
    # drop status as 'terminated'
    dfi = dfi[dfi['status'].isin(['Failed', 'Running', 'Interrupted', 'Waiting'])]
    # drop empty 'machine_cpu' rows
    dfi.dropna(subset=['machine_cpu'], inplace=True)

    # if start_time is NaN, then fill it with start_time_task, if still NaN, then fill it with
    # start_time_job
    dfi['start_time'].fillna(dfi['start_time_task'], inplace=True)
    dfi['start_time'].fillna(dfi['start_time_job'], inplace=True)
    # if end_time is NaN, then fill it with end_time_task
    dfi['end_time'].fillna(dfi['end_time_task'], inplace=True)
    dfi['end_time'].fillna(dfi['end_time_job'], inplace=True)

    # Drop start_time_task, end_time_task, start_time_job, end_time_job
    dfi.drop(columns=['start_time_task', 'end_time_task', 'start_time_job', 'end_time_job'],
             inplace=True)

    # Drop rows with NaN in the selected sensor columns
    dfi.dropna(subset=dfs_n, inplace=True)
    # dfi.dropna(subset=time_n, inplace=True)

    # Convert timestamps
    dfi['start_time'] = pd.to_datetime(dfi['start_time'], unit='s', origin='unix', utc=True).dt.tz_convert('Asia/Shanghai')
    dfi['end_time'] = pd.to_datetime(dfi['end_time'], unit='s', origin='unix', utc=True).dt.tz_convert('Asia/Shanghai')
    dfi['time'] = (dfi['end_time'] - dfi['start_time']).dt.total_seconds()

    # Sort by group and start_time
    dfi = dfi.sort_values(by=['group', 'time'])

    # select 2,000 rows
    dfi = dfi[:2000]

    # Export to CSV
    dfi.to_csv(DATA_DIR + 'machine_no_terminated.csv', index=True)

    print("数据已整合并保存至 'machine_no_terminated.csv'.")


load_data()
