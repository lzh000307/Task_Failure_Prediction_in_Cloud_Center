import os.path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from dataset import dataset
from network import BiLSTM
# Assuming 'utils.py' contains necessary custom functions such as get_df, get_dfa, etc.
from utils import *

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data\\')


def load_data():
    # Directory where data files are stored
    # get the absolute path of the current working directory

    # # 加载所有数据表
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'data\\')
    # Load the dataframes
    # dfj = get_df(DATA_DIR + 'pai_job_table.csv')
    dft = get_df(DATA_DIR + 'pai_task_table.csv')
    dfi = get_df(DATA_DIR + 'pai_instance_table.csv')
    dfs = get_df(DATA_DIR + 'pai_sensor_table.csv')
    # dfp = get_df(DATA_DIR + 'pai_machine_spec.csv')
    # dfm = get_df(DATA_DIR + 'pai_machine_metric.csv')
    dfg = get_df(DATA_DIR + 'pai_group_tag_table.csv')

    # We only need failed and terminated instances
    dfi = dfi[dfi['status'].isin(['Failed', 'Terminated'])]

    # Select Needed Columns
    dfi_n = ['job_name', 'task_name', 'inst_name','status', 'inst_id', 'worker_name', 'machine', 'start_time',
             'end_time']
    dft_n = ['job_name', 'task_name', 'inst_num', 'gpu_type']
    # dfj_n = []
    dfs_n = ['job_name', 'worker_name', 'cpu_usage', 'gpu_wrk_util', 'avg_mem', 'max_mem', 'avg_gpu_wrk_mem',
             'max_gpu_wrk_mem', 'read', 'write']
    # dfp_n = []
    # dfm_n = []
    dfg_n = ['inst_id', 'group', 'workload']

    dfi = dfi[dfi_n]
    dft = dft[dft_n]
    # dfj = dfj[dfj_n]
    dfs = dfs[dfs_n]
    # dfp = dfp[dfp_n]
    # dfm = dfm[dfm_n]
    dfg = dfg[dfg_n]



    # # 合并实例表与任务和作业表
    # dfi = dfi.merge(dft, on=['job_name', 'task_name'], how='left', suffixes=('', '_drop'))
    # dfi = dfi.merge(dfj, on='job_name', how='left', suffixes=('', '_drop'))
    dfi = dfi.merge(dft, on=['job_name', 'task_name'], how='left')
    dfi = dfi.merge(dfs, on=['job_name', 'worker_name'], how='left')
    dfi = dfi.merge(dfg, on='inst_id', how='left')

    # drop empty dfs columns
    dfi.dropna(subset=dfs_n, inplace=True)

    # 计算时间差，并转换时间戳
    dfi['start_time'] = pd.to_datetime(dfi['start_time'], unit='s', origin='unix', utc=True).dt.tz_convert(
        'Asia/Shanghai')
    dfi['end_time'] = pd.to_datetime(dfi['end_time'], unit='s', origin='unix', utc=True).dt.tz_convert('Asia/Shanghai')
    dfi['time'] = (dfi['end_time'] - dfi['start_time']).dt.total_seconds()

    # 删除不需要的时间列和其他列
    columns_to_drop = ['end_time', 'start_time'] + [col for col in dfi.columns if 'time' in col and col != 'time']
    dfi.drop(columns=columns_to_drop, inplace=True)

    # 导出到CSV
    dfi.to_csv(DATA_DIR + 'integrated_instance_data.csv', index=True)

    print("数据已整合并保存至 'integrated_instance_data.csv'.")


load_data()