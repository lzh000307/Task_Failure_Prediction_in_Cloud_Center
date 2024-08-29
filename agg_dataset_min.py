import os.path
from utils import *

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data\\')


def load_data():
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'data\\')
    dft = get_df(DATA_DIR + 'pai_task_table.csv')
    dfi = get_df(DATA_DIR + 'pai_instance_table.csv')
    dfs = get_df(DATA_DIR + 'pai_sensor_table.csv')
    dfg = get_df(DATA_DIR + 'pai_group_tag_table.csv')
    dfm = get_df(DATA_DIR + 'pai_machine_metric.csv')

    dfi_n = ['job_name', 'task_name', 'inst_name','status', 'inst_id', 'worker_name', 'machine', 'start_time', 'end_time']
    dft_n = ['job_name', 'task_name', 'inst_num', 'gpu_type']
    dfs_n = ['job_name', 'worker_name', 'cpu_usage', 'gpu_wrk_util', 'avg_mem', 'max_mem', 'avg_gpu_wrk_mem', 'max_gpu_wrk_mem', 'read', 'write']
    dfg_n = ['inst_id', 'group', 'workload']
    dfm_n = ['worker_name', 'machine', 'machine_cpu', 'machine_cpu_kernel', 'machine_gpu', 'machine_load_1', 'machine_num_worker']
    time_n = ['start_time', 'end_time']

    dfi = dfi[dfi_n]
    dft = dft[dft_n]
    dfs = dfs[dfs_n]
    dfg = dfg[dfg_n]
    dfm = dfm[dfm_n]

    dfi = dfi.merge(dft, on=['job_name', 'task_name'], how='left')
    dfi = dfi.merge(dfs, on=['job_name', 'worker_name'], how='left')
    dfi = dfi.merge(dfm, on=['worker_name', 'machine'], how='left')
    dfi = dfi.merge(dfg, on='inst_id', how='left')

    dfi['dfs_filled'] = ~dfi[dfs_n].isna().any(axis=1)
    dfi['dfm_filled'] = ~dfi[dfm_n].isna().any(axis=1)

    # Filter the dataframe for the different conditions
    both_filled = dfi[dfi['dfs_filled'] & dfi['dfm_filled']]
    only_dfs_filled = dfi[dfi['dfs_filled']]
    only_dfm_filled = dfi[dfi['dfm_filled']]

    # Count status in the specified conditions
    status_counts_both = both_filled['status'].value_counts()
    status_counts_dfs = only_dfs_filled['status'].value_counts()
    status_counts_dfm = only_dfm_filled['status'].value_counts()

    print("When both dfs and dfm are not empty, the number of states:")
    print(status_counts_both)
    print("\nWhen dfs is not empty (regardless of dfm), the number of states:")
    print(status_counts_dfs)
    print("\nWhen dfm is not empty (regardless of dfs), the number of states:")
    print(status_counts_dfm)

load_data()