import os.path
from utils import *

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data\\')


def load_data():
    # Existing code to load and preprocess data
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'data\\')
    dft = get_df(DATA_DIR + 'pai_task_table.csv')
    dfi = get_df(DATA_DIR + 'pai_instance_table.csv')
    dfs = get_df(DATA_DIR + 'pai_sensor_table.csv')
    dfg = get_df(DATA_DIR + 'pai_group_tag_table.csv')
    dfm = get_df(DATA_DIR + 'pai_machine_metric.csv')

    # dfi = dfi[dfi['status'].isin(['Failed', 'Terminated'])]

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


    dfi.dropna(subset=dfs_n, inplace=True)
    # dfi.dropna(subset=dfm_n, inplace=True)
    dfi.dropna(subset=time_n, inplace=True)

    dfi['start_time'] = pd.to_datetime(dfi['start_time'], unit='s', origin='unix', utc=True).dt.tz_convert('Asia/Shanghai')
    dfi['end_time'] = pd.to_datetime(dfi['end_time'], unit='s', origin='unix', utc=True).dt.tz_convert('Asia/Shanghai')
    dfi['time'] = (dfi['end_time'] - dfi['start_time']).dt.total_seconds()

    # select status in ['Failed', 'Terminated']
    dfi = dfi[dfi['status'].isin(['Failed', 'Running', 'Interrupted', 'Waiting'])]

    # columns_to_drop = ['end_time', 'start_time'] + [col for col in dfi.columns if 'time' in col and col != 'time']
    # dfi.drop(columns=columns_to_drop, inplace=True)

    # Randomly sample 5000 rows from the DataFrame if it contains enough rows
    # if len(dfi) >= 5000:
    #     dfi_sampled = dfi.sample(n=5000, random_state=42)  # Using a fixed seed for reproducibility
    # else:
    #     dfi_sampled = dfi  # If not enough rows, take the entire DataFrame

    # Export the sampled data to CSV
    dfi.to_csv(DATA_DIR + 'min_5000_no_Terminated.csv', index=True)

    print("数据已整合并保存至 'min_5000.csv'.")


load_data()