import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve


def plot_model_comparisons(results, metric_names, model_names):
    epochs = results.shape[2]
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))  # Adjust for five validation metrics

    for i, ax in enumerate(axes.flatten()):
        if i >= len(metric_names):  # We have more subplots than metrics
            ax.axis('off')  # Hide unused subplot
            continue
        for j in range(results.shape[0]):
            ax.plot(range(epochs), results[j, i, :], label=f"{model_names[j]}")
        ax.set_title(metric_names[i])
        ax.set_xlabel('Epochs')
        ax.set_ylabel(metric_names[i])
        ax.legend()

    plt.tight_layout()
    plt.show()

def plot_model_comparisons_bar(results, metric_names, model_names):
    # model_names = ["RNN", "LSTM", "BiLSTM"]
    # model_names = ["NCP", "BiNCP", "RNN", "LSTM", "BiLSTM"]
    epochs = results.shape[2]
    num_metrics = len(metric_names)
    num_models = len(model_names)

    fig, axes = plt.subplots(nrows=num_metrics, figsize=(10, 2 * num_metrics), sharex=True)
    bar_width = 0.15  # Reduce the bar width to prevent overlap
    index = np.arange(epochs)

    for metric_idx, ax in enumerate(axes):
        min_val = np.min(results[:, metric_idx, :])  # Find the minimum value for this metric
        max_val = np.max(results[:, metric_idx, :])  # Find the maximum value for this metric
        range_val = max_val - min_val
        margin = 0.1 * range_val  # 10% margin on each side

        for model_idx, model_name in enumerate(model_names):
            metric_data = results[model_idx, metric_idx, :]
            pos = index + model_idx * (bar_width + 0.05)  # Adjust position to add space between bars
            ax.bar(pos, metric_data, bar_width, label=model_name, alpha=0.7)

        ax.set_ylim(min_val - margin, max_val + margin)  # Set y-axis limits with margin
        ax.set_ylabel(metric_names[metric_idx])
        ax.set_title(f'Epoch Comparison for {metric_names[metric_idx]}')
        ax.set_xticks(index + bar_width / 2 * (num_models - 1))
        ax.set_xticklabels([f'Epoch {i+1}' for i in range(epochs)])
        ax.legend()

    plt.tight_layout()
    plt.show()

def plot_best_comparisons_bar(results, metric_names, model_names):
    num_metrics = len(metric_names)
    num_models = len(model_names)

    # 设定不同的颜色，确保颜色数量足够
    # colors = list(mcolors.TABLEAU_COLORS)  # 使用Matplotlib的Tableau颜色，或选择其他颜色列表
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown',
              'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    fig, axes = plt.subplots(nrows=num_metrics, figsize=(4, 10), sharex=True)  # 更紧凑的图形

    bar_width = 1  # 将柱子宽度设置为1，让柱子之间没有间隙

    for metric_idx, ax in enumerate(axes):
        max_vals = np.zeros(num_models)

        for model_idx in range(num_models):
            max_vals[model_idx] = np.max(results[model_idx, metric_idx, :])

        x_pos = np.arange(num_models)
        bars = ax.bar(x_pos, max_vals, bar_width, alpha=0.7)

        # 为每个柱子设置不同的颜色
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        ax.set_ylabel(metric_names[metric_idx])
        ax.set_title(f'Maximum {metric_names[metric_idx]} Across All Epochs')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names)
        # move legend outside of the plot`
        ax.legend(bars, model_names, title="Model", loc="center left", bbox_to_anchor=(1, 0.5))
        # 设置Y轴范围为0.97到1
        # ax.set_ylim([0.985, 1])
        # find the min value on the y-axis
        min_val = np.min(max_vals)
        # find the max value on the y-axis
        max_val = np.max(max_vals)
        # set the y-axis limit with margin
        ax.set_ylim(min_val - 0.1 * (max_val - min_val), max_val + 0.1 * (max_val - min_val))
        # empty the x-axis label
        ax.set_xlabel('')



    plt.tight_layout()
    plt.show()


def plot_roc_from_saved_data(filename):
    epoch_labels_outputs = np.load(filename, allow_pickle=True)

    plt.figure()
    for idx, (labels, outputs) in enumerate(epoch_labels_outputs):
        labels = labels.ravel()
        outputs = outputs.ravel()
        print(labels.shape, outputs.shape)
        fpr, tpr, thresholds = roc_curve(labels, outputs)
        plt.plot(fpr, tpr, label=f'Model {idx + 1}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    # save the results
    # np.save('results_ncplstm_no_posweight_3070.npy', results)
    # # epoch_labels_outputs to numpy
    # epoch_labels_outputs = np.array(epoch_labels_outputs)
    # np.save('epoch_labels_outputs_3070.npy', epoch_labels_outputs)

    # 加载保存的结果数据
    results = np.load('results_ncplstm_no_posweight_0805_all.npy')
    # remove 2-4
    # results = np.delete(results, 2, axis=0)
    # results = np.delete(results, 2, axis=0)
    # result2 = np.load('results_ncplstm_no_posweight_0805_rnn_lstm.npy')
    # results = np.vstack((results, result2[:2]))
    # result2 = np.load('results_ncplstm_no_posweight_s2sbilstm.npy')
    # results = np.vstack((results, result2[:1]))
    # results_lstm = np.load('results_lstm.npy')
    # results = np.delete(results, 2, axis=0)
    # results = np.load('results_ncplstm_no_posweight.npy')
    # save
    # np.save('results_ncp.npy', results)
    # combine the results
    # results = np.vstack((results, results_lstm))

    print(results.shape)

    # remove result's to (2, 6, 10)
    print(results.shape)
    model_names = ["S2S_LSTM", "S2S_BiLSTM", "NCP", "BiNCP", "BiLSTM", "RNN", "LSTM"]
    # model_names = ["S2S_LSTM", "S2S_BiLSTM", "BiLSTM", "RNN", "LSTM"]
    # model_names = ["NCP", "BiNCP", "RNN", "LSTM", "BiLSTM"]

    # Plot the model comparisons for validation metrics
    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    # plot_model_comparisons_bar(results[:, 1:, 2:], metric_names, model_names)  # Skip train_losses for plotting
    plot_best_comparisons_bar(results[:, 2:, 1:], metric_names, model_names)  # Skip train_losses for plotting
    # plot_roc_from_saved_data('epoch_labels_outputs_3070.npy')
