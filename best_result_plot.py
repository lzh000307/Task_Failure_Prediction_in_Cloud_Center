import matplotlib
import numpy as np
from matplotlib import pyplot as plt

models = ['S2S_LSTM', 'S2S_BiLSTM', 'NCP', 'BiNCP', 'BiLSTM', 'RNN', 'LSTM']

# Data for each model
data = {
    'Accuracy': {
        'S2S_LSTM': 0.9889091609,
        'S2S_BiLSTM': 0.9892549507,
        'NCP': 0.9919862868,
        'BiNCP': 0.9930761303,
        'BiLSTM': 0.9919647590,
        'RNN': 0.9900689696,
        'LSTM': 0.9902815564
    },
    'Precision': {
        'S2S_LSTM': 0.9955969177,
        'S2S_BiLSTM': 0.9953480387,
        'NCP': 0.9941247321,
        'BiNCP': 0.9969759973,
        'BiLSTM': 0.9967105427,
        'RNN': 0.9943955695,
        'LSTM': 0.9952730509
    },
    'Recall': {
        'S2S_LSTM': 0.9923909127,
        'S2S_BiLSTM': 0.9937312624,
        'NCP': 0.9980993809,
        'BiNCP': 0.9964598903,
        'BiLSTM': 0.9946980991,
        'RNN': 0.9938981863,
        'LSTM': 0.9942353396
    },
    'F1 Score': {
        'S2S_LSTM': 0.9931726452,
        'S2S_BiLSTM': 0.9934029005,
        'NCP': 0.9950916322,
        'BiNCP': 0.9957505983,
        'BiLSTM': 0.9950614262,
        'RNN': 0.9939006503,
        'LSTM': 0.9940324477
    }
}

# Find min and max values to adjust the y-axis
all_values = np.array([value for metric_data in data.values() for value in metric_data.values()])
min_value = all_values.min()
max_value = all_values.max()

# Set font size for better readability
plt.rcParams.update({'font.size': 12})

# Create a bar chart for each metric with adjusted y-axis limits
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

for i, metric in enumerate(['Accuracy', 'Precision', 'Recall', 'F1 Score']):
    ax = axs[i]
    x = np.arange(len(models))
    width = 0.6

    ax.bar(x, [data[metric][model] for model in models], width, color=colors)

    ax.set_xlabel('Model')
    ax.set_ylabel('Percentage')
    ax.set_title(metric)
    ax.set_xticks(x)
    # set x-axis labels to model names
    ax.set_xticklabels(models, rotation=45)
    # check the x-axis labels are not cut off

    ax.set_ylim(min_value - 0.002, max_value + 0.002)  # Adjusting y-axis to show better distinction
    print(f"{metric} x-axis labels: {[label.get_text() for label in ax.get_xticklabels()]}")


plt.tight_layout()
plt.show()
print(matplotlib.__version__)