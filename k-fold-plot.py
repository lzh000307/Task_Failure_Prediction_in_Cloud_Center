import matplotlib.pyplot as plt
import numpy as np

# Data for the 4 folds
folds = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4']

metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
models = ['NCP', 'BiNCP', 'BiLSTM']

# Data for each fold
data = {
    'Accuracy': {
        'NCP': [0.9915510240, 0.9889299012, 0.9914345368, 0.9922245322],
        'BiNCP': [0.9918158114, 0.9913917204, 0.9914426419, 0.9916019786],
        'BiLSTM': [0.9906597887, 0.9901947781, 0.9892915500, 0.9892988531]
    },
    'Precision': {
        'NCP': [0.9944140182, 0.9941024944, 0.9933405095, 0.9928499813],
        'BiNCP': [0.9950945818, 0.9957444023, 0.9950805747, 0.9955914478],
        'BiLSTM': [0.9942586274, 0.9956660897, 0.9955523807, 0.9954669875]
    },
    'Recall': {
        'NCP': [0.9959556068, 0.9948354735, 0.9974250718, 0.9982253648],
        'BiNCP': [0.9950991471, 0.9952511694, 0.9958336231, 0.9956216454],
        'BiLSTM': [0.9942724256, 0.9931687310, 0.9918559948, 0.9925066573]
    },
    'F1 Score': {
        'NCP': [0.9948185913, 0.9932201241, 0.9947648025, 0.9952558038],
        'BiNCP': [0.9949758605, 0.9947352697, 0.9947614157, 0.9948645725],
        'BiLSTM': [0.9942655264, 0.9939919404, 0.9934234760, 0.9934445892]
    }
}

colors = {
    'NCP': 'steelblue',
    'BiNCP': 'orange',
    'BiLSTM': 'green'
}

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

for i, metric in enumerate(metrics):
    ax = axs[i]
    x = np.arange(len(folds))
    width = 0.2

    # Find the minimum value for this metric to adjust the y-axis starting point
    all_values = np.concatenate([data[metric][model] for model in models])
    y_min = min(all_values) - 0.002  # Start y-axis a bit below the minimum value

    ax.bar(x - width, data[metric]['NCP'], width, label='NCP', color=colors['NCP'])
    ax.bar(x, data[metric]['BiNCP'], width, label='BiNCP', color=colors['BiNCP'])
    ax.bar(x + width, data[metric]['BiLSTM'], width, label='BiLSTM', color=colors['BiLSTM'])

    ax.set_xlabel('Fold')
    ax.set_ylabel('Percentage')
    ax.set_title(metric)
    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.set_ylim([y_min, 1.0])  # Set y-axis limits to make differences more visible
    ax.legend()


plt.tight_layout()
plt.show()
