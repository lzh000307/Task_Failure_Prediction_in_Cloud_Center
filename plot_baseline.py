import matplotlib.pyplot as plt

# Epochs
epochs = list(range(1, 11))

# CNN log data
cnn_data = {
    "Validation Loss": [0.2571, 0.2481, 0.2452, 0.2439, 0.2409, 0.2421, 0.2382, 0.2421, 0.2354, 0.2414],
    "Validation Accuracy": [0.8834, 0.8811, 0.8842, 0.8832, 0.8933, 0.8892, 0.8936, 0.9015, 0.8928, 0.9023],
    "Precision": [0.9538, 0.9587, 0.9582, 0.9597, 0.9557, 0.9587, 0.9566, 0.9480, 0.9585, 0.9481],
    "Recall": [0.9024, 0.8945, 0.8989, 0.8961, 0.9130, 0.9048, 0.9125, 0.9318, 0.9095, 0.9328],
    "F1 Score": [0.9274, 0.9255, 0.9276, 0.9268, 0.9339, 0.9310, 0.9340, 0.9398, 0.9334, 0.9404]
}

# Bi-LSTM log data
bilstm_data = {
    "Validation Loss": [0.2983, 0.2786, 0.2670, 0.2605, 0.2558, 0.2520, 0.2485, 0.2456, 0.2425, 0.2397],
    "Validation Accuracy": [0.8412, 0.8573, 0.8661, 0.8751, 0.8752, 0.8770, 0.8813, 0.8852, 0.8851, 0.8872],
    "Precision": [0.9550, 0.9552, 0.9571, 0.9555, 0.9573, 0.9581, 0.9569, 0.9565, 0.9578, 0.9578],
    "Recall": [0.8476, 0.8678, 0.8771, 0.8902, 0.8884, 0.8899, 0.8966, 0.9020, 0.9005, 0.9031],
    "F1 Score": [0.8981, 0.9094, 0.9154, 0.9217, 0.9216, 0.9228, 0.9258, 0.9285, 0.9283, 0.9297]
}

metrics = ["Validation Loss", "Validation Accuracy", "Precision", "Recall", "F1 Score"]
colors = ['blue', 'green']
labels = ['CNN', 'Bi-LSTM']

# Rearrange the subplots to be in a 2x3 grid with the last subplot occupying two columns
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Flatten the axs array for easier iteration
axs = axs.flatten()

# Plotting the same data with the new layout
for i, metric in enumerate(metrics):
    axs[i].plot(epochs, cnn_data[metric], marker='o', color=colors[0], label=labels[0])
    axs[i].plot(epochs, bilstm_data[metric], marker='o', color=colors[1], label=labels[1])
    axs[i].set_title(metric)
    axs[i].set_xlabel('Epoch')
    axs[i].set_ylabel(metric)
    axs[i].legend()
    axs[i].grid(True)

# Hide the last subplot (the 6th subplot) as we only have 5 metrics
fig.delaxes(axs[-1])


plt.tight_layout()
plt.show()