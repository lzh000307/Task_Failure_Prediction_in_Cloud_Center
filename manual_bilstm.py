import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BiLSTM, self).__init__()
        self.forward_lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.backward_lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        # Forward LSTM
        out_forward, _ = self.forward_lstm(x)

        # Backward LSTM
        out_backward, _ = self.backward_lstm(torch.flip(x, [1]))

        # Flip the backward output to match the sequence order
        out_backward = torch.flip(out_backward, [1])

        # Concatenate the forward and backward outputs
        out = torch.cat((out_forward, out_backward), dim=2)

        return out


# Test the BiLSTM
batch_size = 3
seq_length = 5
input_size = 4
hidden_size = 6
num_layers = 1

x = torch.randn(batch_size, seq_length, input_size)

model = BiLSTM(input_size, hidden_size, num_layers)
output = model(x)
print(output.shape)  # Expected shape: (batch_size, seq_length, 2*hidden_size)


class StandardBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(StandardBiLSTM, self).__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        out, _ = self.bilstm(x)
        return out


# Instantiate both models
model_manual = BiLSTM(input_size, hidden_size, num_layers)
model_standard = StandardBiLSTM(input_size, hidden_size, num_layers)

# Copy parameters from the standard BiLSTM to the manual BiLSTM
model_manual.forward_lstm.load_state_dict(model_standard.bilstm.state_dict()['_all_weights'][0][:2])
model_manual.backward_lstm.load_state_dict(model_standard.bilstm.state_dict()['_all_weights'][0][2:])

# Ensure both models produce the same output
output_manual = model_manual(x)
output_standard = model_standard(x)

print(torch.allclose(output_manual, output_standard, atol=1e-6))  # Expected output: True
