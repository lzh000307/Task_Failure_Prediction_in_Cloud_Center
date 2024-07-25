import torch
import torch.nn as nn

import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # BiLSTM的输出维度是hidden_dim的两倍

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(x.device)  # 2 for bidirection
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(x.device)  # 2 for bidirection

        #x = x.unsqueeze(1)  # Ensure x is (batch_size, sequence_length, input_size)
        out, _ = self.lstm(x, (h0, c0))  # x should now be (batch_size, sequence_length, input_size)
        # out = self.fc(out[:, -1, :])  # Use the output of the last LSTM cell
        out = self.fc(out)  # Apply the fully connected layer to each time step
        return out