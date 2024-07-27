import torch
import torch.nn as nn

import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Set bidirectional to False to use a standard LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=False)
        # Adjust the dimension for the output layer since it's not bidirectional anymore
        self.fc = nn.Linear(hidden_dim, output_dim)  # Output dimension matches the hidden_dim of LSTM

    def forward(self, x):
        batch_size = x.size(0)
        # Initialize hidden state and cell state for unidirectional LSTM
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)

        # No need to change the dimensionality of x here unless required by specific use case
        out, _ = self.lstm(x, (h0, c0))  # x should now be (batch_size, sequence_length, input_size)
        # Use the output of the last LSTM cell for the output layer
        out = self.fc(out)  # Apply the fully connected layer to each time step
        return out