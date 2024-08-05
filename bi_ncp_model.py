import torch
import torch.nn as nn
from ncps.torch import CfC
from ncps.wirings import AutoNCP


class BiNCPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(BiNCPModel, self).__init__()
        wiring = AutoNCP(hidden_dim, output_dim)
        self.rnn_forward = CfC(input_dim, wiring)
        self.rnn_backward = CfC(input_dim, wiring)
        self.fc = nn.Linear(output_dim * 2, output_dim)

    def forward(self, x):
        batch_size = x.size(0)

        # Forward direction
        h0_forward = torch.zeros(batch_size, self.rnn_forward.state_size).to(x.device)
        output_forward, _ = self.rnn_forward(x, h0_forward)

        # Backward direction
        h0_backward = torch.zeros(batch_size, self.rnn_backward.state_size).to(x.device)
        x_reversed = torch.flip(x, dims=[1])
        output_backward, _ = self.rnn_backward(x_reversed, h0_backward)
        output_backward = torch.flip(output_backward, dims=[1])

        # Concatenate forward and backward outputs
        output = torch.cat((output_forward, output_backward), dim=-1)
        output = self.fc(output)
        return output

