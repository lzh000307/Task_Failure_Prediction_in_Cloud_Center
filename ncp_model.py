import torch
import torch.nn as nn
from ncps.torch import CfC
from ncps.wirings import AutoNCP

class NCPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(NCPModel, self).__init__()
        wiring = AutoNCP(hidden_dim, output_dim)
        self.rnn = CfC(input_dim, wiring)
        self.fc = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(batch_size, self.rnn.state_size).to(x.device)
        output, _ = self.rnn(x, h0)
        output = self.fc(output)
        return output

