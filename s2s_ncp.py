import torch
import torch.nn as nn
from ncps.torch import CfC
from ncps.wirings import AutoNCP

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(Encoder, self).__init__()
        wiring = AutoNCP(hidden_dim, output_dim)
        self.rnn = CfC(input_dim, wiring)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        batch_size = x.size(0)
        print(x.shape)
        h0 = torch.zeros(batch_size, self.rnn.state_size, device=x.device)
        print(h0.shape)
        outputs, hn = self.rnn(x, h0)
        hn = self.fc(hn)
        return hn

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(Decoder, self).__init__()
        wiring = AutoNCP(hidden_dim, output_dim)
        self.rnn = CfC(input_dim, wiring)
        self.fc = nn.Linear(output_dim, output_dim)

    def forward(self, x, hidden):
        # batch_size = x.size(0)
        h0 = hidden
        outputs, hn = self.rnn(x, h0)
        predictions = self.fc(outputs)
        return predictions, hn


class s2s_ncp(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(s2s_ncp, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, hidden_dim // 2, num_layers)
        self.decoder = Decoder(output_dim, hidden_dim, output_dim, num_layers)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        hidden = self.encoder(x)

        decoder_input = torch.zeros(batch_size, 10, 1).to(x.device)  # Initialize decoder input with appropriate dimension
        decoder_outputs = []

        for t in range(seq_len):
            print(decoder_input.shape)
            print(hidden.shape)
            decoder_output, hidden = self.decoder(decoder_input, hidden)
            decoder_outputs.append(decoder_output)
            decoder_input = decoder_output

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        return decoder_outputs
