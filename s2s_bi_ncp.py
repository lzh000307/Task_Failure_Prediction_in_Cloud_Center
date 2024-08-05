import torch
import torch.nn as nn
from ncps.torch import CfC
from ncps.wirings import AutoNCP

class BiNCPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(BiNCPEncoder, self).__init__()
        wiring = AutoNCP(hidden_dim, output_dim)
        self.rnn_forward = CfC(input_dim, wiring)
        self.rnn_backward = CfC(input_dim, wiring)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

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
        hidden = self.fc(output[:, -1, :])
        return hidden.unsqueeze(0)
class NCPDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(NCPDecoder, self).__init__()
        wiring = AutoNCP(hidden_dim, output_dim)
        self.rnn = CfC(input_dim, wiring)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden):
        batch_size = x.size(0)
        h0 = hidden
        outputs, hn = self.rnn(x, h0)
        predictions = self.fc(outputs)
        return predictions, hn


class s2s_bi_ncp(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(s2s_bi_ncp, self).__init__()
        self.encoder = BiNCPEncoder(input_dim, hidden_dim, hidden_dim // 2, num_layers)
        self.decoder = NCPDecoder(hidden_dim, hidden_dim, output_dim, num_layers)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        hidden = self.encoder(x)

        decoder_input = torch.zeros(batch_size, 1, 1).to(
            x.device)  # Initialize decoder input with appropriate dimension
        decoder_outputs = []

        for t in range(seq_len):
            decoder_output, hidden = self.decoder(decoder_input, hidden)
            decoder_outputs.append(decoder_output)
            decoder_input = decoder_output

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        return decoder_outputs
