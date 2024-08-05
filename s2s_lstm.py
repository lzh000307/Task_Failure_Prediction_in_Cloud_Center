import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers=1):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(output_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden, cell):
        outputs, (hidden, cell) = self.lstm(x, (hidden, cell))
        predictions = self.fc(outputs)
        return predictions, hidden, cell


class s2s_lstm(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(s2s_lstm, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, num_layers)
        self.decoder = Decoder(output_dim, hidden_dim, num_layers)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        hidden, cell = self.encoder(x)

        decoder_input = torch.zeros(batch_size, 1, 1).to(x.device)
        decoder_outputs = []

        for t in range(seq_len):
            decoder_output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            decoder_outputs.append(decoder_output)
            decoder_input = decoder_output

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        return decoder_outputs