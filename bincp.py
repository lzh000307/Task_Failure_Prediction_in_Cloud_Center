import torch
import torch.nn as nn

class BiNCP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(BiNCP, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 定义前向和后向的 NCP 层
        self.recurrent_fwd = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.recurrent_bwd = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)

        # 全连接层，结合前向和后向的输出
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)

        # 初始化隐藏状态
        h0_fwd = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        h0_bwd = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)

        # 前向计算
        out_fwd, _ = self.recurrent_fwd(x, h0_fwd)

        # 反向计算，需要反转输入序列
        x_bwd = torch.flip(x, [1])
        out_bwd, _ = self.recurrent_bwd(x_bwd, h0_bwd)
        out_bwd = torch.flip(out_bwd, [1])  # 再次反转，使其与原序列对齐

        # 结合前向和后向的输出
        out = torch.cat((out_fwd, out_bwd), dim=2)

        # 通过全连接层
        out = self.fc(out)
        return out