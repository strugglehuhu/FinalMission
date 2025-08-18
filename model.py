# model.py
import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: [batch, seq_len, 1]
        out, _ = self.lstm(x)          # [batch, seq_len, hidden]
        out = out[:, -1, :]            # last time-step
        out = self.fc(out)             # [batch, 1]
        return out
