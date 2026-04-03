import torch.nn as nn

class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_layers=2, num_classes=8, dropout=0.2):
        super(CNN_LSTM_Model, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: [B, 7, 12, 512]

        B, C, F, D = x.shape
        x = x.view(B, C * F, D)  # [B, 84, 512]

        lstm_out, _ = self.lstm(x)

        # better than last timestep
        out = lstm_out.mean(dim=1)

        out = self.dropout(out)
        out = self.fc(out)

        return out