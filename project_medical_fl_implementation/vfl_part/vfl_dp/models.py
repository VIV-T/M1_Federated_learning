import torch.nn as nn

class ClientModel(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, emb_dim)
        )

    def forward(self, x):
        return self.net(x)


class ServerModel(nn.Module):
    def __init__(self, n_clients, emb_dim, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_clients*emb_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        return self.net(x)