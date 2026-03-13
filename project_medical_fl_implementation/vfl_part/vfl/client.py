import torch
import flwr as fl

class VFLClient(fl.client.NumPyClient):
    def __init__(self, local_model, x):
        """
        local_model: modèle client
        x: données locales
        """
        self.local_model = local_model
        self.x = x
        self.optimizer = torch.optim.Adam(self.local_model.parameters(), lr=0.001)

    def get_parameters(self, config):
        return [p.detach().numpy() for p in self.local_model.parameters()]

    def set_parameters(self, parameters):
        for p, new in zip(self.local_model.parameters(), parameters):
            p.data = torch.tensor(new, dtype=p.dtype)

    # Forward pass côté client → embedding
    def forward(self):
        self.local_model.eval()
        with torch.no_grad():
            embedding = self.local_model(self.x)
        return embedding

    # Fit appelé par Flower. We ignore any server-side gradients here
    # because the server handles the VFL gradient updates itself.
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        return self.get_parameters(None), len(self.x), {}

    # Evaluation côté client n'est plus nécessaire (server fait la loss)
    def evaluate(self, parameters, config):
        return 0.0, len(self.x), {}