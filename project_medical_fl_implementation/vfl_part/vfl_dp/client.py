import yaml
import torch
import flwr as fl
from pathlib import Path

config_path = Path(__file__).resolve().parent.parent / "config/config.yaml"

# Charger la configuration
with open(config_path, "r") as f:
    config = yaml.safe_load(f)


dp_config = config["differential_privacy"]
epsilon = float(dp_config["epsilon"])
delta = float(dp_config["delta"])
max_grad_norm = float(dp_config["max_grad_norm"])
noise_multiplier = float(dp_config["noise_multiplier"])

def clip_embeddings(embeddings: torch.Tensor, max_norm: float) -> torch.Tensor:
    """Clip embeddings to have l2 norm at most `max_norm` per example."""
    norm = torch.norm(embeddings, p=2, dim=1, keepdim=True)
    factor = (max_norm / (norm + 1e-12)).clamp(max=1.0)
    return embeddings * factor


def add_dp_noise(
    embeddings: torch.Tensor,
    epsilon: float,
    delta: float,
    max_norm: float,
    noise_multiplier: float | None = None,
) -> torch.Tensor:
    """Add Gaussian noise to embeddings to provide (epsilon, delta)-DP.

    If `noise_multiplier` is provided, it is treated as the Gaussian noise multiplier
    (i.e., `sigma = noise_multiplier * max_norm`). Otherwise, we use the classic
    (epsilon, delta)-DP Gaussian mechanism.
    """
    device = embeddings.device
    if noise_multiplier is not None:
        sigma = noise_multiplier * max_norm
    else:
        sigma = (
            max_norm
            * torch.sqrt(2 * torch.log(torch.tensor(1.25 / delta, device=device)))
            / epsilon
        )
    noise = torch.randn_like(embeddings) * sigma
    return embeddings + noise

class VFLClient(fl.client.NumPyClient):
    def __init__(self, local_model, x, y=None):
        self.local_model = local_model
        self.x = x
        self.y = y

        # Paramètres de confidentialité différentielle (DP)
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier

        self.optimizer = torch.optim.Adam(self.local_model.parameters(), lr=config["lr"])

        # Les données locales sont conservées dans `self.x`.
        # Le clipping + le bruit différentiels sont appliqués aux embeddings lors de l'envoi au serveur.

    def get_parameters(self, config):
        return [p.detach().numpy() for p in self.local_model.parameters()]

    def set_parameters(self, parameters):
        for p, new in zip(self.local_model.parameters(), parameters):
            p.data = torch.tensor(new, dtype=p.dtype)

    def get_embeddings(self) -> torch.Tensor:
        """Compute embeddings for the entire local dataset."""
        self.local_model.eval()
        with torch.no_grad():
            embeddings = self.local_model(self.x)
        return embeddings

    def get_private_embeddings(self) -> torch.Tensor:
        """Return DP-protected embeddings to send to the server."""
        embeddings = self.get_embeddings()
        clipped_embeddings = clip_embeddings(embeddings, self.max_grad_norm)
        noisy_embeddings = add_dp_noise(
            clipped_embeddings,
            self.epsilon,
            self.delta,
            self.max_grad_norm,
            noise_multiplier=self.noise_multiplier,
        )
        return noisy_embeddings

    def apply_embedding_grad(self, embedding_grad: torch.Tensor):
        """Apply gradient w.r.t. the embeddings to update the local model."""
        self.local_model.train()
        self.optimizer.zero_grad()
        embeddings = self.local_model(self.x)
        embeddings.backward(embedding_grad)
        self.optimizer.step()

    def fit(self, parameters, config):
        # Flower uses this to send updated parameters to the client at the start of each round.
        self.set_parameters(parameters)
        return self.get_parameters(config), len(self.x), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.local_model.eval()
        with torch.no_grad():
            embeddings = self.local_model(self.x)
        return 0.0, len(self.x), {"embeddings": embeddings}