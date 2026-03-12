import torch
import torch.nn as nn
import torch.nn.functional as F
from vfl.models import ServerModel

class VFLServer:
    def __init__(self, n_clients, emb_dim, n_classes=3):
        """
        n_clients: nombre de clients
        emb_dim: dimension embedding de chaque client
        n_classes: classes de sortie
        """
        self.model = ServerModel(n_clients, emb_dim, n_classes)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    # Calcul loss et gradient sur embeddings
    def train_step(self, embeddings, y):
        self.model.train()
        # concaténation des embeddings
        x = torch.cat(embeddings, dim=1)
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean().item()
        # récupérer gradient de chaque embedding pour le client
        embedding_grads = [e.grad.detach().clone() for e in embeddings]
        return loss.item(), acc, embedding_grads