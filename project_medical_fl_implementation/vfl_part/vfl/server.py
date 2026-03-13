import torch
import torch.nn as nn
import torch.nn.functional as F
from vfl.models import ServerModel


def _macro_f1_score(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> float:
    """Compute macro F1 score for multi-class classification."""
    eps = 1e-9
    f1s = []
    for c in range(num_classes):
        tp = ((y_pred == c) & (y_true == c)).sum().item()
        fp = ((y_pred == c) & (y_true != c)).sum().item()
        fn = ((y_pred != c) & (y_true == c)).sum().item()
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1s.append(2 * precision * recall / (precision + recall + eps))
    return float(sum(f1s) / len(f1s))


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
        f1 = _macro_f1_score(y, preds, logits.size(1))
        # récupérer gradient de chaque embedding pour le client
        embedding_grads = [e.grad.detach().clone() for e in embeddings]
        return loss.item(), acc, f1, embedding_grads
