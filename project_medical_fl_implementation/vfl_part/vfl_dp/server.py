import torch
import torch.nn as nn
import torch.nn.functional as F


def _macro_f1_score(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> float:
    """Compute macro F1 score for multi-class classification."""
    # y_true / y_pred are expected to be 1D class labels
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


class ServerModel(nn.Module):
    def __init__(self, n_clients, emb_dim, n_classes=3):
        super(ServerModel, self).__init__()
        self.fc = nn.Linear(n_clients * emb_dim, n_classes)

    def forward(self, x):
        return self.fc(x)


class VFLServer:
    def __init__(self, n_clients, emb_dim, n_classes=3, epsilon=1.0, delta=1e-5):
        self.model = ServerModel(n_clients, emb_dim, n_classes)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.epsilon = epsilon
        self.delta = delta
        self.n_classes = n_classes

    def train_step(self, embeddings, y):
        """Train the server model on the concatenated embeddings.

        Returns:
            loss: float loss value
            accuracy: float accuracy on the current batch
            f1: float macro F1 score
            embedding_grads: list of tensors corresponding to the gradient w.r.t. each client's embeddings
        """
        self.model.train()

        # Ensure we can compute gradients w.r.t. embeddings.
        embeddings = [emb.detach().requires_grad_(True) for emb in embeddings]
        x = torch.cat(embeddings, dim=1)

        self.optimizer.zero_grad()
        output = self.model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()

        # Collect gradients for each client's embedding
        embedding_grads = [emb.grad.clone() for emb in embeddings]

        self.optimizer.step()

        # Compute accuracy and F1 for logging
        with torch.no_grad():
            pred = output.argmax(dim=1)
            accuracy = pred.eq(y).sum().item() / y.shape[0]
            f1 = _macro_f1_score(y, pred, self.n_classes)

        return loss.item(), accuracy, f1, embedding_grads

    def evaluate(self, embeddings, y):
        self.model.eval()
        with torch.no_grad():
            x = torch.cat(embeddings, dim=1)
            output = self.model(x)
            loss = F.cross_entropy(output, y)
            pred = output.argmax(dim=1, keepdim=True)
            accuracy = pred.eq(y.view_as(pred)).sum().item() / y.shape[0]
        return loss.item(), accuracy
