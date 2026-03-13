import torch
from torch.nn import CrossEntropyLoss
from fluke.client import Client

class DPClient(Client):
    def __init__(
        self,
        index,
        train_set,
        test_set,
        optimizer_cfg,
        local_epochs: int = 5,
        fine_tuning_epochs: int = 0,
        clipping: float = 1.0,
        dp_noise_multiplier: float = 1.0,
        persistency: bool = True,
        **kwargs,
    ):
        # Utilisation de CrossEntropyLoss pour la classification binaire (2 sorties)
        loss_fn = CrossEntropyLoss()

        super().__init__(
            index=index,
            train_set=train_set,
            test_set=test_set,
            optimizer_cfg=optimizer_cfg,
            loss_fn=loss_fn,
            local_epochs=local_epochs,
            fine_tuning_epochs=fine_tuning_epochs,
            clipping=clipping,
            persistency=persistency,
        )

        self.hyper_params.update(
            dp_noise_multiplier=dp_noise_multiplier
        )

    def _add_dp_noise(self):
        C = self.hyper_params.clipping
        sigma = self.hyper_params.dp_noise_multiplier
        
        # Le bruit doit être proportionnel à la sensibilité (C) et inversement 
        # proportionnel à la taille du batch pour une analyse DP correcte.
        # Ici, on l'applique tel quel sur le gradient moyen du batch.
        for p in self.model.parameters():
            if p.grad is not None:
                noise = torch.normal(
                    mean=0,
                    std=sigma * C, 
                    size=p.grad.shape,
                    device=p.grad.device,
                )
                p.grad.add_(noise)

    def fit(self, override_local_epochs: int = 0) -> float:
        epochs = (
            override_local_epochs
            if override_local_epochs > 0
            else self.hyper_params.local_epochs
        )

        self.model.train()
        self.model.to(self.device)

        if self.optimizer is None:
            self.optimizer, self.scheduler = self._optimizer_cfg(self.model)

        running_loss = 0.0

        for _ in range(epochs):
            for X, y in self.train_set:
                X, y = X.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()

                y_hat = self.model(X)
                loss = self.hyper_params.loss_fn(y_hat, y)

                loss.backward()

                # 1. Clipping : On borne l'influence de chaque mise à jour
                self._clip_grads(self.model)

                # 2. Bruit : On rend les mises à jour confidentielles
                self._add_dp_noise()

                # 3. Step
                self.optimizer.step()
                running_loss += loss.item()

            self.scheduler.step()

        self.model.cpu()
        return running_loss / (epochs * len(self.train_set))