import torch
from torch.nn import CrossEntropyLoss
from fluke.client import Client

# --------------------------------------------------
# Client with Differential Privacy (Local DP)
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
        # Loss set here (simple YAML compatible)
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

        # Add DP hyperparameter
        self.hyper_params.update(
            dp_noise_multiplier=dp_noise_multiplier
        )

    # --------------------------------------------------
    # Addition of Gaussian noise (Local DP)
    # --------------------------------------------------
    def _add_dp_noise(self):

        C = self.hyper_params.clipping
        sigma = self.hyper_params.dp_noise_multiplier

        if C <= 0:
            raise ValueError(
                "Clipping must be > 0 to ensure Differential Privacy."
            )

        for p in self.model.parameters():
            if p.grad is not None:
                noise = torch.normal(
                    mean=0,
                    std=sigma * C,
                    size=p.grad.shape,
                    device=p.grad.device,
                )
                p.grad.add_(noise)

    # --------------------------------------------------
    # Override the fit to inject LDP
    # --------------------------------------------------
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
            for _, (X, y) in enumerate(self.train_set):

                X, y = X.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()

                y_hat = self.model(X)
                loss = self.hyper_params.loss_fn(y_hat, y)

                loss.backward()

                # 1. Gradient clipping (already provided by Fluke)
                self._clip_grads(self.model)

                # 2. Addition of Gaussian noise (Local DP)
                self._add_dp_noise()

                # 3. Update
                self.optimizer.step()

                running_loss += loss.item()

            self.scheduler.step()

        running_loss /= epochs * len(self.train_set)

        self.model.cpu()
        torch.cuda.empty_cache()

        return running_loss