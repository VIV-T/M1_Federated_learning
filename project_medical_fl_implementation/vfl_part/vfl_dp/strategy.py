import os
from pathlib import Path

import torch
import flwr as fl


class VFLStrategy(fl.server.strategy.Strategy):
    def __init__(self, server, clients, metrics_logger, model_dir="models", **kwargs):
        super().__init__(**kwargs)
        self.server = server
        self.clients = clients
        self.metrics_logger = metrics_logger
        self.best_f1 = -float("inf")
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    # Méthode obligatoire
    def initialize_parameters(self, client_manager):
        return None

    # Round de fit
    def configure_fit(self, server_round, parameters, client_manager):
        # Flower expects a list of (ClientProxy, FitIns) tuples.
        # We sample the same number of clients as we have local models and store a mapping
        # from the client ID (as string) to our local client index.
        selected = client_manager.sample(num_clients=len(self.clients))
        selected_cids = [str(client.cid) for client in selected]
        self._cid_to_client_idx = {cid: idx for idx, cid in enumerate(selected_cids)}

        client_instructions = []
        for idx, client_proxy in enumerate(selected):
            client_params = self.clients[idx].get_parameters(None)
            parameters_proto = fl.common.ndarrays_to_parameters(client_params)
            client_instructions.append(
                (client_proxy, fl.common.FitIns(parameters_proto, {}))
            )
        return client_instructions

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        active_idxs = []
        embeddings = []
        print(f"[VFLStrategy] aggregate_fit results count={len(results)}")

        # Collect private embeddings from each selected client
        for client_proxy, fit_res in results:
            cid = str(client_proxy.cid)
            idx = self._cid_to_client_idx.get(cid)
            if idx is None or idx >= len(self.clients):
                print(f"[VFLStrategy] missing mapping for cid={cid!r} (idx={idx})")
                continue

            params = fl.common.parameters_to_ndarrays(fit_res.parameters)
            client = self.clients[idx]
            client.set_parameters(params)

            # Apply local differential privacy to the embeddings before sending them to the server
            emb = client.get_private_embeddings()
            embeddings.append(emb)
            active_idxs.append(idx)

        # Compute loss/accuracy/F1 on server and obtain gradients w.r.t. each client's embedding
        loss, acc, f1, embedding_grads = self.server.train_step(
            embeddings, self.clients[0].y
        )

        # Apply gradients back to clients and log per-client metrics
        for i, idx in enumerate(active_idxs):
            client = self.clients[idx]
            grad = embedding_grads[i]
            client.apply_embedding_grad(grad)

            # Log per-client metrics (e.g., grad norm, embedding norm)
            emb = embeddings[i]
            self.metrics_logger.log_client(
                server_round,
                idx,
                embedding_norm=float(emb.norm().item()),
                grad_norm=float(grad.norm().item()),
            )

        # Save best model by F1 score
        if f1 > self.best_f1:
            self.best_f1 = f1

            dp_tag = self.model_dir.name

            # Server model
            torch.save(
                self.server.model.state_dict(),
                self.model_dir / f"dp_best_server_f1_{dp_tag}.pt",
            )
            # Client models
            for i, client in enumerate(self.clients):
                torch.save(
                    client.local_model.state_dict(),
                    self.model_dir / f"dp_best_client_{i}_f1_{dp_tag}.pt",
                )
            print(f"[VFLStrategy] Saved best models (F1={f1:.4f}) to {self.model_dir}")

        # Log global metrics (including F1)
        self.metrics_logger.log(server_round, loss, acc, f1=f1)

        # Return a placeholder global parameter set (Flower requires it)
        # Here we simply return the parameters of the first client.
        return (
            fl.common.ndarrays_to_parameters(
                self.clients[0].get_parameters(None)
            ),
            {},
        )

    # Méthodes d’évaluation obligatoires
    def configure_evaluate(self, server_round, parameters, client_manager):
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    # Méthode evaluate obligatoire avec trois arguments (server_round, parameters, config).
    # Flower may call this with only (server_round, parameters), so we allow config to be optional.
    def evaluate(self, server_round, parameters, config=None):
        # On gère la loss/accuracy ailleurs, donc on renvoie juste un placeholder
        return 0.0, {}