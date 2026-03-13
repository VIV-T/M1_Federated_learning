import flwr as fl

class VFLStrategy(fl.server.strategy.Strategy):
    def __init__(self, server, clients, metrics_logger, **kwargs):
        super().__init__(**kwargs)
        self.server = server
        self.clients = clients
        self.metrics_logger = metrics_logger

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
        for client_proxy, fit_res in results:
            cid = str(client_proxy.cid)
            idx = self._cid_to_client_idx.get(cid)
            if idx is None or idx >= len(self.clients):
                print(f"[VFLStrategy] missing mapping for cid={cid!r} (idx={idx})")
                continue
            params = fl.common.parameters_to_ndarrays(fit_res.parameters)
            self.clients[idx].set_parameters(params)
            emb = self.clients[idx].forward()
            emb.requires_grad_()
            embeddings.append(emb)
            active_idxs.append(idx)

        # calcul loss et gradient côté serveur
        loss, acc, embedding_grads = self.server.train_step(embeddings, self.clients[0].y)

        # Log per-client metrics (e.g., grad norm, embedding norm)
        for i, idx in enumerate(active_idxs):
            emb = embeddings[i]
            grad = embedding_grads[i]
            self.metrics_logger.log_client(
                server_round,
                idx,
                embedding_norm=float(emb.norm().item()),
                grad_norm=float(grad.norm().item()),
            )

        new_parameters = []
        for i, client in enumerate(self.clients):
            params, _, _ = client.fit(client.get_parameters(None), {})
            new_parameters.append(params)

        self.metrics_logger.log(server_round, loss, acc)

        # Return something consistent with Flower's expectation: the server's global parameters.
        # This example does not use a global model, so we just return the parameters from the first client.
        return fl.common.ndarrays_to_parameters(new_parameters[0]), {}

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