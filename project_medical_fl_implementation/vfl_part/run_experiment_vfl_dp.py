import torch
from pathlib import Path

import flwr as fl
from vfl_dp.dataset import load_dataset
from vfl_dp.split import vertical_split
from vfl_dp.models import ClientModel, ServerModel
from vfl_dp.client import VFLClient, epsilon, delta, max_grad_norm, noise_multiplier
from vfl_dp.server import VFLServer
from vfl_dp.strategy import VFLStrategy
from vfl_dp.metrics import MetricsLogger

EMB_DIM = 8
ROUNDS = 20


def _dp_tag() -> str:
    # Use DP params to build a stable tag for filenames/dirs
    def fmt(x: float) -> str:
        return str(x).replace(".", "p")

    return f"dp_eps{fmt(epsilon)}_delta{fmt(delta)}_max{fmt(max_grad_norm)}_noise{fmt(noise_multiplier)}"

def main():
    df = load_dataset()
    data_dict, y = vertical_split(df)

    dp_tag = _dp_tag()
    metrics_logger = MetricsLogger(
        log_path=f"logs/metrics_global_{dp_tag}.csv",
        client_log_path=f"logs/metrics_clients_{dp_tag}.csv",
    )

    # Utiliser tous les clients générés par `vertical_split` (pas besoin de hardcoder le nombre)
    num_clients = len(data_dict)

    # Initialiser clients
    clients = []
    for key in data_dict:
        client_model = ClientModel(input_dim=data_dict[key].shape[1], emb_dim=EMB_DIM)
        client = VFLClient(client_model, data_dict[key])
        client.y = y.long()
        clients.append(client)

    # Initialiser serveur
    server = VFLServer(num_clients, EMB_DIM)

    # Stratégie personnalisée
    model_dir = Path(__file__).resolve().parent / "models" / dp_tag
    strategy = VFLStrategy(
        server=server,
        clients=clients,
        metrics_logger=metrics_logger,
        model_dir=model_dir,
    )

    # Lancer simulation
    from flwr.common import Context
    from itertools import count

    node_id_to_idx: dict[int, int] = {}
    client_idx_counter = count()

    def client_fn(context: Context):
        # Flower uses an internal node_id to identify clients.
        # Map each node_id to a stable index in our `clients` list.
        idx = node_id_to_idx.setdefault(context.node_id, next(client_idx_counter))
        return clients[idx].to_client()

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy,
    )

    metrics_logger.save()

if __name__ == "__main__":
    main()