import torch
import flwr as fl
from vfl.dataset import load_dataset
from vfl.split import vertical_split
from vfl.models import ClientModel, ServerModel
from vfl.client import VFLClient
from vfl.server import VFLServer
from vfl.strategy import VFLStrategy
from vfl.metrics import MetricsLogger

EMB_DIM = 8
ROUNDS = 20

def main():
    df = load_dataset()
    data_dict, y = vertical_split(df)
    metrics_logger = MetricsLogger()

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
    strategy = VFLStrategy(server=server, clients=clients, metrics_logger=metrics_logger)

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