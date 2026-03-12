import csv
import os

class MetricsLogger:
    def __init__(self, log_path="logs/metrics.csv", client_log_path="logs/metrics_clients.csv"):
        self.log_path = log_path
        self.client_log_path = client_log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        os.makedirs(os.path.dirname(client_log_path), exist_ok=True)
        self.history = []
        self.history_clients = []

    def _append_to_csv(self, path: str, row: dict):
        """Append a row to a CSV file, creating it with a header if needed."""
        exists = os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def log(self, round, loss, accuracy):
        row = {"round": round, "loss": loss, "accuracy": accuracy}
        self.history.append(row)
        self._append_to_csv(self.log_path, row)
        print(f"[Round {round}] Loss={loss:.4f}, Accuracy={accuracy:.4f}")

    def log_client(self, round, client_id, **metrics):
        row = {"round": round, "client_id": client_id}
        row.update(metrics)
        self.history_clients.append(row)
        self._append_to_csv(self.client_log_path, row)

    def save(self):
        keys = ["round", "loss", "accuracy"]
        with open(self.log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.history)
        print(f"Metrics saved to {self.log_path} (entries={len(self.history)})")

        # Client metrics
        print(f"Client metrics entries: {len(self.history_clients)}")
        if self.history_clients:
            client_keys = list({k for r in self.history_clients for k in r.keys()})
            with open(self.client_log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=client_keys)
                writer.writeheader()
                writer.writerows(self.history_clients)
            print(f"Client metrics saved to {self.client_log_path}")