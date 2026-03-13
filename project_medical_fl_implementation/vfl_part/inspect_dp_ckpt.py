import torch
from pathlib import Path

model_dir = Path(__file__).resolve().parent / "models"
dp_paths = sorted(model_dir.glob("dp_best_server_f1*.pt"))
print("dp checkpoints found", dp_paths)
if not dp_paths:
    raise SystemExit("no dp checkpoint found")

dp_ckpt = dp_paths[0]
print("loading dp", dp_ckpt)
dp_state = torch.load(dp_ckpt, weights_only=True)
print("dp_state type", type(dp_state))
if isinstance(dp_state, dict):
    print("dp top-level keys:", list(dp_state.keys())[:20])

# Also inspect non-DP checkpoint if present
other_paths = sorted(model_dir.glob("best_server_f1*.pt"))
print("non-dp checkpoints found", other_paths)
if other_paths:
    ckpt = other_paths[0]
    print("loading non-dp", ckpt)
    state = torch.load(ckpt, weights_only=True)
    print("non-dp state type", type(state))
    if isinstance(state, dict):
        print("non-dp top-level keys:", list(state.keys())[:20])
