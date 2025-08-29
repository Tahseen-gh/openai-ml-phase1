"""Minimal training loop stub.
Skips if torch is not installed; useful to keep tests light in CI.
"""


def train_minimal(epochs: int = 1):
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim

        x = torch.randn(64, 10)
        y = torch.randn(64, 1)
        model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
        opt = optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
        return float(loss.detach().cpu())
    except Exception:
        # Torch not installed or CPU-only environment; return sentinel.
        return None
