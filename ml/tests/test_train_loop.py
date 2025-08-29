from ml.train_loop import train_minimal


def test_train_loop_runs():
    loss = train_minimal(epochs=1)
    # If torch is missing, we accept None. If present, expect finite loss.
    assert loss is None or (isinstance(loss, float) and loss > 0.0)


def test_train_loop_with_mocked_torch(monkeypatch):
    import types
    import sys

    class DummyTensor:
        def __init__(self, value: float = 1.0) -> None:
            self.value = value

        def backward(self) -> None:  # pragma: no cover - no real grad calc
            pass

        def detach(self) -> "DummyTensor":
            return self

        def cpu(self) -> "DummyTensor":
            return self

        def __float__(self) -> float:  # allow float() conversion
            return self.value

    class DummyLinear:
        def __call__(self, x: DummyTensor) -> DummyTensor:
            return x

    class DummyReLU:
        def __call__(self, x: DummyTensor) -> DummyTensor:
            return x

    class DummySequential:
        def __init__(self, *layers: object) -> None:
            self.layers = layers

        def parameters(self):  # pragma: no cover - no params
            return []

        def train(self) -> None:
            pass

        def __call__(self, x: DummyTensor) -> DummyTensor:
            for layer in self.layers:
                x = layer(x)
            return x

    class DummyAdam:
        def __init__(self, params, lr: float):
            pass

        def zero_grad(self) -> None:
            pass

        def step(self) -> None:
            pass

    class DummyMSELoss:
        def __call__(self, pred: DummyTensor, y: DummyTensor) -> DummyTensor:
            return DummyTensor(0.5)

    dummy_torch = types.SimpleNamespace(
        randn=lambda *a, **k: DummyTensor(),
        nn=types.SimpleNamespace(
            Sequential=DummySequential,
            Linear=lambda in_, out_: DummyLinear(),
            ReLU=DummyReLU,
            MSELoss=DummyMSELoss,
        ),
        optim=types.SimpleNamespace(Adam=DummyAdam),
    )

    monkeypatch.setitem(sys.modules, "torch", dummy_torch)
    monkeypatch.setitem(sys.modules, "torch.nn", dummy_torch.nn)
    monkeypatch.setitem(sys.modules, "torch.optim", dummy_torch.optim)

    loss = train_minimal(epochs=1)
    assert isinstance(loss, float)
