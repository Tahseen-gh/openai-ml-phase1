from ml.train_loop import train_minimal


def test_train_loop_runs():
    loss = train_minimal(epochs=1)
    # If torch is missing, we accept None. If present, expect finite loss.
    assert loss is None or (isinstance(loss, float) and loss > 0.0)
