from fastapi_app.app.telemetry import init_otel


def test_init_otel_no_endpoint(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    assert init_otel() is False


def test_init_otel_with_endpoint(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost")
    assert init_otel() is True


def test_init_otel_handles_exception(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost")

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("opentelemetry.sdk.trace.TracerProvider", boom)
    assert init_otel() is False
