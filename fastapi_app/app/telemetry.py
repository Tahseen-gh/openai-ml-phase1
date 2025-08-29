# fastapi_app/app/telemetry.py
def init_otel() -> bool:
    """
    Initialize OpenTelemetry tracing if OTEL_EXPORTER_OTLP_ENDPOINT is set.
    Safe no-op otherwise.
    """
    try:
        import os

        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        if not endpoint:
            return False

        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({"service.name": "openai-ml-phase1"})
        provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        return True
    except Exception:
        return False
