"""Per-application Prometheus registry without high-cardinality labels."""

from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


@dataclass(frozen=True)
class ServiceMetrics:
    registry: CollectorRegistry
    requests: Counter
    failures: Counter
    duration: Histogram
    texts: Counter
    model_ready: Gauge
    index_size: Gauge


def create_metrics() -> ServiceMetrics:
    registry = CollectorRegistry()
    return ServiceMetrics(
        registry=registry,
        requests=Counter(
            "embedding_http_requests_total",
            "HTTP requests by bounded route and status",
            ("route", "status"),
            registry=registry,
        ),
        failures=Counter(
            "embedding_http_failures_total",
            "Expected inference failures by category",
            ("category",),
            registry=registry,
        ),
        duration=Histogram(
            "embedding_http_request_duration_seconds",
            "HTTP request duration by bounded route",
            ("route",),
            registry=registry,
        ),
        texts=Counter(
            "embedding_texts_encoded_total",
            "Number of texts encoded",
            registry=registry,
        ),
        model_ready=Gauge(
            "embedding_model_ready",
            "One when a model is ready",
            registry=registry,
        ),
        index_size=Gauge(
            "embedding_index_size",
            "Number of indexed documents",
            registry=registry,
        ),
    )
