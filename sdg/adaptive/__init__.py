"""
Adaptive concurrency control for maximizing throughput with vLLM/SGLang backends.

This module provides:
- DynamicSemaphore: Real-time capacity adjustable semaphore
- AdaptiveController: AIMD-based concurrency controller
- AdaptiveConcurrencyManager: High-level manager with metrics collection
- MetricsCollector: Backend metrics collection for vLLM/SGLang
- RequestBatcher: Request batching for improved throughput
"""

from .controller import AdaptiveController, AdaptiveConcurrencyManager, DynamicSemaphore
from .metrics import MetricsCollector, BackendMetrics, MetricsType
from .batcher import RequestBatcher, AdaptiveRequestBatcher, PendingRequest, BatchResult

__all__ = [
    # Dynamic semaphore
    "DynamicSemaphore",
    # Adaptive controller
    "AdaptiveController",
    "AdaptiveConcurrencyManager",
    # Metrics
    "MetricsCollector",
    "BackendMetrics",
    "MetricsType",
    # Batching
    "RequestBatcher",
    "AdaptiveRequestBatcher",
    "PendingRequest",
    "BatchResult",
]
