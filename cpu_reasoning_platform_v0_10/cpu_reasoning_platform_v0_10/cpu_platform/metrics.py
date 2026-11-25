"""Process-local counters. No prompt data and no user content is retained."""
from collections import Counter
from threading import Lock


class Metrics:
    def __init__(self):
        self._lock = Lock()
        self._counts = Counter()
        self._inflight = 0
        self._latencies_ms = []

    def begin(self):
        with self._lock:
            self._inflight += 1

    def end(self, status: str, route: str, latency_ms: float, calls: int):
        with self._lock:
            self._inflight -= 1
            self._counts["requests_total"] += 1
            self._counts[f"status_{status}"] += 1
            self._counts[f"route_{route}"] += 1
            self._counts["model_calls_total"] += calls
            self._latencies_ms.append(latency_ms)
            if len(self._latencies_ms) > 1000:
                self._latencies_ms = self._latencies_ms[-1000:]

    def snapshot(self):
        with self._lock:
            latencies = sorted(self._latencies_ms)
            return {
                **dict(self._counts),
                "inflight": self._inflight,
                "recent_latency_p50_ms": (latencies[int((len(latencies)-1)*.5)] if latencies else None),
                "recent_latency_p95_ms": (latencies[int((len(latencies)-1)*.95)] if latencies else None),
                "latency_window": len(latencies),
            }
