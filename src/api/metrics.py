"""
api/metrics.py

In-memory counters backing GET /metrics (ARCHITECTURE.md §5): request
counts, latency percentiles, score histogram, flag rate. Per-process state,
same caveat as rate_limit.py -- correct for a single instance, resets on
restart/redeploy, does not aggregate across instances. A real deployment
would export these to Cloud Monitoring instead; this is the zero-dependency
version that needs nothing running alongside the API.
"""

from collections import deque

SCORE_BUCKET_LABELS = ["[0.0-0.2)", "[0.2-0.4)", "[0.4-0.6)", "[0.6-0.8)", "[0.8-1.0]"]
MAX_TRACKED_LATENCIES = 5000


class MetricsTracker:
    def __init__(self):
        self.request_count = 0
        self.score_count = 0
        self.flagged_count = 0
        self.blocked_count = 0
        self._latencies: deque[float] = deque(maxlen=MAX_TRACKED_LATENCIES)
        self._score_histogram = dict.fromkeys(SCORE_BUCKET_LABELS, 0)

    def record_request(self) -> None:
        self.request_count += 1

    def record_score(self, probability: float, flagged: bool, decision: str,
                      latency_ms: float) -> None:
        self.score_count += 1
        if flagged:
            self.flagged_count += 1
        if decision == "BLOCK":
            self.blocked_count += 1
        self._latencies.append(latency_ms)

        bucket_idx = min(int(probability * len(SCORE_BUCKET_LABELS)), len(SCORE_BUCKET_LABELS) - 1)
        self._score_histogram[SCORE_BUCKET_LABELS[bucket_idx]] += 1

    def _percentile(self, p: float) -> float:
        if not self._latencies:
            return 0.0
        ordered = sorted(self._latencies)
        idx = min(int(len(ordered) * p), len(ordered) - 1)
        return ordered[idx]

    def snapshot(self) -> dict:
        return {
            "request_count": self.request_count,
            "score_count": self.score_count,
            "flagged_count": self.flagged_count,
            "blocked_count": self.blocked_count,
            "flag_rate": (self.flagged_count / self.score_count) if self.score_count else 0.0,
            "latency_p50_ms": self._percentile(0.50),
            "latency_p95_ms": self._percentile(0.95),
            "latency_p99_ms": self._percentile(0.99),
            "score_histogram": dict(self._score_histogram),
        }
