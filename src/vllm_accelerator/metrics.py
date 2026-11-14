"""
Inference Performance & Telemetry Exporter
Tracks TTFT (Time-To-First-Token), tokens/sec, speculative acceptance rate, and KV cache utilization.
"""

from typing import Dict, Any

class InferenceMetricsCollector:
    def __init__(self):
        self.total_requests = 0
        self.total_tokens_emitted = 0
        self.total_ttft_ms = 0.0

    def record_completed_request(self, token_count: int, ttft_ms: float):
        self.total_requests += 1
        self.total_tokens_emitted += token_count
        self.total_ttft_ms += ttft_ms

    def export_metrics(self, kv_utilization: float, acceptance_rate: float) -> Dict[str, Any]:
        avg_ttft = round(self.total_ttft_ms / max(self.total_requests, 1), 2)
        return {
            "vllm_total_requests_processed": self.total_requests,
            "vllm_total_tokens_emitted": self.total_tokens_emitted,
            "vllm_avg_time_to_first_token_ms": avg_ttft,
            "vllm_speculative_acceptance_rate_pct": acceptance_rate,
            "vllm_kv_cache_utilization_pct": kv_utilization,
            "vllm_continuous_batching_active": True
        }
