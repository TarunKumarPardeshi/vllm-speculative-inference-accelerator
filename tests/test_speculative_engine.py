"""
Unit and Integration Tests for vLLM Speculative Decoding Inference Accelerator
"""

import pytest
from fastapi.testclient import TestClient

from src.vllm_accelerator.kv_cache_manager import PagedKVCacheManager
from src.vllm_accelerator.speculative_engine import SpeculativeEngine
from src.vllm_accelerator.scheduler import ContinuousBatchScheduler
from src.vllm_accelerator.api import app

client = TestClient(app)

def test_paged_kv_cache_manager():
    manager = PagedKVCacheManager(num_blocks=100, block_size=16)
    assert manager.get_utilization() == 0.0

    # Allocate for 35 tokens -> requires 3 blocks (16 * 3 = 48 slots)
    blocks = manager.allocate("seq-1", prompt_len=35)
    assert len(blocks) == 3
    assert manager.get_utilization() == 3.0

    # Append slot without crossing block boundary
    b = manager.append_slot("seq-1", current_len=36)
    assert b == blocks[-1]

    # Free sequence
    manager.free("seq-1")
    assert manager.get_utilization() == 0.0

def test_speculative_engine_step():
    engine = SpeculativeEngine("target-70b", "draft-1b", k_tokens=4)
    res = engine.step_generation(["enterprise", "cloud", "architecture"])

    assert len(res["draft_tokens"]) == 4
    assert res["num_accepted"] >= 1
    assert res["speedup_factor"] >= 1.0
    assert engine.get_aggregate_acceptance_rate() > 0.0

def test_continuous_batch_scheduler():
    kv = PagedKVCacheManager(num_blocks=200, block_size=16)
    eng = SpeculativeEngine("target-70b", "draft-1b", k_tokens=4)
    sched = ContinuousBatchScheduler(kv, eng, max_batch_size=4)

    seq = sched.submit_request("req-1", "Benchmarking transformer serving", max_tokens=12)
    assert seq.is_finished is False

    while not seq.is_finished:
        sched.step()

    assert seq.is_finished is True
    assert len(seq.generated_tokens) >= 12
    assert kv.get_utilization() == 0.0  # Blocks freed upon completion

def test_api_chat_completions():
    resp = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "How does continuous batching work?"}],
        "max_tokens": 16
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert "speculative_metrics" in data
    assert data["speculative_metrics"]["effective_speedup"] > 1.0
