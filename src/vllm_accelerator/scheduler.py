"""
Continuous Batching Scheduler
Iteratively schedules token generation across concurrent sequences without waiting for batch completion.
"""

from typing import List, Dict, Any, Optional
import time
from .kv_cache_manager import PagedKVCacheManager
from .speculative_engine import SpeculativeEngine

class RequestSequence:
    def __init__(self, request_id: str, prompt: str, max_tokens: int = 32):
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_tokens = prompt.split()
        self.generated_tokens: List[str] = []
        self.max_tokens = max_tokens
        self.is_finished = False
        self.arrival_time = time.time()
        self.first_token_time: Optional[float] = None

class ContinuousBatchScheduler:
    def __init__(self, kv_manager: PagedKVCacheManager, engine: SpeculativeEngine, max_batch_size: int = 16):
        self.kv_manager = kv_manager
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.waiting_queue: List[RequestSequence] = []
        self.running_batch: List[RequestSequence] = []

    def submit_request(self, request_id: str, prompt: str, max_tokens: int = 32) -> RequestSequence:
        seq = RequestSequence(request_id, prompt, max_tokens)
        self.waiting_queue.append(seq)
        return seq

    def step(self) -> List[Dict[str, Any]]:
        """
        Executes one iteration-level continuous batch step across all active sequences.
        """
        # 1. Admit waiting requests if capacity exists
        while self.waiting_queue and len(self.running_batch) < self.max_batch_size:
            seq = self.waiting_queue.pop(0)
            try:
                self.kv_manager.allocate(seq.request_id, len(seq.prompt_tokens))
                self.running_batch.append(seq)
            except MemoryError:
                # KV memory full, hold in queue
                self.waiting_queue.insert(0, seq)
                break

        step_outputs = []
        finished_seqs = []

        # 2. Speculative decoding step for each running sequence
        for seq in self.running_batch:
            context = seq.prompt_tokens + seq.generated_tokens
            step_res = self.engine.step_generation(context)
            new_tokens = step_res["accepted_tokens"]

            if seq.first_token_time is None:
                seq.first_token_time = time.time()

            seq.generated_tokens.extend(new_tokens)

            # Update KV cache blocks
            for _ in new_tokens:
                self.kv_manager.append_slot(seq.request_id, len(seq.prompt_tokens) + len(seq.generated_tokens))

            # Check completion
            if len(seq.generated_tokens) >= seq.max_tokens:
                seq.is_finished = True
                finished_seqs.append(seq)
                self.kv_manager.free(seq.request_id)

            step_outputs.append({
                "request_id": seq.request_id,
                "emitted_tokens": new_tokens,
                "num_tokens": len(new_tokens),
                "is_finished": seq.is_finished
            })

        # Remove finished sequences from running batch
        self.running_batch = [s for s in self.running_batch if not s.is_finished]

        return step_outputs
