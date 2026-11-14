"""
PagedAttention KV Cache Block Manager
Simulates virtual memory page-table allocation for Key-Value cache tensors to eliminate external fragmentation.
"""

from typing import Dict, List, Optional
import math

class PhysicalBlock:
    def __init__(self, block_number: int, block_size: int = 16):
        self.block_number = block_number
        self.block_size = block_size
        self.ref_count = 0
        self.is_allocated = False

class PagedKVCacheManager:
    def __init__(self, num_blocks: int = 2048, block_size: int = 16):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.free_blocks: List[int] = list(range(num_blocks))
        self.block_table: Dict[str, List[int]] = {}  # seq_id -> list of physical block numbers

    def allocate(self, seq_id: str, prompt_len: int) -> List[int]:
        """
        Allocates physical memory pages for a sequence based on token length.
        """
        needed_blocks = math.ceil(prompt_len / self.block_size)
        if len(self.free_blocks) < needed_blocks:
            raise MemoryError("Out of KV Cache physical memory blocks.")

        allocated = []
        for _ in range(needed_blocks):
            b_id = self.free_blocks.pop(0)
            allocated.append(b_id)

        self.block_table[seq_id] = allocated
        return allocated

    def append_slot(self, seq_id: str, current_len: int) -> int:
        """
        Appends a new token position into the KV cache page table.
        Allocates a new block if current block is full.
        """
        blocks = self.block_table.get(seq_id, [])
        if current_len % self.block_size == 0:
            if not self.free_blocks:
                raise MemoryError("KV Cache full during auto-regressive generation.")
            new_block = self.free_blocks.pop(0)
            blocks.append(new_block)
            self.block_table[seq_id] = blocks
            return new_block
        return blocks[-1]

    def free(self, seq_id: str) -> None:
        if seq_id in self.block_table:
            blocks = self.block_table.pop(seq_id)
            self.free_blocks.extend(blocks)

    def get_utilization(self) -> float:
        used_blocks = self.num_blocks - len(self.free_blocks)
        return round((used_blocks / self.num_blocks) * 100, 2)
