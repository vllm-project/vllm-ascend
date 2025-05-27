from typing import Tuple
from typing import Optional
import numpy as np
import pytest
import torch
import torch_npu  # noqa: F401
import vllm  # noqa: F401

import vllm_ascend.platform  # noqa: F401


def _create_mla_cache(
    num_blocks: int,
    block_size: int,
    entry_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    device: str,
) -> torch.Tensor:
    cache_dtype = torch.uint8 if kv_cache_dtype == "fp8" else dtype
    return torch.zeros(num_blocks,
                       block_size,
                       entry_size,
                       dtype=cache_dtype,
                       device=device)


def _fill_mla_cache(cache: torch.Tensor, kv_cache_dtype: str):
    rand_dtype = torch.float16 if kv_cache_dtype == "fp8" else cache.dtype
    vals = torch.randn(*cache.shape, device=cache.device, dtype=rand_dtype)
    cache.copy_(vals)

def benchmark_npu(fn, num_iterations=100, num_warmup_iterations=50):
    """
    Benchmark function for NPU operations
    
    Args:
        fn: Function to benchmark
        num_iterations: Number of timing iterations
        num_warmup_iterations: Number of warmup iterations
    
    Returns:
        float: Minimum elapsed time in seconds
    """
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    times = np.zeros(num_iterations + num_warmup_iterations)

    # Run iterations
    for i in range(num_warmup_iterations + num_iterations):
        with torch.no_grad():
            start.record()
            fn()  # Execute the function
            end.record()
        torch.npu.synchronize()
        times[i] = start.elapsed_time(end)

    # Remove warmup iterations and convert to seconds
    times = times[num_warmup_iterations:]
    elapsed_time = np.amin(times) / 1000
    return elapsed_time


def gather_cache_torch(
        src_cache: torch.Tensor,  # [NUM_BLOCKS, BLOCK_SIZE, HEAD, ENTRIES]
        dst: torch.Tensor,  # [TOT_TOKENS, ENTRIES]
        block_table: torch.Tensor,  # [BATCH, BLOCK_INDICES]
        cu_seq_lens: torch.Tensor,  # [BATCH+1]
        batch_size: int,
        seq_starts: Optional[torch.Tensor] = None  # Optional: [BATCH]
) -> None:
    """
    Gather sequence data from source cache to destination tensor
    Args:
        src_cache: Source cache tensor [NUM_BLOCKS, BLOCK_SIZE, HEAD, ENTRIES]
        dst: Destination tensor [TOT_TOKENS, ENTRIES]
        block_table: Block table mapping [BATCH, BLOCK_INDICES]
        cu_seq_lens: Cumulative sequence lengths [BATCH+1]
        batch_size: Batch size
        seq_starts: Optional, starting offsets for each batch [BATCH]
    """
    # Basic parameter checks
    assert src_cache.dtype == dst.dtype, "src_cache and dst must have same dtype"
    assert block_table.dtype == torch.int32, "block_table must be int32"
    assert cu_seq_lens.dtype == torch.int32, "cu_seq_lens must be int32"

    if seq_starts is not None:
        assert seq_starts.dtype == torch.int32, "seq_starts must be int32"

    block_size = src_cache.size(1)
    # Process each batch
    for bid in range(batch_size):
        # Get sequence start and end positions for current batch
        seq_start = cu_seq_lens[bid].item()
        seq_end = cu_seq_lens[bid + 1].item()
        seq_len = seq_end - seq_start

        if seq_len == 0:
            continue

        # Calculate required number of blocks
        tot_blocks = (seq_len + block_size - 1) // block_size

        # Calculate block offset if seq_starts is provided
        offset = 0
        if seq_starts is not None:
            offset = seq_starts[bid].item() // block_size

        # Get block table for current batch
        batch_block_table = block_table[bid, offset:offset + tot_blocks]
        # Calculate complete blocks and last partial block
        full_blocks = tot_blocks - 1 if seq_len % block_size else tot_blocks
        partial_block_size = seq_len % block_size if seq_len % block_size else 0
        # Copy complete blocks
        dst_start = seq_start
        for i in range(full_blocks):
            block_id = batch_block_table[i].item()
            # Copy entire block, remove HEAD dimension
            dst[dst_start:dst_start +
                block_size] = src_cache[block_id].squeeze(1)
            dst_start += block_size

        # Handle last incomplete block
        if partial_block_size > 0:
            block_id = batch_block_table[full_blocks].item()
            dst[dst_start:dst_start + partial_block_size] = \
            src_cache[block_id, :partial_block_size].squeeze(1)

@pytest.mark.parametrize("kv_lora_rank", [512])
@pytest.mark.parametrize("qk_rope_head_dim", [64])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("num_blocks", [1024])
@pytest.mark.parametrize("max_seq_len", [512])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kv_cache_dtype",
                         ["auto"])  # You can also test "fp8" if needed.
@torch.inference_mode()
def test_gather_cache_mla(kv_lora_rank, qk_rope_head_dim, block_size,
                          num_blocks, max_seq_len, batch_size, dtype,
                          kv_cache_dtype):
    # Set random seed and device
    device = "npu:{0}"
    seed = 0
    torch.manual_seed(seed)
    torch.set_default_device(device)
    entry_size = kv_lora_rank + qk_rope_head_dim
    src_cache = _create_mla_cache(num_blocks, block_size, entry_size, dtype,
                                  kv_cache_dtype, device)
    _fill_mla_cache(src_cache, kv_cache_dtype=kv_cache_dtype)
    seq_len_tensor = torch.randint(0,
                                   max_seq_len + 1, (batch_size, ),
                                   device=device)

    total_tokens = seq_len_tensor.sum()
    cu_seq_lens = torch.empty((batch_size + 1),
                              dtype=torch.int32,
                              device=device)
    cu_seq_lens[0] = 0
    cu_seq_lens[1:] = seq_len_tensor.cumsum(dim=0).to(dtype=torch.int32)

    # tot_blocks_tensor = (seq_len_tensor + block_size - 1) // block_size
    block_table = torch.empty((batch_size, num_blocks),
                              dtype=torch.int32,
                              device=device)

    for b in range(batch_size):
        perm = torch.randperm(num_blocks, device=device)
        block_table[b, :] = perm

    expected = torch.zeros((total_tokens, entry_size),
                           dtype=src_cache.dtype,
                           device=device)
    max_start = max_seq_len // 2
    seq_starts = torch.randint(0,
                               max_start + 1, (batch_size, ),
                               dtype=torch.int32,
                               device=device)
    gather_cache_torch(src_cache, expected, block_table, cu_seq_lens,
                       batch_size, seq_starts)


    # torch_npu._npu_paged_cache_load
    kv_c = torch.empty(
        (total_tokens, 1, kv_lora_rank), dtype=torch.float16, device=device
    )
    k_pe = torch.empty(
        (total_tokens, 1, qk_rope_head_dim), dtype=torch.float16, device=device
    )
    cached_kv_c, cached_k_pe = src_cache.split(
        [kv_lora_rank, qk_rope_head_dim], dim=2
    )
    cached_kv_c = cached_kv_c.view(num_blocks, block_size, 1, kv_lora_rank).to(torch.float16)
    cached_k_pe = cached_k_pe.view(num_blocks, block_size, 1, qk_rope_head_dim).to(torch.float16)
    torch_npu._npu_paged_cache_load(
        cached_kv_c,
        cached_k_pe,
        block_table,
        seq_len_tensor.int(),
        seq_starts,
        kv_c,
        k_pe
    )

    torch_npu_result = torch.cat([kv_c, k_pe], dim=2).view(total_tokens, -1)
    torch.testing.assert_close(torch_npu_result, expected.to(torch.float16))

    def ref_fn():
        gather_cache_torch(src_cache, expected, block_table, cu_seq_lens,
                       batch_size, seq_starts)
        
    def npu_fn():
        torch_npu._npu_paged_cache_load(
            cached_kv_c,
            cached_k_pe,
            block_table,
            seq_len_tensor.int(),
            seq_starts,
            kv_c,
            k_pe
        )
        torch_npu_result = torch.cat([kv_c, k_pe], dim=2).view(total_tokens, -1)

    # Benchmark both implementations
    ref_time = benchmark_npu(ref_fn)
    custom_time = benchmark_npu(npu_fn)

    # Print performance results
    print("\nPerformance Results:")
    print(f"Reference implementation: {ref_time*1000:.3f} ms")
    print(f"Custom implementation: {custom_time*1000:.3f} ms")
    print(f"Speedup: {ref_time/custom_time:.2f}x")