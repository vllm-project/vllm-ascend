import torch
from vllm.v1.worker.gpu.block_table import _compute_slot_mappings_kernel as ref_compute_slot_mappings_kernel

from vllm_ascend.worker.v2.block_table import _compute_slot_mappings_kernel as ascend_compute_slot_mappings_kernel


def test_compute_slot_mapping_npu_kernel():
    """
    Computes the physical slot IDs in KV cache for each token in the current batch.
    This function maps the logical positions of tokens to their actual storage locations
    in the block-managed KV cache, which is critical for efficient memory access in LLM inference.

    Input:
        - max_num_batched_tokens (int): Maximum preallocated batched tokens in KV cache (memory limit)
        - idx_mapping (torch.Tensor): [num_reqs], int32 → Virtual-to-actual request index mapping
        - query_start_loc (torch.Tensor): [num_reqs+1], int32 → Batch-level token start positions per request
        - positions (torch.Tensor): [num_tokens], int64 → Per-token logical sequence positions in requests
        - block_table_ptrs (torch.Tensor): [num_kv_cache_groups], int32 → Pointers to block tables (virtual→physical)
        - block_table_strides (torch.Tensor): [num_kv_cache_groups], int32 → Stride for block table addressing
        - block_sizes_tensor (torch.Tensor): [num_kv_cache_groups], int32 → Token capacity per KV cache block
        - slot_mappings (torch.Tensor): [num_kv_cache_groups, max_num_batched_tokens], int32 → Output slot ID tensor
        - slot_mappings_stride0 (int): Stride of the first dimension of slot_mappings (memory layout)
        - cp_rank (int): Current device rank in column-parallel (CP) group
        - CP_SIZE (int): Total devices in CP parallel group
        - CP_INTERLEAVE (bool): Enable interleaved CP computation (memory access optimization)
        - PAD_ID (int): Padding value for invalid slot IDs (-1)
        - TRITON_BLOCK_SIZE (int): Block size for Triton kernel execution (hardware optimization),
                'TOTAL_BLOCK_SIZE' must be greater than the 'position / (block_size * CP_SIZE) + 1024'

    Output:
        - slot_mappings (torch.Tensor): [num_kv_cache_groups, max_num_batched_tokens], int32 → Output slot ID tensor
    """

    torch.manual_seed(42)

    device = "npu" if torch.npu.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    max_num_batched_tokens = 8192
    idx_mapping = torch.tensor([63], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 5], dtype=torch.int32, device=device)
    positions = torch.tensor([0, 1, 2, 3, 4, 0, 0, 0], dtype=torch.int64, device=device)

    num_kv_cache_groups = 1
    max_num_reqs = 64
    max_num_blocks = 320
    block_tables: list[torch.Tensor] = []
    for i in range(num_kv_cache_groups):
        block_table = torch.randint(0, 320, (max_num_reqs, max_num_blocks), dtype=torch.int32, device=device)
        block_tables.append(block_table)
    block_table_ptrs = torch.tensor([t.data_ptr() for t in block_table], dtype=torch.uint64, device=device)
    block_table_strides = torch.tensor([320], dtype=torch.int32, device=device)

    block_sizes_tensor = torch.tensor([128], dtype=torch.int32, device=device)
    slot_mappings = torch.zeros(size=(1, 8192), dtype=torch.int64, device=device)
    ref_slot_mappings = torch.zeros(size=(1, 8192), dtype=torch.int64, device=device)
    cp_rank = 0
    cp_size = 1
    cp_interleave = 1
    num_reqs = query_start_loc.shape[0] - 1
    num_groups = num_kv_cache_groups

    try:
        ascend_compute_slot_mappings_kernel[(num_groups, num_reqs + 1)](
            max_num_batched_tokens,
            idx_mapping,
            query_start_loc,
            positions,
            block_table_ptrs,
            block_table_strides,
            block_sizes_tensor,
            slot_mappings,
            slot_mappings.stride(0),
            cp_rank,
            CP_SIZE=cp_size,
            CP_INTERLEAVE=cp_interleave,
            PAD_ID=-1,
            TRITON_BLOCK_SIZE=1024,  # type: ignore
            TOTAL_BLOCK_SIZE=4096,
        )

        ref_compute_slot_mappings_kernel[(num_groups, num_reqs + 1)](
            max_num_batched_tokens,
            idx_mapping,
            query_start_loc,
            positions,
            block_table_ptrs,
            block_table_strides,
            block_sizes_tensor,
            ref_slot_mappings,
            ref_slot_mappings.stride(0),
            cp_rank,
            CP_SIZE=cp_size,
            CP_INTERLEAVE=cp_interleave,
            PAD_ID=-1,
            TRITON_BLOCK_SIZE=1024,  # type: ignore
        )

        # ========== Verify results ==========
        assert torch.equal(slot_mappings, ref_slot_mappings), (
            f"ascend output differs from gpu reference.\n"
            f"Max diff: {torch.max(torch.abs(slot_mappings - ref_slot_mappings))}\n"
            f"Mean diff: {torch.mean(torch.abs(slot_mappings - ref_slot_mappings).float())}"
        )

    except Exception as e:
        print(f"Error during executionm: {e}")
        import traceback

        traceback.print_exc()
