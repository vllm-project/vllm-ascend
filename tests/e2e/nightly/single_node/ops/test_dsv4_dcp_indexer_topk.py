# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Exercise A2 QLI candidates and the production DCP HCCL merge without a model.

Build the changed custom operators, then run on eight idle A2 NPUs:

    torchrun --standalone --nproc_per_node=8 \
        tests/e2e/nightly/single_node/ops/test_dsv4_dcp_indexer_topk.py

Uses 32-row compressed-page striping and K=512. Equal-score cutoffs may choose
other valid IDs than unsharded QLI, so compare selected score multisets as well
as index validity, uniqueness, padding, and agreement between all ranks.
"""

import os

import torch
import torch.distributed as dist
import torch_npu
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_dcp_group,
    init_distributed_environment,
    initialize_model_parallel,
)

from vllm_ascend.attention.context_parallel.dsv4_indexer import gather_dcp_topk
from vllm_ascend.utils import enable_custom_op

BLOCK_SIZE = 32
TOPK = 512
NUM_HEADS = 64
HEAD_DIM = 128


def _cache(global_lengths: list[int], dcp_size: int, dcp_rank: int):
    """Build equal physical page slots, with only owned pages visible locally."""
    max_pages = max(1, (max(global_lengths) + BLOCK_SIZE - 1) // BLOCK_SIZE)
    local_pages = (max_pages + dcp_size - 1) // dcp_size
    batch_size = len(global_lengths)
    # Local page j owns global compressed page j*D+r in every request.
    page_ids = torch.arange(local_pages, dtype=torch.int64) * dcp_size + dcp_rank
    global_rows = page_ids[:, None] * BLOCK_SIZE + torch.arange(BLOCK_SIZE)
    scales = (1 + global_rows.float() / 1024).to(torch.float16)
    scales = scales[None, ..., None].repeat(batch_size, 1, 1, 1).reshape(-1, BLOCK_SIZE, 1)
    keys = torch.zeros((batch_size * local_pages, BLOCK_SIZE, 1, HEAD_DIM), dtype=torch.int8)
    keys[..., 0] = 1
    block_table = torch.arange(batch_size * local_pages, dtype=torch.int32).view(batch_size, local_pages)
    local_lengths = [
        sum(min(BLOCK_SIZE, max(0, length - int(page) * BLOCK_SIZE)) for page in page_ids) for length in global_lengths
    ]
    return keys.npu(), scales.npu(), block_table.npu(), local_lengths


def _qli(global_lengths: list[int], dcp_size: int, dcp_rank: int, return_value: int):
    keys, scales, block_table, local_lengths = _cache(global_lengths, dcp_size, dcp_rank)
    batch_size = len(global_lengths)
    query = torch.zeros((batch_size, NUM_HEADS, HEAD_DIM), dtype=torch.int8, device="npu")
    query[..., 0] = 1
    weights = torch.ones((batch_size, NUM_HEADS), dtype=torch.float16, device="npu")
    query_scale = torch.full_like(weights, 1 / NUM_HEADS)
    cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device="npu")
    seqused_k = torch.tensor(local_lengths, dtype=torch.int32, device="npu")
    cmp_residual_k = torch.zeros(batch_size, dtype=torch.int32, device="npu")
    metadata = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
        num_heads_q=NUM_HEADS,
        num_heads_k=1,
        head_dim=HEAD_DIM,
        topk=TOPK,
        quant_mode=2,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        batch_size=batch_size,
        max_seqlen_q=1,
        max_seqlen_k=max(1, max(local_lengths)) * 4,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=0,
        cmp_ratio=4,
    )
    return torch.ops._C_ascend.npu_quant_lightning_indexer_v2(
        query=query,
        key=keys,
        weights=weights,
        query_dequant_scale=query_scale,
        key_dequant_scale=scales,
        topk=TOPK,
        quant_mode=2,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        block_table=block_table,
        metadata=metadata,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=0,
        cmp_ratio=4,
        return_value=return_value,
    )


def _score(global_rows: torch.Tensor) -> torch.Tensor:
    # Existing QLI computes QK/1024 -> FP16, then applies weights*query_scale
    # and key_scale. These one-hot inputs make that exact scalar arithmetic.
    return (1 + global_rows.float() / 1024).to(torch.float16).float() / 1024


@torch.inference_mode()
def _run_case(global_lengths: list[int]) -> None:
    group = get_dcp_group()
    local_indices, local_scores = _qli(global_lengths, group.world_size, group.rank_in_group, return_value=2)
    assert local_scores.dtype == torch.float32
    local_ids_cpu, local_values_cpu = local_indices.cpu().squeeze(1), local_scores.cpu().squeeze(1)
    for request, length in enumerate(global_lengths):
        valid = local_ids_cpu[request] >= 0
        ids = local_ids_cpu[request, valid]
        owned_global_ids = (
            ids.div(BLOCK_SIZE, rounding_mode="floor") * group.world_size + group.rank_in_group
        ) * BLOCK_SIZE + ids.remainder(BLOCK_SIZE)
        assert torch.all(owned_global_ids < length)
        torch.testing.assert_close(local_values_cpu[request, valid], _score(owned_global_ids), rtol=0, atol=0)
        assert torch.all(torch.isneginf(local_values_cpu[request, ~valid]))
    global_indices = gather_dcp_topk(local_indices, local_scores, BLOCK_SIZE, group.world_size, group.rank_in_group)
    actual = global_indices.cpu().squeeze(1)

    # A reference unsharded QLI is run independently on every rank. Avoid the
    # legacy mode-0 all-empty path: this patch intentionally changes only mode 2.
    reference = _qli(global_lengths, 1, 0, return_value=0)[0].cpu().squeeze(1) if any(global_lengths) else None
    for request, length in enumerate(global_lengths):
        valid_count = min(TOPK, length)
        selected = actual[request, :valid_count]
        assert torch.all((selected >= 0) & (selected < length))
        assert selected.unique().numel() == valid_count
        assert torch.all(actual[request, valid_count:] == -1)
        expected_scores = _score(torch.arange(length)).topk(valid_count).values
        actual_scores = _score(selected).sort(descending=True).values
        torch.testing.assert_close(actual_scores, expected_scores, rtol=0, atol=0)
        if reference is not None:
            reference_indices = reference[request, :valid_count]
            reference_scores = _score(reference_indices).sort(descending=True).values
            torch.testing.assert_close(actual_scores, reference_scores, rtol=0, atol=0)

    # This uses the real HCCL DCP group after the production asynchronous stream
    # join, detecting rank disagreement or unfinished communication on return.
    agreement = torch.empty(
        (group.world_size * global_indices.shape[0], *global_indices.shape[1:]),
        dtype=global_indices.dtype,
        device=global_indices.device,
    )
    dist.all_gather_into_tensor(agreement, global_indices.contiguous(), group=group.device_group)
    for rank_result in agreement.cpu().reshape(group.world_size, *global_indices.shape):
        torch.testing.assert_close(rank_result.squeeze(1), actual, rtol=0, atol=0)
    if group.rank_in_group == 0:
        print(f"DCP{group.world_size} QLI/HCCL TopK passed: global compressed rows={global_lengths}", flush=True)


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch_npu.npu.set_device(local_rank)
    soc_version = torch_npu.npu.get_soc_version()
    assert 220 <= soc_version <= 225 or 250 <= soc_version <= 256, "requires arch22 A2/A3 NPUs"
    assert world_size == 8, "run with torchrun --nproc_per_node=8 for TP8+DCP8"
    enable_custom_op()
    config = VllmConfig(
        parallel_config=ParallelConfig(tensor_parallel_size=world_size, decode_context_parallel_size=world_size)
    )
    with set_current_vllm_config(config):
        try:
            init_distributed_environment(
                world_size=world_size,
                rank=rank,
                local_rank=local_rank,
                distributed_init_method="env://",
                backend="hccl",
            )
            initialize_model_parallel(
                tensor_model_parallel_size=world_size,
                decode_context_model_parallel_size=world_size,
                backend="hccl",
            )
            _run_case([0, 7, 37, 511, 4099])
            _run_case([0, 0, 0])
            torch.npu.synchronize()
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


if __name__ == "__main__":
    main()
