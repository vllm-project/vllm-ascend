# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A2/A3 operator regressions; simulate each DCP shard on a single NPU."""

import pytest
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

enable_custom_op()


@pytest.fixture(autouse=True)
def require_arch22():
    soc = torch_npu.npu.get_soc_version()
    if not (220 <= soc <= 225 or 250 <= soc <= 256):
        pytest.skip("DCP causal-coordinate kernels require an A2/A3 NPU")


def owned_groups(raw_end, ratio, size, rank):
    # Enumerate global groups, independently of the kernel's local-count formula.
    return [j for j in range(max(raw_end, 0) // ratio) if (j * ratio // 128) % size == rank]


def check_indexer_rows(indices, lengths, query_lengths, size, rank):
    indices = indices.cpu().squeeze(1)
    offset = 0
    for length, query_length in zip(lengths, query_lengths):
        for q in range(query_length):
            visible = len(owned_groups(length - query_length + q + 1, 4, size, rank))
            row = indices[offset + q]
            valid = row[row >= 0].tolist()
            # All visible entries fit in Top512. This detects both future
            # leakage and the old masking of valid history on other ranks.
            assert sorted(valid) == list(range(visible))
            assert (row[visible:] == -1).all()
        offset += query_length


@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize(
    ("lengths", "query_lengths"),
    [
        ((8192, 1029, 0), (1, 1, 0)),
        ((8192, 1029, 0), (8, 4, 0)),
        ((8192, 1029, 0), (4096, 129, 0)),
        ((1024, 8, 0), (1024, 8, 0)),
        ((129, 8, 0), (129, 8, 0)),
    ],
)
@torch.inference_mode()
def test_dcp_indexer_causal_prefix(rank, lengths, query_lengths):
    cu_q = torch.tensor(
        [0, query_lengths[0], sum(query_lengths[:2]), sum(query_lengths)], dtype=torch.int32, device="npu"
    )
    local_lengths = [len(owned_groups(length, 4, 8, rank)) for length in lengths]
    residuals = [length % 4 if length and ((length - 1) // 128) % 8 == rank else 0 for length in lengths]
    context = torch.tensor([8, rank, 128, *lengths], dtype=torch.int32, device="npu")
    seqused_k = torch.tensor(local_lengths, dtype=torch.int32, device="npu")
    residual = torch.tensor(residuals, dtype=torch.int32, device="npu")
    query = torch.ones((sum(query_lengths), 64, 128), dtype=torch.int8, device="npu")
    weights = torch.ones(query.shape[:2], dtype=torch.float16, device="npu")
    # Two 128-entry physical pages per request; distinct request block tables.
    key = torch.ones((6, 128, 1, 128), dtype=torch.int8, device="npu")
    scales = torch.ones((6, 128, 1), dtype=torch.float16, device="npu")
    table = torch.arange(6, dtype=torch.int32, device="npu").view(3, 2)
    metadata = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
        64,
        1,
        128,
        512,
        2,
        cu_seqlens_q=cu_q,
        seqused_k=seqused_k,
        cmp_residual_k=residual,
        batch_size=3,
        max_seqlen_q=max(query_lengths),
        max_seqlen_k=256,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=4,
        dcp_context=context,
    )
    indices, _ = torch.ops._C_ascend.npu_quant_lightning_indexer_v2(
        query,
        key,
        weights,
        weights,
        scales,
        512,
        2,
        cu_seqlens_q=cu_q,
        seqused_k=seqused_k,
        cmp_residual_k=residual,
        block_table=table,
        metadata=metadata,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=4,
        dcp_context=context,
    )
    check_indexer_rows(indices, lengths, query_lengths, 8, rank)


@pytest.mark.parametrize("rank", [0, 7])
@torch.inference_mode()
def test_dcp_indexer_graph_replay_updates_causal_prefix(rank):
    query = torch.ones((8, 64, 128), dtype=torch.int8, device="npu")
    weights = torch.ones((8, 64), dtype=torch.float16, device="npu")
    key = torch.ones((3, 128, 1, 128), dtype=torch.int8, device="npu")
    scales = torch.ones((3, 128, 1), dtype=torch.float16, device="npu")
    table = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="npu")
    cu_q = torch.tensor([0, 8], dtype=torch.int32, device="npu")
    context = torch.tensor([8, rank, 128, 8191], dtype=torch.int32, device="npu")
    seqused_k = torch.tensor([len(owned_groups(8191, 4, 8, rank))], dtype=torch.int32, device="npu")
    residual = torch.tensor([3 if rank == 7 else 0], dtype=torch.int32, device="npu")

    def run():
        metadata = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
            64,
            1,
            128,
            512,
            2,
            cu_seqlens_q=cu_q,
            seqused_k=seqused_k,
            cmp_residual_k=residual,
            batch_size=1,
            max_seqlen_q=8,
            max_seqlen_k=384,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=3,
            cmp_ratio=4,
            dcp_context=context,
        )
        return torch.ops._C_ascend.npu_quant_lightning_indexer_v2(
            query,
            key,
            weights,
            weights,
            scales,
            512,
            2,
            cu_seqlens_q=cu_q,
            seqused_k=seqused_k,
            cmp_residual_k=residual,
            block_table=table,
            metadata=metadata,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=3,
            cmp_ratio=4,
            dcp_context=context,
        )[0]

    run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        indices = run()
    for length in (8191, 8192, 8193, 8320):
        context[-1] = length
        seqused_k[0] = len(owned_groups(length, 4, 8, rank))
        residual[0] = length % 4 if ((length - 1) // 128) % 8 == rank else 0
        graph.replay()
        check_indexer_rows(indices, (length,), (8,), 8, rank)


@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize("query_length", [1, 4, 8])
@pytest.mark.parametrize("length", [129, 8192])
@torch.inference_mode()
def test_dcp_c128_attention_local_read_bound(rank, query_length, length):
    context = torch.tensor([8, rank, 128, length], dtype=torch.int32, device="npu")
    query = torch.zeros((query_length, 64, 512), dtype=torch.bfloat16, device="npu")
    # SharedKV requires one FP32 sink logit per Q head. -inf contributes
    # zero softmax mass, preserving the uniform-over-KV reference below.
    sinks = torch.full((query.shape[1],), float("-inf"), dtype=torch.float32, device="npu")
    num_ori_blocks = (length + 127) // 128
    ori = torch.ones((num_ori_blocks, 128, 1, 512), dtype=torch.bfloat16, device="npu")
    # Unused local slots are canaries: a global compressed length must never
    # authorize reading them. The last stored entry may also be in the future
    # of an earlier MTP query, although it is valid for the final query.
    cmp = torch.full((1, 128, 1, 512), 1024, dtype=torch.bfloat16, device="npu")
    groups = owned_groups(length, 128, 8, rank)
    for local, global_group in enumerate(groups):
        cmp[0, local].fill_(512 + global_group * 4)
    cu_q = torch.tensor([0, query_length], dtype=torch.int32, device="npu")
    seq_lens = torch.tensor([length], dtype=torch.int32, device="npu")
    cu_ori = torch.tensor([0, length], dtype=torch.int32, device="npu")
    cu_cmp = torch.tensor([0, len(groups)], dtype=torch.int32, device="npu")
    ori_table = torch.arange(num_ori_blocks, dtype=torch.int32, device="npu").view(1, num_ori_blocks)
    cmp_table = torch.zeros((1, 1), dtype=torch.int32, device="npu")

    def run():
        metadata = torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata(
            64,
            1,
            512,
            cu_seqlens_q=cu_q,
            cu_seqlens_ori_kv=cu_ori,
            cu_seqlens_cmp_kv=cu_cmp,
            seqused_kv=seq_lens,
            batch_size=1,
            max_seqlen_q=query_length,
            max_seqlen_kv=length,
            cmp_ratio=128,
            cmp_topk=0,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=127,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
            dcp_context=context,
        )
        return torch.ops._C_ascend.npu_sparse_attn_sharedkv(
            query,
            ori_kv=ori,
            cmp_kv=cmp,
            ori_block_table=ori_table,
            cmp_block_table=cmp_table,
            cu_seqlens_q=cu_q,
            cu_seqlens_ori_kv=cu_ori,
            cu_seqlens_cmp_kv=cu_cmp,
            seqused_kv=seq_lens,
            sinks=sinks,
            metadata=metadata,
            softmax_scale=512**-0.5,
            cmp_ratio=128,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=127,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
            dcp_context=context,
        )[0]

    def check(result, current_length):
        expected = []
        for q in range(query_length):
            raw_end = current_length - query_length + q + 1
            visible = owned_groups(raw_end, 128, 8, rank)
            # Q=0 and sinks=-inf give uniform weights over SWA + local C128.
            window = min(raw_end, 128)
            expected.append((window + sum(512 + j * 4 for j in visible)) / (window + len(visible)))
        golden = torch.tensor(expected).view(query_length, 1, 1).expand(query_length, 64, 512)
        torch.testing.assert_close(result.float().cpu(), golden, rtol=0.005, atol=0.15)

    check(run(), length)
    if rank == 7 and query_length == 8 and length == 8192:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
            output = run()
        for replay_length in (8191, 8192, 8191):
            context[-1] = replay_length
            seq_lens[0] = replay_length
            cu_ori[-1] = replay_length
            cu_cmp[-1] = len(owned_groups(replay_length, 128, 8, rank))
            graph.replay()
            check(output, replay_length)
