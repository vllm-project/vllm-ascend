# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run with torchrun --nproc-per-node=2/4/8 -m pytest --noconftest (A3/A5)."""

import gc
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist

pytest.importorskip("torch_npu")
import torch_npu  # noqa: E402
from vllm.distributed.parallel_state import (  # noqa: E402
    destroy_distributed_environment,
    init_distributed_environment,
    init_model_parallel_group,
)

from tests.e2e.nightly.single_node.ops.singlecard_ops.triton.test_glm5_next_dcp_indexer import (  # noqa: E402
    MAX_LEN,
    POOL,
    make_inputs,
    reference,
    require_dtype,
    to_device,
    update_inputs,
)
from vllm_ascend.attention import indexer_kpool  # noqa: E402
from vllm_ascend.attention.context_parallel.kpool_pcp import (  # noqa: E402
    build_kpool_pcp_query_layout,
    gather_kpool_pcp_queries,
    restore_kpool_pcp_queries,
)
from vllm_ascend.ops.triton.glm5_next_dcp_indexer import glm5_next_dcp_indexer  # noqa: E402

RANKS = int(os.environ.get("WORLD_SIZE", "1"))
pytestmark = pytest.mark.skipif(RANKS not in (2, 4, 8), reason="requires isolated DCP2/4/8")


@pytest.fixture(scope="module")
def context():
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    init_distributed_environment(
        world_size=RANKS, rank=rank, local_rank=local_rank, distributed_init_method="env://", backend="hccl"
    )
    group = init_model_parallel_group(
        [list(range(RANKS))],
        local_rank=local_rank,
        backend="hccl",
        group_name="glm_kpool_dcp",
        use_device_communicator=False,
    )
    retained = []
    yield rank, group, retained
    torch.npu.synchronize()
    dist.barrier(group=group.cpu_group)
    for graph, *_ in retained:
        graph.reset()
    retained.clear()
    gc.collect()
    group.destroy()
    destroy_distributed_environment()


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("interleave", [128, 512])
def test_real_candidate_collectives_and_graph(context, quantized, interleave):
    rank, group, retained = context
    require_dtype(quantized)
    device = to_device(make_inputs(RANKS, rank, interleave, quantized))
    query, cache, weights, scales, metadata, _ = device

    def run():
        return glm5_next_dcp_indexer(
            (query.float() * weights.float().unsqueeze(-1)).sum(1).contiguous(),
            cache,
            metadata,
            group,
            topk=2048,
            pool_size=POOL,
            interleave=interleave,
            max_model_len=MAX_LEN,
            scale_cache=scales,
        )

    for _ in range(3):
        run()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = run()
    for step in range(3):
        host = make_inputs(RANKS, rank, interleave, quantized, step)
        update_inputs(device, host)
        graph.replay()
        expected = reference(host[0], host[2], host[4], 2048, step)
        torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
    retained.append((graph, device, result))


@pytest.mark.parametrize("full_dcp", [False, True], ids=["dcp_equals_pcp", "dcp_equals_tp_pcp"])
@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("layout_case", ["mixed", "empty", "prompt1_mtp5", "prompt2_mtp5"])
def test_joint_pcp_dcp_candidate_order(context, monkeypatch, full_dcp, quantized, layout_case):
    rank, world_group, _ = context
    require_dtype(quantized)
    pcp, tp = 2, RANKS // 2
    pcp_ranks = [[t + p * tp for p in range(pcp)] for t in range(tp)]
    # Logical DCP rank order is PCP-first; HCCL gathers in global-rank order.
    # Global candidate IDs must make the merge independent of that difference.
    dcp_ranks = [[r for ranks in pcp_ranks for r in ranks]] if full_dcp else pcp_ranks
    groups = []
    try:
        for name, ranks in (("glm_kpool_joint_pcp", pcp_ranks), ("glm_kpool_joint_dcp", dcp_ranks)):
            groups.append(
                init_model_parallel_group(
                    ranks,
                    local_rank=int(os.environ["LOCAL_RANK"]),
                    backend="hccl",
                    group_name=name,
                    use_device_communicator=False,
                )
            )
        pcp_group, dcp_group = groups
        monkeypatch.setattr(indexer_kpool, "get_pcp_group", lambda: pcp_group)
        monkeypatch.setattr(indexer_kpool, "get_dcp_group", lambda: dcp_group)
        local_rows = {
            "mixed": [[0, 1, 4, 5, 7], [0, 2, 3, 6]],
            "empty": [list(range(8)), []],
            # Actual PCP layouts for one/two remaining prompt tokens + MTP5:
            # the six-row extend is replicated, the seven-row extend is split.
            "prompt1_mtp5": [list(range(6)), list(range(6))],
            "prompt2_mtp5": [[0, 1, 6], [2, 3, 4, 5]],
        }[layout_case]
        rows = max(row for chunk in local_rows for row in chunk) + 1
        width = max(map(len, local_rows)) + 1
        padded = torch.zeros(pcp, width, dtype=torch.long)
        restore = torch.full((rows,), -1, dtype=torch.long)
        for p, chunk in enumerate(local_rows):
            padded[p, : len(chunk)] = torch.tensor(chunk, dtype=torch.long)
            for offset, row in enumerate(chunk):
                if restore[row] < 0:
                    restore[row] = p * width + offset
        assert (restore >= 0).all()
        actual = len(local_rows[pcp_group.rank_in_group])
        layout = build_kpool_pcp_query_layout(
            SimpleNamespace(num_input_tokens=width, num_actual_tokens=actual),
            SimpleNamespace(
                padded_gather_idx=padded.flatten().npu(),
                hidden_restore_idx=restore.npu(),
                global_batch=SimpleNamespace(num_tokens=rows),
            ),
            pcp_group,
        )
        for step in range(3):
            host = make_inputs(dcp_group.world_size, dcp_group.rank_in_group, 128, quantized, step)
            if layout_case.startswith("prompt"):
                host[4].cum_query_lens = torch.tensor([rows], dtype=torch.int32)
                host[4].positions[:rows] = torch.arange(1536, 1536 + rows)
                host[4].seq_lens = torch.tensor([(1536 + rows) // POOL], dtype=torch.int32)
                host[4].block_table = host[4].block_table[:1]
            query, cache, weights, scales, metadata, _ = to_device(host)
            # All ranks, including a rank without local queries, score the same
            # canonical request rows against their own compressed cache shard.
            global_query = gather_kpool_pcp_queries(query[layout.local_indices], layout, pcp_group)
            global_weights = gather_kpool_pcp_queries(weights[layout.local_indices], layout, pcp_group).bfloat16()
            selected = glm5_next_dcp_indexer(
                (global_query.float() * global_weights.float().unsqueeze(-1)).sum(1).contiguous(),
                cache,
                metadata,
                dcp_group,
                topk=2048,
                pool_size=POOL,
                interleave=128,
                max_model_len=MAX_LEN,
                scale_cache=scales,
            )
            result = restore_kpool_pcp_queries(selected, layout, width, fill_value=-1).cpu()
            reference_rows = reference(host[0], host[2], host[4], 2048, step)
            expected = torch.full((width, 1, 2048 + POOL - 1), -1, dtype=torch.int32)
            expected[:actual] = reference_rows[local_rows[pcp_group.rank_in_group]]
            torch.testing.assert_close(result, expected, rtol=0, atol=0)

            # Compare the production reduce-before-gather dispatch with the old
            # ordering using 64 heads, nontrivial weight rounding, and actual
            # NPU FP8 query quantization. With torchrun=4 and full_dcp this is
            # TP2/PCP2/DCP4, including both short-prefill + MTP5 layouts.
            generator = torch.Generator().manual_seed(21835 + step)
            query = torch.randn(rows, 64, 128, generator=generator).bfloat16().npu()[layout.local_indices]
            weights = torch.randn(rows, 64, generator=generator).npu()[layout.local_indices]
            if quantized:
                quant, scale = torch_npu.npu_dynamic_quant(query, dst_type=torch.float8_e4m3fn)
                query = quant.float() * scale.unsqueeze(-1)
            old_query = gather_kpool_pcp_queries(query, layout, pcp_group)
            old_weights = gather_kpool_pcp_queries(weights, layout, pcp_group).bfloat16()
            old_qbar = (old_query.float() * old_weights.float().unsqueeze(-1)).sum(1).contiguous()
            old_selected = glm5_next_dcp_indexer(
                old_qbar,
                cache,
                metadata,
                dcp_group,
                topk=2048,
                pool_size=POOL,
                interleave=128,
                max_model_len=MAX_LEN,
                scale_cache=scales,
            )
            shared = torch.full((width + 1, 2048 + POOL + 3), -123, dtype=torch.int32, device="npu")
            backend = SimpleNamespace(
                topk_tokens=2048,
                index_kpool=POOL,
                _dcp_interleave=128,
                _max_model_len=MAX_LEN,
                topk_indices_buffer=shared,
            )
            local_metadata = SimpleNamespace(pcp_dcp_query_layout=layout, write_metadata=metadata)
            gather = Mock(wraps=pcp_group.all_gather)
            with monkeypatch.context() as patch:
                patch.setattr(pcp_group, "all_gather", gather)
                result = indexer_kpool.Glm5NextKPoolIndexerBackend._select_dcp_candidates(
                    backend,
                    query,
                    weights,
                    cache,
                    local_metadata,
                    scales,
                )
            gather.assert_called_once()
            sent = gather.call_args.args[0]
            assert sent.shape == (width, 128) and sent.dtype == torch.float32
            new_qbar = gather_kpool_pcp_queries(sent, layout, pcp_group)
            torch.testing.assert_close(new_qbar, old_qbar, rtol=0, atol=0)
            expected = restore_kpool_pcp_queries(old_selected, layout, width, fill_value=-1)
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
            assert result.data_ptr() == shared.data_ptr()
            assert (shared[-1].cpu() == -123).all() and (shared[:, 2048 + POOL - 1 :].cpu() == -123).all()
    finally:
        torch.npu.synchronize()
        dist.barrier(group=world_group.cpu_group)
        for group in reversed(groups):
            group.destroy()
