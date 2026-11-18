# SPDX-License-Identifier: Apache-2.0
"""Launch with torchrun -m pytest --noconftest on two or more NPUs."""

import os

import pytest
import torch
import torch.distributed as dist

from tests.e2e.nightly.single_node.ops.multicard_ops_a3.test_sfa_dcp import context  # noqa: F401
from tests.e2e.nightly.single_node.ops.singlecard_ops.test_kimi_packed_cp_cache import map_addresses
from tests.e2e.nightly.single_node.ops.singlecard_ops.test_kv_quant_sparse_flash_attention_vllm import (
    _make_inputs,
    _run_custom_op,
)
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.models.kimi_k3.packed_sfa_cache import packed_sfa_cache_views
from vllm_ascend.ops.triton.sparse_index_remap import remap_sparse_indices_triton

RANKS = int(os.environ.get("WORLD_SIZE", "1"))
pytestmark = pytest.mark.skipif(RANKS not in (2, 4, 8), reason="requires DCP2/4/8")


@pytest.mark.parametrize("pcp", [False, True])
@pytest.mark.parametrize("graph_mode", [False, True])
@torch.inference_mode()
def test_packed_cp_attention_matches_unsharded(context, pcp, graph_mode):  # noqa: F811
    rank, group, retained = context
    inputs = _make_inputs()
    baseline = _run_custom_op(inputs)[0].reshape(1, 64, 512)
    source = inputs["key"].reshape(-1, 656)
    if pcp:
        # PCP producers own contiguous input chunks; KV ownership instead
        # alternates every 128 tokens across the unified DCP group.
        local_input = source.chunk(RANKS)[rank].contiguous()
        gathered = torch.empty_like(source)
        dist.all_gather_into_tensor(gathered, local_input, group=group.device_group)
        source = gathered
    positions = torch.arange(source.shape[0], device="npu")
    owned = positions // 128 % RANKS == rank
    local_source = source[owned]
    manager_blocks = local_source.shape[0] // 256
    spec = AscendMLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=656,
        dtype=torch.int8,
        cache_sparse_sfa_c8=True,
        is_block_outermost=True,
    )
    row_bytes = 4 * 128 * 656
    backing = torch.zeros((manager_blocks + 1) * row_bytes, dtype=torch.int8, device="npu")
    (cache,) = packed_sfa_cache_views(backing, spec, manager_blocks + 1, row_bytes)
    ids = torch.arange(manager_blocks, 0, -1, dtype=torch.int32, device="npu")
    table = (ids[:, None] * 2 + torch.arange(2, dtype=torch.int32, device="npu")).view(1, -1)
    slots = (ids[:, None] * 256 + torch.arange(256, dtype=torch.int32, device="npu")).flatten()
    mapped = map_addresses(spec, row_bytes, table, slots)
    pages, offsets = mapped.slot_mapping // 128, mapped.slot_mapping % 128
    cache[pages.long(), offsets.long(), 0] = local_source
    inputs.update(
        key=cache,
        value=cache,
        block_table=mapped.block_table_tensor,
        sparse_indices=remap_sparse_indices_triton(inputs["sparse_indices"], RANKS, rank, 128),
        actual_seq_lengths_kv=torch.tensor([local_source.shape[0]], dtype=torch.int32, device="npu"),
    )

    def invoke():
        output, maxima, sums = _run_custom_op(inputs, return_softmax_lse=True)
        return torch.ops.vllm.dcp_a2a_fused(
            output.reshape(1, 64, 512),
            maxima.reshape(1, 64, 1),
            1 if pcp else RANKS,
            1,
            group.unique_name,
            group.unique_name if pcp else None,
            softmax_sum=sums.reshape(1, 64, 1),
        )

    actual = invoke()
    graph = None
    if graph_mode:
        for _ in range(3):
            invoke()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual = invoke()
        graph.replay()
    expected = baseline if pcp else baseline.chunk(RANKS, dim=1)[rank]
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0.02, atol=0.03)
    if graph is not None:
        budget = 0.03 + expected.cpu().double().abs() * 0.02
        retained.append((graph, [(actual, expected.cpu(), budget)], (inputs, backing, mapped)))
