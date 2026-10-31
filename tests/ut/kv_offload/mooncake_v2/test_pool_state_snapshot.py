# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_ascend.core.kv_cache_interface import AscendIndexerKPoolTailSpec
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker import MooncakePullRecvingThread
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.cache_layout import describe_cache_bindings
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_cache_layout import PoolCacheLayout
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.restore_planner import RestorePlanner, SourceCacheSlice
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.state_snapshot import (
    bind_circular_snapshot,
    bind_mamba_checkpoint,
    plan_mamba_restore,
)

from .helpers import make_pp_metadata


def tail(capacity, padding=0):
    spec = AscendIndexerKPoolTailSpec(
        block_size=capacity,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
        sliding_window=4,
        compress_ratio=4,
    )
    storage = torch.full((2, capacity + padding, 4), -1.0)
    tensor = storage[:, :capacity]
    config = SimpleNamespace(
        num_blocks=2,
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=["model.layers.1.tail"],
                kv_cache_spec=spec,
            )
        ],
    )
    binding = describe_cache_bindings(config, {"model.layers.1.tail": tensor})["model.layers.1.tail"][0]
    return storage, tensor, binding


def source_ranges(targets, prefix="source"):
    return [
        SourceCacheSlice(item.cache, f"{prefix}/{index}", item.cache.span_bytes, 0)
        for index, item in enumerate(targets)
    ]


def execute(plan, source_addresses):
    for item in plan:
        ctypes.memmove(item.destination, source_addresses[item.key] + item.source_offset, item.size)


@pytest.mark.parametrize("capacity,boundary", [(4, 7), (7, 8), (7, 15)])
def test_tail_uses_v2_whole_page_rule_with_independent_page_stride(capacity, boundary):
    _, p, p_binding = tail(capacity, 1)
    storage, d, d_binding = tail(capacity, 3)
    p[1].copy_(torch.arange(p[1].numel()).reshape_as(p[1]))
    source = bind_circular_snapshot(p_binding, [1], boundary)
    target = bind_circular_snapshot(d_binding, [1], boundary)
    plan = RestorePlanner().build(source_ranges([source]), [target])
    execute(plan, {"source/0": source.address})
    torch.testing.assert_close(d[1], p[1])
    assert torch.all(storage[:, capacity:] == -1)
    assert torch.all(d[0] == -1)


def test_tail_rejects_mtp_capacity_conversion_like_v2():
    _, _, p = tail(4)
    _, _, d = tail(7)
    source = bind_circular_snapshot(p, [1], 7)
    target = bind_circular_snapshot(d, [1], 7)
    with pytest.raises(ValueError):
        RestorePlanner().build(source_ranges([source]), [target])
    with pytest.raises(ValueError, match="exactly one"):
        bind_circular_snapshot(p, [0, 1], 7)


def kda(local_heads, lookahead, *, recover=False, mode="none"):
    head_dim, history = 2, 3
    shapes = ((history + lookahead, 3 * local_heads * head_dim), (local_heads, head_dim, head_dim))
    dtypes = (torch.bfloat16, torch.float32)
    if recover:
        shapes += ((local_heads, lookahead + 1, head_dim), (local_heads, lookahead + 1, head_dim * 2))
        dtypes += (torch.float32, torch.bfloat16)
    spec = MambaSpec(
        block_size=4,
        shapes=shapes,
        dtypes=dtypes,
        mamba_cache_mode=mode,
        num_speculative_blocks=0 if recover else lookahead,
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    )
    tensors = tuple(torch.full((4, *shape), -1.0, dtype=dtype) for shape, dtype in zip(shapes, dtypes, strict=True))
    config = SimpleNamespace(
        num_blocks=4,
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=["model.layers.0.linear_attn"],
                kv_cache_spec=spec,
            )
        ],
    )
    components = describe_cache_bindings(config, {"model.layers.0.linear_attn": tensors})["model.layers.0.linear_attn"]
    return spec, tensors, components


@pytest.mark.parametrize("lookahead,recover", [(0, False), (3, False), (3, True)])
def test_mamba_pool_restore_uses_v2_projection_and_full_component_policy(lookahead, recover):
    p_spec, p, p_components = kda(2, lookahead, recover=recover)
    d_spec, d, d_components = kda(1, lookahead, recover=recover, mode="align")
    for tensor in p:
        tensor[1].copy_(torch.arange(tensor[1].numel()).reshape_as(tensor[1]))
    source = bind_mamba_checkpoint(p_components, p_spec, [1], 9, rank=0, size=1)
    # Align column floor((9-1)/4)=2 selects block2, not a speculative tail.
    target = bind_mamba_checkpoint(d_components, d_spec, [0, 1, 2, 3], 9, rank=1, size=2)
    plan = plan_mamba_restore(source_ranges(source), target)
    execute(plan, {f"source/{index}": item.address for index, item in enumerate(source)})
    for projection in range(3):
        torch.testing.assert_close(
            d[0][2, :, projection * 2 : (projection + 1) * 2], p[0][1, :, projection * 4 + 2 : (projection + 1) * 4]
        )
    for d_tensor, p_tensor in zip(d[1:], p[1:], strict=True):
        torch.testing.assert_close(d_tensor[2, 0], p_tensor[1, 1])
        assert torch.all(d_tensor[:2] == -1)


def test_mamba_pool_gathers_all_v2_source_shards_before_reporting_success():
    d_spec, d, d_components = kda(2, 3)
    target = bind_mamba_checkpoint(d_components, d_spec, [1], 7, rank=0, size=1)
    sources, addresses, keepalive = [], {}, []
    for rank in range(2):
        spec, tensors, components = kda(1, 3)
        keepalive.append(tensors)
        for tensor in tensors:
            tensor[1].fill_(rank + 3)
        bound = bind_mamba_checkpoint(components, spec, [1], 7, rank=rank, size=2)
        sources.extend(source_ranges(bound, prefix=str(rank)))
        addresses.update({f"{rank}/{index}": item.address for index, item in enumerate(bound)})
    execute(plan_mamba_restore(sources, target), addresses)
    for projection in range(3):
        assert torch.all(d[0][1, :, projection * 4 : projection * 4 + 2] == 3)
        assert torch.all(d[0][1, :, projection * 4 + 2 : (projection + 1) * 4] == 4)
    assert torch.all(d[1][1, 0] == 3)
    assert torch.all(d[1][1, 1] == 4)
    with pytest.raises(ValueError, match="coverage"):
        plan_mamba_restore(sources[:2], target)


@pytest.mark.parametrize("target_lookahead,target_recover", [(3, False), (3, True)])
def test_mamba_keeps_v2_mtp_geometry_compatibility_checks(target_lookahead, target_recover):
    p_spec, _, p_components = kda(1, 0)
    d_spec, _, d_components = kda(1, target_lookahead, recover=target_recover)
    source = bind_mamba_checkpoint(p_components, p_spec, [1], 7, rank=0, size=1)
    target = bind_mamba_checkpoint(d_components, d_spec, [1], 7, rank=0, size=1)
    with pytest.raises(ValueError, match="metadata"):
        plan_mamba_restore(source_ranges(source), target)


@pytest.mark.parametrize("p_tp,p_pcp,d_tp,d_pcp", [(1, 4, 2, 1), (2, 1, 1, 4), (1, 4, 4, 1), (4, 1, 1, 4)])
@pytest.mark.parametrize("lookahead", [0, 3])
def test_pool_kda_worker_shards_match_v2_transfer_addresses(p_tp, p_pcp, d_tp, d_pcp, lookahead):
    """Compare complete plans, including PCP-to-TP rank selection, with V2.

    Both paths describe the same CPU state buffers. No transport or NPU is
    constructed; this proves address planning, not model execution support.
    """
    layer = "model.layers.0.linear_attn"

    def bind_worker(spec, components, tp, pcp, tp_rank, pcp_rank, blocks):
        worker = PoolCacheLayout(
            cache_config=SimpleNamespace(kv_cache_groups=[None]),
            specs={layer: spec},
            bindings={layer: components},
            tp_size=tp,
            pcp_size=pcp,
            tp_rank=tp_rank,
            pcp_rank=pcp_rank,
            dcp_rank=0,
            dcp_size=1,
            total_num_kv_heads=None,
            pp_rank=0,
            base_layers=1,
            pool_head_rank=tp_rank,
            pool_layer_offset=0,
            data_plane="gva",
        )
        return worker.build_target_slices((blocks,), 0, 9)

    sources, source_addresses, keepalive, bases = [], {}, [], {}
    for pcp_rank in range(p_pcp):
        bases[pcp_rank] = {}
        for tp_rank in range(p_tp):
            spec, tensors, components = kda(8 // (p_tp * p_pcp), lookahead)
            keepalive.append(tensors)
            bound = bind_worker(spec, components, p_tp, p_pcp, tp_rank, pcp_rank, [1])
            prefix = f"pcp{pcp_rank}/tp{tp_rank}"
            sources.extend(source_ranges(bound, prefix))
            source_addresses.update({f"{prefix}/{index}": item.address for index, item in enumerate(bound)})
            bases[pcp_rank][tp_rank] = [[component.address for component in components]]
    remote = make_pp_metadata(
        layer_names=[layer],
        block_shapes=[[component.shape[1:] for component in components]],
        block_strides=[[component.byte_strides[0] for component in components]],
        block_lens=[[component.page_span for component in components]],
        pcp_tp_base_addrs=bases,
    )
    for pcp_rank in range(d_pcp):
        for tp_rank in range(d_tp):
            spec, tensors, components = kda(8 // (d_tp * d_pcp), lookahead, mode="align")
            keepalive.append(tensors)
            targets = bind_worker(spec, components, d_tp, d_pcp, tp_rank, pcp_rank, [0, 1, 2, 3])
            plan = plan_mamba_restore(sources, targets)
            thread = MooncakePullRecvingThread.__new__(MooncakePullRecvingThread)
            thread.tp_size, thread.pcp_size = d_tp, d_pcp
            thread.tp_rank, thread.pcp_rank = tp_rank, pcp_rank
            thread.layer_names = [layer]
            thread.kv_caches_base_addr = [[component.address for component in components]]
            thread.block_shapes = [[component.shape[1:] for component in components]]
            thread.block_strides = [[component.byte_strides[0] for component in components]]
            thread.block_lens = [[component.page_span for component in components]]
            local_addresses, remote_addresses, lengths = [], [], []
            groups = thread._get_layer_remote_tp_rank_groups(0, 0, spec, remote, p_pcp, p_tp, 1)
            for (remote_rank,) in groups:
                remote_tp, remote_pcp = divmod(remote_rank, p_pcp)
                thread._append_mamba_transfer_addresses(
                    spec,
                    remote_pcp,
                    remote_tp,
                    p_tp,
                    {(0, 0): [("request", [2], [1])]},
                    remote,
                    local_addresses,
                    remote_addresses,
                    lengths,
                )
            assert sorted(
                (item.destination, source_addresses[item.key] + item.source_offset, item.size) for item in plan
            ) == sorted(zip(local_addresses, remote_addresses, lengths, strict=True))
