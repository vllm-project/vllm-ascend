# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HiSparse INT8 Indexer and resolver/refill/Attention integration coverage."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.sfa_hisparse import AscendHiSparseImpl
from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec
from vllm_ascend.hisparse.binding import _quantized_indexer_views, write_hisparse_indexer_c8
from vllm_ascend.hisparse.c8 import pack_c8_components
from vllm_ascend.utils import AscendDeviceType, bootstrap_custom_op_env, get_ascend_device_type


@pytest.fixture(autouse=True)
def _require_arch22():
    bootstrap_custom_op_env()
    if not torch.npu.is_available() or get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3):
        pytest.skip("requires A2/A3")
    import vllm_ascend.vllm_ascend_C  # type: ignore[import-not-found, import-untyped]  # noqa: F401


def test_int8_indexer_strided_store_and_selection():
    torch.manual_seed(29)
    pages, block, width, offset = 32, 128, 128, 256
    spec = AscendSFAIndexerCacheSpec(
        block_size=block,
        num_kv_heads=1,
        head_size=width,
        dtype=torch.int8,
        scale_dim=1,
        scale_dtype=torch.float16,
        cache_sparse_li_c8=True,
    )
    stride = spec.page_size_bytes + 256
    raw = torch.full((offset + pages * stride + 256,), 37, dtype=torch.int8, device="npu")
    cache = raw.as_strided((pages, 1, block, 130), (stride, block * 130, 130, 1), storage_offset=offset)
    key, scale = _quantized_indexer_views(cache, spec)
    values, scales = torch_npu.npu_dynamic_quant(torch.randn(pages * block, width, device="npu", dtype=torch.bfloat16))
    slots = torch.arange(pages * block, dtype=torch.int32, device="npu")
    write_hisparse_indexer_c8(key, scale, values, scales, slots)
    torch.testing.assert_close(key.cpu().reshape(-1, width), values.cpu(), atol=0, rtol=0)
    torch.testing.assert_close(scale.cpu().flatten(), scales.half().cpu().flatten(), atol=0, rtol=0)
    raw_cpu = raw.cpu()
    assert torch.all(raw_cpu[:offset] == 37) and torch.all(raw_cpu[offset + pages * stride :] == 37)
    for page in range(pages):
        assert torch.all(raw_cpu[offset + page * stride + spec.page_size_bytes : offset + (page + 1) * stride] == 37)
    query, query_scale = torch_npu.npu_dynamic_quant(torch.randn(4, 64, width, device="npu", dtype=torch.bfloat16))
    kwargs = dict(
        query=query,
        weights=torch.randn(4, 64, device="npu", dtype=torch.float16),
        query_dequant_scale=query_scale.half(),
        actual_seq_lengths_query=torch.tensor([4], dtype=torch.int32, device="npu"),
        actual_seq_lengths_key=torch.tensor([pages * block], dtype=torch.int32, device="npu"),
        block_table=torch.arange(pages, dtype=torch.int32, device="npu").view(1, -1),
        query_quant_mode=0,
        key_quant_mode=0,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=2048,
        sparse_mode=3,
        pre_tokens=2**63 - 1,
        next_tokens=2**63 - 1,
    )
    actual = torch_npu.npu_quant_lightning_indexer(key=key, key_dequant_scale=scale.squeeze(2), **kwargs)
    expected = torch_npu.npu_quant_lightning_indexer(
        key=key.contiguous(), key_dequant_scale=scale.squeeze(2).contiguous(), **kwargs
    )
    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0, rtol=0)


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("clip_capacity", [False, True])
def test_int8_resolver_refill_attention(capture, clip_capacity):
    """Use the production compact resolver ABI all the way through C8 SFA."""
    from vllm.v1.hisparse.runtime import HiSparseRuntime

    from vllm_ascend.hisparse.cache_ops import HiSparseNPUOps, ResolverWorkspace
    from vllm_ascend.hisparse.host_pool import allocate_mapped_host

    torch.manual_seed(71)
    rows, topk, block, hot_size, width, stride = 4, 2048, 128, 2048, 656, 256
    blocks = hot_size // block + 1
    try:
        host = allocate_mapped_host((blocks * block, width), torch.int8)
    except RuntimeError as exc:
        # ACL_ERROR_RT_FEATURE_NOT_SUPPORT or a successful query with no AIV support.
        if str(exc) in (
            "HiSparse requires AIV host-memory mapping: status=207000, capability=0",
            "HiSparse requires AIV host-memory mapping: status=0, capability=0",
        ):
            pytest.skip(str(exc))
        raise
    mapping = host._hisparse_registration
    nope = torch.randint(-127, 128, (blocks * block, 512), dtype=torch.int8)
    rope = torch.randn(blocks * block, 64, dtype=torch.bfloat16)
    scales = torch.rand(blocks * block, 4) * 0.02 + 0.001
    host.copy_(pack_c8_components(nope, rope, scales).view(torch.int8))
    backing = torch.full((blocks, stride, width), 13, dtype=torch.int8, device="npu")
    cache = backing[:, :block]
    capacity = 2 * stride + 63 if clip_capacity else (blocks - 1) * stride + block
    workspace = ResolverWorkspace(rows, hot_size, hot_size, torch.device("npu", torch.npu.current_device()))
    ops = HiSparseNPUOps(mapping, workspace, attention_capacity=capacity, attention_block_stride=stride)
    indices = torch.empty((rows, topk), dtype=torch.int32, device="npu")
    requests = torch.zeros(rows, dtype=torch.int32, device="npu")
    table = torch.arange(1, blocks, dtype=torch.int32, device="npu")[None]
    resident = torch.zeros_like(table)
    tags = torch.full((1, hot_size), -1, dtype=torch.int32, device="npu")
    lru = torch.arange(hot_size, dtype=torch.int16)[None].npu()
    out, sh, sd = (torch.empty_like(indices) for _ in range(3))
    # A padded output row must not overwrite its neighboring sentinel columns.
    attention_backing = torch.full((rows, topk + 16), -9, dtype=torch.int32, device="npu")
    attention = attention_backing[:, :topk]
    counts, swaps = (torch.empty(rows, dtype=torch.int32, device="npu") for _ in range(2))
    valid_rows = torch.tensor([rows], dtype=torch.int32, device="npu")
    query = torch.randn(rows, 4, 512, dtype=torch.bfloat16)
    query_rope = torch.randn(rows, 4, 64, dtype=torch.bfloat16)
    q, qr = query.npu(), query_rope.npu()
    lengths = tuple(torch.empty(1, dtype=torch.int32, device="npu") for _ in range(2))
    impl = SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64, scale=576**-0.5)
    impl._c8_length_workspace = lambda: lengths
    runtime = HiSparseRuntime.__new__(HiSparseRuntime)
    runtime._host_cache, runtime.cache_ops = host, ops
    runtime.hot = SimpleNamespace(cache=cache, block_size=block, attention_block_stride=stride)
    runtime.hot_block_table = table
    runtime.request_state_indices = torch.zeros(1, dtype=torch.int32, device="npu")
    runtime.region_stride = runtime.max_union_rows = hot_size
    runtime.index_group = SimpleNamespace(
        device_global_indices=tags,
        lru_slots=lru,
        swap_stats=torch.zeros(2, dtype=torch.int64, device="npu"),
        shared_topk=SimpleNamespace(
            device_topk_rows=out,
            physical_topk_indices=attention,
            valid_topk_counts=counts,
            swap_host_physical_rows=sh,
            swap_device_physical_rows=sd,
            swap_counts=swaps,
        ),
    )
    resident_handle = SimpleNamespace(block_table=resident, view=runtime.hot)

    def forward():
        runtime._resolve_residency(
            resident=resident_handle,
            req_id_per_token=requests,
            num_valid_rows=valid_rows,
            block_table=table,
            topk_indices=indices,
            block_size=block,
            return_valid_counts=True,
            shared_rows=slice(0, rows),
        )
        runtime._swap_rows(slice(0, rows))
        return AscendHiSparseImpl._c8_physical_attention(
            impl,
            q,
            qr,
            cache,
            attention,
            indices_are_mapped=True,
        )

    graph = None
    latent = (nope.float() * scales.repeat_interleave(128, dim=-1)).bfloat16().float()
    keys = torch.cat((latent, rope.float()), dim=-1)
    queries = torch.cat((query.float(), query_rope.float()), dim=-1)
    try:
        for phase, live in enumerate((rows, rows - 1, 0, rows)):
            logical = (torch.arange(rows * topk).view(rows, topk) + phase * 97).to(torch.int32) % 512
            logical[:, 1::3] = -1
            logical[1] = -1
            indices.copy_(logical)
            valid_rows.fill_(live)
            tags.fill_(-1)
            lru.copy_(torch.arange(hot_size, dtype=torch.int16)[None])
            if capture and graph is None:
                actual = forward()
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    actual = forward()
                tags.fill_(-1)
                lru.copy_(torch.arange(hot_size, dtype=torch.int16)[None])
            if graph is None:
                actual = forward()
            else:
                graph.replay()
            torch.npu.synchronize()
            virtual = out.cpu()
            mapped = torch.where(virtual >= 0, virtual // block * stride + virtual % block, -1)
            wanted = torch.full_like(mapped, -1)
            expected = torch.zeros(rows, 4, 512)
            cache_cpu = cache.cpu()
            for row in range(rows):
                keep = (mapped[row] >= 0) & (mapped[row] < capacity)
                selected = mapped[row, keep]
                wanted[row, : selected.numel()] = selected
                assert counts[row].item() == selected.numel()
                if row >= live:
                    assert torch.all(virtual[row] == -1)
                else:
                    assert torch.equal(virtual[row] >= 0, logical[row] >= 0)
                positions = torch.nonzero(virtual[row] >= 0).flatten()
                physical = virtual[row, positions].long()
                host_ids = logical[row, positions].long() + block
                torch.testing.assert_close(cache_cpu[physical // block, physical % block], host[host_ids])
                host_ids = logical[row, keep].long() + block
                if host_ids.numel():
                    scores = queries[row] @ keys[host_ids].T * impl.scale
                    expected[row] = scores.softmax(-1) @ latent[host_ids]
            torch.testing.assert_close(attention.cpu(), wanted)
            assert torch.all(attention_backing[:, topk:].cpu() == -9)
            assert torch.all(backing[:, block:].cpu() == 13)
            torch.testing.assert_close(actual.float().cpu(), expected, atol=0.005, rtol=0.02)
            assert torch.all(workspace.errors[:rows, 0].cpu() == 0)
    finally:
        torch.npu.synchronize()
        if graph is not None:
            graph.reset()
        mapping.close()
