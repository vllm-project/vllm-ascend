# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import pytest
import torch
from vllm.config import CacheConfig
from vllm.v1.core import kv_cache_planning, kv_cache_utils
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import CircularBufferSpec

from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSlidingWindowMLASpec,
    register_ascend_kv_cache_specs,
)
from vllm_ascend.models.deepseek_v41.cache_config import (
    LayerPackedKVCacheConfig,
    get_deepseek_v41_kv_cache_config,
    get_deepseek_v41_pool_bytes_per_block,
    group_cache_specs,
    make_cache_groups,
)
from vllm_ascend.worker.v2.utils import LayerPackedKVBlockOps


def _cache_specs(layer_packed=False):
    register_all_kvcache_specs(None)
    register_ascend_kv_cache_specs()
    specs = {}
    for layer in range(40):
        prefix = f"model.layers.{layer}.self_attn"
        specs[prefix + ".swa_cache"] = AscendSlidingWindowMLASpec(
            block_size=128,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
            sliding_window=128,
            model_version="deepseek_v41",
        )
        if layer not in (2, 8, 14, 20):
            continue
        ratio = 1 if layer == 20 else 2
        common = dict(block_size=128, num_kv_heads=1, tokens_per_state=ratio, model_version="deepseek_v41")
        specs[prefix + ".long_kv_cache"] = AscendMLAAttentionSpec(
            **common, head_size=544, dtype=torch.int8, scale_dtype=torch.float32, layer_packed=layer_packed
        )
        specs[prefix + ".indexer.k_cache"] = AscendMLAAttentionSpec(
            **common, head_size=128, dtype=torch.int8, scale_dim=1, scale_dtype=torch.float16
        )
        if ratio == 2:
            specs[prefix + ".compressor.state_cache"] = CircularBufferSpec(
                block_size=32, num_kv_heads=1, head_size=1024, dtype=torch.float32, head_size_v=0
            )
    return specs


def _dummy_context_batch(num_reqs):
    return NS(
        num_reqs=num_reqs,
        max_query_len=1,
        num_tokens=num_reqs,
        num_scheduled_tokens=np.ones(num_reqs, dtype=np.int32),
        seq_lens=torch.ones(num_reqs, dtype=torch.int32),
        seq_lens_cpu_upper_bound=np.ones(num_reqs, dtype=np.int32),
        num_computed_tokens_np=np.zeros(num_reqs, dtype=np.int32),
        num_computed_prefill_tokens_np=np.zeros(num_reqs, dtype=np.int32),
        query_start_loc_np=np.arange(num_reqs + 1),
        positions=torch.zeros(num_reqs, dtype=torch.int64),
    )


@pytest.mark.parametrize("draft_layers", [0, 3])
def test_layer_packed_views_and_capacity(draft_layers):
    from tests.deepseek_v41_utils import allocate_cache_views

    specs = _cache_specs(layer_packed=True)
    for layer in range(draft_layers):
        specs[f"model.mtp.{layer}.swa_cache"] = specs["model.layers.0.self_attn.swa_cache"]
    groups = make_cache_groups(group_cache_specs(dict(reversed(list(specs.items())))))
    assert [len(g.layer_names) for g in groups] == [8, 2, 1] + [2] * 20 + ([2, 1] if draft_layers else [])
    assert groups == make_cache_groups(group_cache_specs(specs))
    assert len({name for g in groups for name in g.layer_names}) == len(specs)
    page = 2 * (128 << 10)
    assert get_deepseek_v41_pool_bytes_per_block(groups) == page
    runtime = NS(cache_config=CacheConfig(prefix_cache_retention_interval=None))
    cfg = get_deepseek_v41_kv_cache_config(runtime, groups, 3 * page + page - 1)
    assert isinstance(cfg, LayerPackedKVCacheConfig)
    assert cfg.num_blocks == 3 and cfg.needs_kv_cache_zeroing
    assert len(cfg.page_owners) == 2
    assert all(d.size == 3 * page and d.block_stride == page // 2 for d in cfg.kv_cache_tensors)
    assert kv_cache_utils.generate_scheduler_kv_cache_config([cfg]).num_blocks == 3
    (raw,), caches = allocate_cache_views(cfg)
    assert raw.numel() == 3 * page
    for group in groups:
        # The members of one table are simultaneous owners of the same ID.
        page_views = []
        for name in group.layer_names:
            view = caches[name]
            for component in view if isinstance(view, tuple) else (view,):
                assert component.shape[0] == cfg.num_blocks
                assert component.stride(0) * component.element_size() == page // 2
                page_views.append(component)
        for i, view in enumerate(page_views, 1):
            view[-1].fill_(i)
        for i, view in enumerate(page_views, 1):
            assert torch.all(view[-1] == i)
            assert torch.count_nonzero(view[:-1]) == 0

    legacy = make_cache_groups(group_cache_specs(_cache_specs()))
    assert get_deepseek_v41_pool_bytes_per_block(legacy) == 4 * (128 << 10)


@pytest.mark.parametrize("component_list", [False, True], ids=["v41-tensor", "glm-kpool-list"])
def test_dummy_ring_clears_assigned_blocks_only(component_list):
    from vllm_ascend.worker.v2.utils import prepare_v41_dummy_ring_state

    num_reqs = 33  # More requests than the 32 token rows inside a V4.1 ring block.
    cache = torch.ones(num_reqs + 3, 32, 1, 8)
    ring = CircularBufferSpec(block_size=32, num_kv_heads=1, head_size=8, dtype=torch.float32, head_size_v=0)
    group = NS(kv_cache_spec=ring, layer_names=["ring"])
    table = torch.full((num_reqs + 1, 1), -1, dtype=torch.int32)
    layer = NS(kv_cache=[cache] if component_list else cache)
    runner = NS(
        kv_cache_config=NS(num_blocks=cache.shape[0], kv_cache_groups=[group]),
        block_tables=NS(input_block_tables=[table]),
        compilation_config=NS(static_forward_context={"ring": layer}),
    )

    prepare_v41_dummy_ring_state(runner, num_reqs)
    assert table[:, 0].tolist() == [*range(1, num_reqs + 1), -1]
    assert torch.all(cache[0] == 1)
    assert torch.all(cache[1 : num_reqs + 1] == 0)
    assert torch.all(cache[num_reqs + 1 :] == 1)


@pytest.mark.parametrize(
    "num_blocks,ring_ids",
    [
        (1, ([], [])),  # Only the null block exists.
        (2, ([], [])),
        (4, ([3], [2])),  # Upstream profiling can allocate just one block per request.
        (8, ([5, 6, 7], [2, 3, 4])),
        (9, ([5, 6, 7, 8], [1, 2, 3, 4])),
    ],
)
def test_small_packed_dummy_pool_separates_ring_writes_and_attention_reads(num_blocks, ring_ids):
    from vllm.v1.worker.gpu.block_table import BlockTables

    from tests.deepseek_v41_utils import allocate_cache_views
    from vllm_ascend.patch.worker.patch_model_runner import set_dummy_context
    from vllm_ascend.worker.v2.model_runner import NPUModelRunner

    groups = make_cache_groups(group_cache_specs(_cache_specs(layer_packed=True)))
    runtime = NS(cache_config=CacheConfig(prefix_cache_retention_interval=None))
    cfg = get_deepseek_v41_kv_cache_config(runtime, groups, num_blocks * (256 << 10))
    (raw,), caches = allocate_cache_views(cfg)
    raw.fill_(1)
    batch = _dummy_context_batch(4)
    tables = [torch.full((4, 3), -1, dtype=torch.int32) for _ in groups]
    provider = object.__new__(BlockTables)
    provider.block_sizes = [group.kv_cache_spec.block_size for group in groups]
    provider.input_block_tables = tables
    provider.v41_shared_ring_groups = (1, 2)
    provider._num_dirty_rows = 0
    provider.slot_mappings = torch.empty(len(groups), 4, dtype=torch.int64)
    runner = object.__new__(NPUModelRunner)
    runner.device = torch.device("cpu")
    runner.pcp_manager = None
    runner.kv_cache_config = cfg
    runner.block_tables = provider
    runner.compilation_config = NS(static_forward_context={n: NS(kv_cache=v) for n, v in caches.items()})
    batch.num_tokens_after_padding = batch.num_tokens
    # Adaptive profiling asks for valid state slots. With zero history the
    # parent must not install 1..N into attention tables of a tiny pool.
    prepared, slots = runner.prepare_dummy_attn(batch, valid_state_slots=True)
    assert all(torch.count_nonzero(t) == 0 for gid, t in enumerate(prepared) if gid not in (1, 2))
    assert torch.all(slots == -1)
    set_dummy_context(batch, provider, 256, num_blocks, 1024, prepared)
    for gid, ids in zip(provider.v41_shared_ring_groups, ring_ids):
        assert tables[gid][:, 0].tolist() == ids + [0] * (batch.num_reqs - len(ids))
        assert torch.all(tables[gid][:, 1:] == 0)
        for name in groups[gid].layer_names:
            assert torch.all(caches[name][ids] == 0)
            assert torch.count_nonzero(caches[name][0]) == caches[name][0].numel()
    read_blocks = num_blocks - sum(map(len, ring_ids))
    expected_reads = torch.arange(12).reshape(4, 3) % read_blocks
    assert all(torch.equal(table, expected_reads) for gid, table in enumerate(tables) if gid not in (1, 2))
    assert batch.seq_lens.tolist() == [257] * batch.num_reqs


@pytest.mark.parametrize("bf16", [True, False], ids=["bf16", "g64"])
def test_shared_unpacked_dummy_context_preserves_ring_pages(bf16):
    from dataclasses import replace

    from tests.deepseek_v41_utils import allocate_cache_views
    from vllm_ascend.patch.worker.patch_model_runner import set_dummy_context
    from vllm_ascend.worker.v2.utils import prepare_v41_dummy_ring_state

    specs = _cache_specs()
    if bf16:
        for name, spec in specs.items():
            if name.endswith(".long_kv_cache"):
                specs[name] = replace(spec, dtype=torch.bfloat16, head_size=512, scale_dtype=torch.int8)
    groups = make_cache_groups(group_cache_specs(specs))
    runtime = NS(cache_config=CacheConfig(prefix_cache_retention_interval=None))
    cfg = get_deepseek_v41_kv_cache_config(runtime, groups, 7 * get_deepseek_v41_pool_bytes_per_block(groups))
    (raw,), caches = allocate_cache_views(cfg)
    raw.fill_(1)
    tables = [torch.zeros((4, 3), dtype=torch.int32) for _ in groups]
    provider = NS(
        input_block_tables=tables, block_sizes=[g.kv_cache_spec.block_size for g in groups], v41_shared_ring_groups=(1,)
    )
    runner = NS(
        kv_cache_config=cfg,
        block_tables=provider,
        compilation_config=NS(static_forward_context={n: NS(kv_cache=v) for n, v in caches.items()}),
    )
    prepare_v41_dummy_ring_state(runner, 4)
    set_dummy_context(_dummy_context_batch(4), provider, 256, 7, 1024)
    assert tables[1][:, 0].tolist() == [3, 4, 5, 6]
    assert all(int(t.max()) < 3 for gid, t in enumerate(tables) if gid != 1)
    for name in groups[1].layer_names:
        ring = caches[name]
        assert torch.count_nonzero(ring[:3]) == ring[:3].numel()
        assert torch.count_nonzero(ring[3:]) == 0


def test_layer_packed_ring_dummy_and_block_copy():
    from vllm.v1.worker.utils import AttentionGroup

    from tests.deepseek_v41_utils import allocate_cache_views
    from vllm_ascend.attention.dsa_v41 import DeepseekV41CacheBackend
    from vllm_ascend.worker.v2.utils import prepare_v41_dummy_ring_state

    groups = make_cache_groups(group_cache_specs(_cache_specs(layer_packed=True)))
    runtime = NS(cache_config=CacheConfig(prefix_cache_retention_interval=None))
    cfg = get_deepseek_v41_kv_cache_config(runtime, groups, 9 * (256 << 10))
    (raw,), caches = allocate_cache_views(cfg)
    layers = {n: NS(kv_cache=v) for n, v in caches.items()}
    tables = [torch.zeros((2, 1), dtype=torch.int32) for _ in groups]
    runner = NS(
        kv_cache_config=cfg,
        block_tables=NS(input_block_tables=tables, v41_shared_ring_groups=(1, 2)),
        compilation_config=NS(static_forward_context=layers),
    )
    raw.fill_(1)
    prepare_v41_dummy_ring_state(runner, 2)
    assert tables[1].flatten().tolist() == [7, 8]
    assert tables[2].flatten().tolist() == [5, 6]
    for g in (1, 2):
        for n in groups[g].layer_names:
            assert torch.count_nonzero(caches[n][tables[g].flatten().long()]) == 0
            assert torch.count_nonzero(caches[n][4]) == caches[n][4].numel()

    owners = [
        AttentionGroup(backend=DeepseekV41CacheBackend, layer_names=[name], kv_cache_spec=spec, kv_cache_group_id=gid)
        for gid, group in enumerate(groups)
        for name, spec in group.kv_cache_spec.kv_cache_specs.items()
    ]
    ops = LayerPackedKVBlockOps(cfg, owners, [g.kv_cache_spec.block_size for g in groups], layers, torch.device("cpu"))
    addrs, strides, sizes, *_ = ops._meta
    slot_stride = (128 << 10) // 4  # The zeroer measures page widths and strides in 32-bit words.
    assert sorted(addrs.tolist()) == [raw.data_ptr(), raw.data_ptr() + cfg.num_blocks * (128 << 10)]
    assert strides.tolist() == sizes.tolist() == [slot_stride, slot_stride]
    slots = raw.view(2, cfg.num_blocks, -1)
    slots[:, 1].fill_(7)
    slots[:, 2].fill_(13)
    ops.copy_blocks([KVCacheBlockCopy(1, 2), KVCacheBlockCopy(2, 1)])
    assert torch.all(slots[:, 1] == 13) and torch.all(slots[:, 2] == 7)
    assert torch.all(slots[:, 0] == 1)


@pytest.mark.parametrize("num_blocks", [1, 9])
def test_pcp_padded_dummy_preserves_live_pages_and_rank_major_slots(num_blocks):
    from tests.deepseek_v41_utils import allocate_cache_views
    from vllm_ascend.patch.worker.patch_model_runner import set_dummy_context
    from vllm_ascend.worker.v2.model_runner import NPUModelRunner
    from vllm_ascend.worker.v2.pcp_manager import AscendPCPManager

    groups = make_cache_groups(group_cache_specs(_cache_specs(layer_packed=True)))
    cfg = get_deepseek_v41_kv_cache_config(NS(cache_config=CacheConfig()), groups, num_blocks * (256 << 10))
    (raw,), caches = allocate_cache_views(cfg)
    raw.fill_(17)
    runner = object.__new__(NPUModelRunner)
    runner.kv_cache_config = cfg
    runner.block_tables = NS(
        v41_shared_ring_groups=(1, 2),
        block_sizes=[g.kv_cache_spec.block_size for g in groups],
        input_block_tables=[torch.full((4, 3), 5, dtype=torch.int32) for _ in groups],
    )
    runner.compilation_config = NS(static_forward_context={k: NS(kv_cache=v) for k, v in caches.items()})
    pcp = runner.pcp_manager = object.__new__(AscendPCPManager)
    pcp._kernel_block_table_groups = {}
    pcp.device = torch.device("cpu")
    pcp.vllm_config = NS(parallel_config=NS(pcp_shard_decode_requests=False))
    pcp.pcp_world_size = 4
    pcp.pcp_rank = 3
    pcp._local_block_tables = tuple(torch.full((4, 3), 7, dtype=torch.int32) for _ in groups)
    pcp._gathered_kv_slot_mappings = torch.zeros((len(groups), 24), dtype=torch.int64)
    batch = _dummy_context_batch(4)
    batch.num_tokens_after_padding = 6
    batch.num_reqs_after_padding = 4
    batch.idx_mapping = torch.arange(4, dtype=torch.int32)
    batch.is_dummy = True
    tables, slots = runner.prepare_dummy_attn(batch, valid_state_slots=True)
    assert slots.shape == (len(groups), 4 * 6) and torch.all(slots == -1)
    assert slots.data_ptr() == pcp._gathered_kv_slot_mappings.data_ptr()
    set_dummy_context(batch, runner.block_tables, 128, num_blocks, 1024, tables)
    ctx = pcp.build_attention_context(batch, tables, slots)
    assert ctx.global_slot_mappings.shape == (len(groups), 6)
    assert torch.all(ctx.global_slot_mappings == -1)
    for gid in (1, 2):
        assert tables[gid].data_ptr() == pcp._local_block_tables[gid].data_ptr()
        assert torch.count_nonzero(tables[gid]) == 0
    assert all(torch.all(t == 5) for t in runner.block_tables.input_block_tables)
    assert torch.all(raw == 17)


@pytest.mark.parametrize("owner,ratio", [(20, 1), (2, 2)])
def test_four_pcp_replicas_restore_and_store_g64_in_their_packed_slots(monkeypatch, owner, ratio):
    from tests.deepseek_v41_utils import allocate_cache_views
    from vllm_ascend.attention import dsa_v41
    from vllm_ascend.attention.context_parallel import dsa_v41_cp

    groups = make_cache_groups(group_cache_specs(_cache_specs(layer_packed=True)))
    cfg = get_deepseek_v41_kv_cache_config(NS(cache_config=CacheConfig()), groups, 6 * (256 << 10))
    # Each PCP peer owns the front and back chunk of a prefill. Two physical
    # blocks in reverse order catch any accidental use of a local position.
    ranks = [[0, 7], [1, 6], [2, 5], [3, 4]]
    tokens = torch.arange(1, 9, dtype=torch.float32)
    scales = tokens[:, None] * torch.arange(1, 9)[None, :]
    codes = torch.arange(-32, 32, dtype=torch.int8).expand(8, 8, 64)
    latent = (codes.float() * scales[:, :, None]).view(8, 512)
    gathered = latent[sum(ranks, [])]
    restore = torch.argsort(torch.tensor(sum(ranks, [])))
    rows = torch.arange(128 - 4, 128 + 4)
    coords = torch.stack((torch.where(rows < 128, 4, 1), rows.remainder(128) // ratio), 1).int()
    if ratio == 2:
        coords[rows % 2 == 0] = -1
    meta = NS(
        swa=NS(num_input_tokens=8),
        rope=lambda *_: (rows, rows),
        indexer=NS(cache=NS(slot_mapping=coords)),
        compressor=NS(cache=NS(slot_mapping=coords), state=NS(c2_source_cos=rows, c2_source_sin=rows)),
    )
    local_swa = object.__new__(dsa_v41_cp.AscendDSAV41PCPMetadata)
    local_swa.local_num_tokens_after_padding = 2
    local_swa.hidden_restore_idx = restore

    def quantize(groups, *, dst_type):
        scale = groups.abs().amax(1) / 32
        return (groups / scale[:, None]).round().to(dst_type), scale

    def scatter(view, slots, updates):
        valid = slots[:, 0] >= 0
        view[slots[valid, 0].long(), slots[valid, 1].long()] = updates[valid]

    monkeypatch.setattr("torch_npu.npu_dynamic_quant", quantize, raising=False)
    monkeypatch.setattr(torch.ops._C_ascend, "npu_scatter_nd_update_sk", scatter, raising=False)
    monkeypatch.setattr(dsa_v41.AscendDSAV41Impl, "_apply_rotary", lambda *_: None)
    impl = object.__new__(dsa_v41_cp.AscendDSAV41PCPImpl)
    impl.role = NS(compress_ratio=ratio)
    impl._quantize_indexer_query = lambda *_: None
    impl._compress_local_c2 = lambda attn, hidden, *_: hidden
    expected_rows = torch.cat((codes.reshape(8, 512), scales.contiguous().view(torch.int8)), dim=-1)
    for rank_rows in ranks:
        (backing,), views = allocate_cache_views(cfg)
        backing.fill_(29)
        view = views[f"model.layers.{owner}.self_attn.long_kv_cache"]

        def all_gather(local, dim, rank_rows=rank_rows):
            assert dim == 0 and torch.equal(local, latent[rank_rows])
            return gathered

        group = NS(all_gather=all_gather)
        monkeypatch.setattr(dsa_v41_cp, "get_pcp_group", lambda group=group: group)
        attn = NS(
            head_dim=512,
            rotary_emb=NS(layername="rope"),
            compressor=torch.nn.Identity(),
            indexer=NS(update_keys=lambda *_: None),
            long_kv_cache=NS(kv_cache=view),
        )
        impl._compress_and_write_caches(attn, latent[rank_rows], None, None, NS(swa=local_swa), meta)
        expected = torch.full_like(backing, 29)
        for row, (block, slot) in enumerate(coords.tolist()):
            if block >= 0:
                start = view.storage_offset() + block * view.stride(0) + slot * 544
                expected[start : start + 544] = expected_rows[row]
        # Includes both slots, all ring/index aliases and the null page.
        assert torch.equal(backing, expected)


def test_pcp_ring_launcher_uses_worker_view_stride(monkeypatch):
    from tests.deepseek_v41_utils import allocate_cache_views, make_cache_config
    from vllm_ascend.models.deepseek_v41 import cache_config
    from vllm_ascend.ops.triton.compressor import compressor_triton

    register_all_kvcache_specs(None)
    register_ascend_kv_cache_specs()
    monkeypatch.setattr(cache_config, "uses_a5_packed_cache", lambda: False)
    cfg = make_cache_config(4, block_size=256, layer_packed=True)
    _, views = allocate_cache_views(cfg)
    ring = views["model.layers.2.self_attn.compressor.state_cache"].squeeze(-2)
    assert ring.stride(0) == 2 * ring[0].numel()
    empty = torch.empty(0, 512)
    controls = torch.zeros(5, 1, dtype=torch.int32)
    out = torch.empty(0, 512, dtype=torch.bfloat16)
    assert (
        compressor_triton.compressor_from_projected(
            empty, empty, ring, controls, out, max_query_len=0, num_cores=1, prev_rows=None, prev_row_indices=None
        )
        is out
    )
    calls = []
    monkeypatch.setattr(
        compressor_triton, "_write_pcp_ring_rows_kernel", {(1,): lambda *args, **kwargs: calls.append(kwargs)}
    )
    idx = torch.tensor([0])
    compressor_triton.write_pcp_ring_rows(torch.zeros(1, 1024), ring, controls, idx, idx, idx)
    assert calls[0]["PAGE_STRIDE"] == ring.stride(0)


def test_h8_smoke_maps_long_and_swa_rows_into_the_shared_pool(monkeypatch):
    from tests.deepseek_v41_utils import pack_test_caches
    from vllm_ascend.models.deepseek_v41 import cache_config

    register_all_kvcache_specs(None)
    register_ascend_kv_cache_specs()
    monkeypatch.setattr(cache_config, "uses_a5_packed_cache", lambda: False)
    table = torch.tensor([[1, 0]], dtype=torch.int32)
    slots = torch.tensor([[0, 3], [-1, -1], [1, 0]], dtype=torch.int32)
    caches = {
        layer: (torch.full((2, 128 if layer == 20 else 64, 1, 544), layer, dtype=torch.int8), slots, None)
        for layer in (2, 8, 14, 20)
    }
    layers = {i: (None, None, None, torch.full((1, 128, 1, 512), i, dtype=torch.bfloat16)) for i in range(2, 40)}
    long, long_ids, swa = pack_test_caches(caches, layers, table, torch.zeros_like(table), 1)
    assert long_ids.tolist() == [[2, 1]]
    assert set(swa) == set(layers)
    for layer, (view, mapped, _) in long.items():
        assert mapped.tolist() == [[1, 3], [-1, -1], [2, 0]]
        assert torch.equal(view[1:3], caches[layer][0])
        assert torch.count_nonzero(view[0]) == 0
    for layer, (view, ids) in swa.items():
        assert ids.min() > 2 and ids[0, 0] == ids[0, 1]
        assert torch.equal(view[ids[0, 0]].unsqueeze(0), layers[layer][3])
        assert torch.count_nonzero(view[0]) == 0


@pytest.mark.parametrize("pcp,tp", [(1, 8), (4, 2)])
def test_layer_packed_rank_planner(pcp, tp):
    from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder

    specs = _cache_specs(layer_packed=True)
    runtime = NS(
        model_config=NS(max_model_len=1024, original_max_model_len=1024, hf_config=NS(model_type="deepseek_v41")),
        max_in_flight_tokens=256,
        scheduler_config=NS(disable_hybrid_kv_cache_manager=False, max_num_seqs=2),
        parallel_config=NS(
            pipeline_parallel_size=1,
            prefill_context_parallel_size=pcp,
            decode_context_parallel_size=1,
            tensor_parallel_size=tp,
        ),
        cache_config=CacheConfig(prefix_cache_retention_interval=None),
        attention_config=NS(hisparse_config=None),
        kv_transfer_config=None,
        speculative_config=None,
    )
    builder = AscendKVCacheConfigBuilder()
    page = 256 << 10
    groups = builder.get_kv_cache_groups(runtime, specs)
    assert builder._get_pool_bytes_per_block(runtime, groups) == page
    budgets = [(256 + i) * page + 17 for i in range(8)]
    with patch(
        "vllm.platforms.current_platform.get_kv_cache_config_builder_cls",
        return_value="vllm_ascend.core.kv_cache_config_builder.AscendKVCacheConfigBuilder",
    ):
        ranks = kv_cache_planning.get_kv_cache_configs(runtime, [specs] * 8, budgets, groups)
    assert len(ranks) == 8
    for rank in ranks:
        assert isinstance(rank, LayerPackedKVCacheConfig)
        assert rank.num_blocks == 256 and rank.kv_cache_tensors == ranks[0].kv_cache_tensors
        assert {d.size for d in rank.kv_cache_tensors} == {256 * page}
    assert len(kv_cache_utils.generate_scheduler_kv_cache_config(ranks).kv_cache_groups) == 23


def test_all_groups_share_one_block_pool_and_reuse_prefix():
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request

    groups = make_cache_groups(group_cache_specs(_cache_specs(layer_packed=True)))
    runtime = NS(cache_config=CacheConfig(prefix_cache_retention_interval=None))
    cfg = get_deepseek_v41_kv_cache_config(runtime, groups, 256 * (256 << 10))
    mgr = KVCacheManager(
        kv_cache_utils.generate_scheduler_kv_cache_config([cfg]),
        1024,
        128,
        128,
        max_in_flight_tokens=256,
        enable_caching=True,
    )
    pool = mgr.block_pool
    assert len(mgr.coordinator.single_type_managers) == 23
    assert all(m.block_pool is pool for m in mgr.coordinator.single_type_managers)
    initial = pool.get_num_free_blocks()
    hashes = [bytes([i]) * 32 for i in range(1, 5)]
    first = Request("first", [1] * 512, SamplingParams(max_tokens=1), None, block_hasher=lambda _: hashes)
    allocated = mgr.allocate_slots(first, 512)
    assert allocated is not None
    ids = [b.block_id for group in allocated.blocks for b in group if not b.is_null]
    assert len(ids) == len(set(ids)) == initial - pool.get_num_free_blocks()
    mgr.free(first)
    assert pool.get_num_free_blocks() == initial
    revisit = Request("revisit", [1] * 512, SamplingParams(max_tokens=1), None, block_hasher=lambda _: hashes)
    hits, tokens, _ = mgr.get_computed_blocks(revisit)
    assert tokens == 384
    assert all(b is pool.blocks[b.block_id] for group in hits.blocks for b in group if not b.is_null)
    assert mgr.allocate_slots(revisit, 128, num_new_computed_tokens=tokens, new_computed_blocks=hits) is not None
    mgr.free(revisit)
    assert mgr.reset_prefix_cache() and pool.get_num_free_blocks() == initial
