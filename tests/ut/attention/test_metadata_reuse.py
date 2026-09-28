# SPDX-License-Identifier: Apache-2.0

"""Full/sliding reuse must match fresh metadata and keep cache ownership."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import (
    CrossAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVQuantMode,
    SlidingWindowSpec,
)

from vllm_ascend.attention.attention_v1 import (
    AscendAttentionMetadataBuilder,
    AscendAttentionState,
    AscendMetadata,
)
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.worker.v2 import attn_utils


@pytest.fixture(autouse=True)
def _cpu_pinned_memory(monkeypatch):
    # These UTs compare CPU metadata; no accelerator allocator is needed.
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda tensor, *args, **kwargs: tensor)


def config(*, speculative=True, tp=1):
    return SimpleNamespace(
        use_v2_model_runner=True,
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1, decode_context_parallel_size=1, tensor_parallel_size=tp
        ),
        model_config=SimpleNamespace(max_model_len=4096, runner_type="generate"),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY),
        speculative_config=SimpleNamespace(num_speculative_tokens=3, parallel_drafting=True) if speculative else None,
        scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
    )


def spec(kind="full", **overrides):
    args = dict(block_size=128, num_kv_heads=4, head_size=128, dtype=torch.bfloat16)
    args.update(overrides)
    return FullAttentionSpec(**args) if kind == "full" else SlidingWindowSpec(**args, sliding_window=1024)


def builder(device="cpu", kv=None, cfg=None, cls=AscendAttentionMetadataBuilder):
    result = cls(kv or spec(), ["layer"], cfg or config(), torch.device(device))
    mask = torch.triu(torch.ones(16, 16, dtype=torch.bool, device=device), diagonal=1)
    result.attn_mask_builder = SimpleNamespace(get_attention_mask=lambda causal, _: mask if causal else None)
    return result


def common(device, case, offset=0, causal=True):
    # Real logical requests, optional dummy query row and short state table.
    queries, lengths, table_rows, actual_tokens, state = {
        "decode": ([1, 1], [40, 60], 2, 2, AscendAttentionState.DecodeOnly),
        "prefill": ([8, 20], [8, 36], 2, 28, AscendAttentionState.ChunkedPrefill),
        "mixed": ([4, 20], [44, 36], 2, 24, AscendAttentionState.ChunkedPrefill),
        "graph_padding": ([4, 4, 4], [44, 36], 2, 8, AscendAttentionState.SpecDecoding),
    }[case]
    qcpu = torch.tensor([0] + np.cumsum(queries).tolist(), dtype=torch.int32)
    lenscpu = torch.tensor(lengths, dtype=torch.int32)
    table = torch.arange(table_rows * 4, dtype=torch.int32, device=device).reshape(table_rows, 4) + offset
    slots = torch.arange(sum(queries) + 4, dtype=torch.int64, device=device) + offset * 128
    slots[actual_tokens:] = -1
    return AscendCommonAttentionMetadata(
        query_start_loc=qcpu.to(device),
        query_start_loc_cpu=qcpu,
        seq_lens=lenscpu.to(device),
        seq_lens_cpu=lenscpu,
        _seq_lens_cpu=lenscpu,
        seq_lens_cpu_upper_bound=lenscpu,
        num_reqs=len(queries),
        num_actual_tokens=actual_tokens,
        max_query_len=max(queries),
        max_seq_len=max(lengths),
        block_table_tensor=table,
        slot_mapping=slots,
        causal=causal,
        is_prefilling=torch.tensor([q > 4 for q in queries]),
        positions=torch.arange(sum(queries), device=device),
        attn_state=state,
    )


def assert_equal(a, b):
    for field, expected in vars(b).items():
        actual = getattr(a, field)
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        else:
            assert actual == expected, field


@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("kind", ["full", "sliding"])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("capture", [True, False])
@pytest.mark.parametrize("case", ["decode", "prefill", "mixed", "graph_padding"])
def test_update_equals_independent_build(device, kind, causal, capture, case):
    src, dst, ref = [builder(device, spec(kind)) for _ in range(3)]
    m, n = common(device, case, causal=causal), common(device, case, 100, causal)

    def build(b, c):
        return b.build_for_cudagraph_capture(c) if capture else b.build(0, c)

    source = build(src, m)
    source_table = source.block_tables.clone()
    source_slots = source.slot_mapping.clone()
    result = dst.update_block_table(source, n.block_table_tensor, n.slot_mapping)
    assert_equal(result, build(ref, n))
    assert result is not source
    torch.testing.assert_close(source.block_tables, source_table)
    torch.testing.assert_close(source.slot_mapping, source_slots)
    for name in ("query_start_loc", "seq_lens", "seq_lens_list", "actual_seq_lengths_q", "attn_mask"):
        assert getattr(result, name) is getattr(source, name)
    assert result.slot_mapping.numel() == m.num_actual_tokens
    if case == "graph_padding":
        assert result.block_tables.shape[0] == 3
        assert torch.count_nonzero(result.block_tables[-1]).item() == 0
    source.reshape_cache_event = object()
    result = dst.update_block_table(source, n.block_table_tensor, n.slot_mapping)
    assert result.reshape_cache_event is None


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("capture", [False, True])
def test_runner_builds_once_per_exact_full_sliding_spec(monkeypatch, enabled, capture):
    cfg = config()
    monkeypatch.setattr(
        attn_utils,
        "get_ascend_config",
        lambda: SimpleNamespace(reuse_kv_cache_groups=enabled),
    )
    specs = [spec()] * 8 + [spec("sliding", num_kv_heads=8)] * 5 + [spec(num_kv_heads=8)]
    builders = [builder(kv=kv, cfg=cfg) for kv in specs]
    groups = [
        [SimpleNamespace(layer_names=[f"layer{i}"], get_metadata_builder=lambda _, b=b: b)]
        for i, b in enumerate(builders)
    ]
    kv_config = KVCacheConfig(
        num_blocks=100,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=[f"layer{i}"], kv_cache_spec=kv) for i, kv in enumerate(specs)],
    )
    tables = [torch.arange(8, dtype=torch.int32).reshape(2, 4) + i * 10 for i in range(14)]
    slots = torch.arange(14 * 8, dtype=torch.int64).reshape(14, 8)
    kwargs = dict(
        attn_groups=groups,
        num_reqs=2,
        num_tokens=8,
        query_start_loc_gpu=torch.tensor([0, 4, 8]),
        query_start_loc_cpu=torch.tensor([0, 4, 8]),
        max_query_len=4,
        seq_lens=torch.tensor([40, 60]),
        max_seq_len=60,
        seq_lens_np=np.array([40, 60], dtype=np.int32),
        block_tables=tables,
        slot_mappings=slots,
        kv_cache_config=kv_config,
        for_cudagraph_capture=capture,
        attn_state=AscendAttentionState.SpecDecoding,
    )
    original = AscendAttentionMetadataBuilder.build
    with patch.object(AscendAttentionMetadataBuilder, "build", autospec=True, side_effect=original) as calls:
        for _ in range(2):
            result = attn_utils.build_attn_metadata(**kwargs)
            assert len({id(m) for m in result.values()}) == 14
            for i in range(14):
                assert result[f"layer{i}"].block_tables is tables[i]
                torch.testing.assert_close(result[f"layer{i}"].slot_mapping, slots[i])
            # A new invocation must consume the new group-local addresses.
            for t in tables:
                t.add_(100)
        assert calls.call_count == (6 if enabled else 28)
        attn_utils.build_attn_metadata(**kwargs, causal={0: False})
        assert calls.call_count == (10 if enabled else 42)


@pytest.mark.parametrize("difference", ["spec", "window", "causal", "config", "kernel", "group_spec", "extra"])
def test_different_metadata_requirements_do_not_match(difference):
    cfg = config()
    a = builder(cfg=cfg)
    b = builder(cfg=cfg)
    kwargs = dict(causal=True, common_kwargs={}, build_kwargs={}, group_spec=a.kv_cache_spec)
    other = dict(kwargs)
    if difference == "spec":
        b.kv_cache_spec = spec(head_size=256)
    elif difference == "window":
        b.kv_cache_spec = spec("sliding")
    elif difference == "causal":
        other["causal"] = False
    elif difference == "config":
        b.vllm_config = config(speculative=False)
    elif difference == "kernel":
        b.kernel_block_size = 256
    elif difference == "group_spec":
        other["group_spec"] = spec(block_size=256)
    else:
        other["common_kwargs"] = {"unsupported": object()}
    assert attn_utils._metadata_reuse_key(a, **kwargs) != attn_utils._metadata_reuse_key(b, **other)


@pytest.mark.parametrize(
    "restriction", ["v1", "pcp", "dcp", "cross", "quant", "static_quant", "derived", "metadata_class", "pooling"]
)
def test_unsupported_builder_cannot_update(restriction):
    cfg = config()
    kv = spec()
    cls = AscendAttentionMetadataBuilder
    if restriction == "v1":
        cfg.use_v2_model_runner = False
    elif restriction == "pcp":
        cfg.parallel_config.prefill_context_parallel_size = 2
    elif restriction == "dcp":
        cfg.parallel_config.decode_context_parallel_size = 2
    elif restriction == "cross":
        kv = CrossAttentionSpec(block_size=128, num_kv_heads=4, head_size=128, dtype=torch.bfloat16)
    elif restriction == "static_quant":
        kv = replace(kv, dtype=torch.int8)
    elif restriction == "quant":
        kv = replace(kv, kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD)
    elif restriction in ("derived", "metadata_class"):

        class Derived(AscendAttentionMetadataBuilder):
            pass

        cls = Derived
        if restriction == "metadata_class":
            cls.metadata_cls = SimpleNamespace
    elif restriction == "pooling":
        cfg.model_config.runner_type = "pooling"
    b = builder(kv=kv, cfg=cfg, cls=cls)
    assert not b.supports_update_block_table
    with pytest.raises(NotImplementedError):
        b.update_block_table(AscendMetadata(), torch.zeros(1, 1), torch.zeros(1))


def test_tp2_and_no_speculative_config_can_reuse():
    cfg = config(speculative=False, tp=2)
    b = builder(cfg=cfg)
    assert b.supports_update_block_table
    source = b.build(0, common("cpu", "mixed"))
    other = common("cpu", "mixed", 100)
    updated = b.update_block_table(source, other.block_table_tensor, other.slot_mapping)
    assert_equal(updated, b.build(0, other))
