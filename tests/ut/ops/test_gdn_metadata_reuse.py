# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests: reused metadata must equal independent full builds."""

import dataclasses
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec

from tests.ut.ops import test_gdn_attn_builder as helpers
from vllm_ascend.attention import metadata_reuse
from vllm_ascend.ops.gdn_attn_builder import AscendGDNAttentionMetadataBuilder
from vllm_ascend.worker.v2 import attn_utils


def builder(device, graph, mode, num_spec=3, config=None, reuse=True):
    cfg = config or helpers._make_vllm_config(
        num_speculative_tokens=num_spec,
        mamba_cache_mode=mode,
        cudagraph_mode=CUDAGraphMode.FULL if graph else CUDAGraphMode.NONE,
    )
    cfg.use_v2_model_runner = True
    if config is None:
        cfg.additional_config = {"reuse_kv_cache_groups": reuse}
    kv = MambaSpec(
        block_size=16,
        shapes=((1,), (1,)),
        dtypes=(torch.float32,),
        mamba_cache_mode=mode,
        num_speculative_blocks=num_spec,
    )
    return AscendGDNAttentionMetadataBuilder(kv, ["layer"], cfg, torch.device(device))


def common(device, query_lens, seq_lens, offset=0):
    m = helpers.create_common_attn_metadata(helpers.BatchSpec(seq_lens, query_lens), 16, torch.device(device))
    m.block_table_tensor = torch.arange(len(query_lens) * 24, dtype=torch.int32, device=device).reshape(-1, 24) + offset
    return m


def snapshot(obj):
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    if hasattr(obj, "__dict__"):
        return {k: snapshot(v) for k, v in vars(obj).items() if k != "_reuse_context"}
    return obj


def equal(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            equal(actual[key], expected[key])
    elif isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    else:
        assert actual == expected


CASES = [
    ("decode", [1, 1], [33, 65], None, None),
    ("prefill", [8, 5], [8, 21], None, None),
    ("decode_prefill", [1, 8], [33, 24], None, None),
    ("spec", [4, 4], [35, 67], [3, 3], None),
    ("mixed", [4, 1, 8], [35, 33, 24], [3, -1, -1], None),
    ("noncontiguous", [4, 8, 4], [35, 24, 67], [3, -1, 3], None),
    ("zero_drafts", [1, 8], [33, 24], [0, -1], None),
    ("spec_padding", [4, 4, 4, 4], [35, 67, 0, 0], [3, 3, -1, -1], 2),
    ("decode_padding", [1, 1, 0, 0], [33, 65, 0, 0], None, 2),
]


@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("mode", ["none", "align"])
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_update_matches_full_build_and_preserves_source(device, graph, mode, case):
    _, queries, lengths, drafts, actual = case
    src, dst, ref = [builder(device, graph, mode) for _ in range(3)]
    kwargs = dict(num_actual_reqs=actual)
    if drafts is not None:
        kwargs.update(
            num_decode_draft_tokens_cpu=torch.tensor(drafts, dtype=torch.int32),
            num_accepted_tokens=torch.tensor([1, 2, 1, 1][: len(drafts)], dtype=torch.int32, device=device),
        )
    m = common(device, queries, lengths)
    target = common(device, queries, lengths, 1000)
    source = src.build(0, m, **kwargs)
    before = snapshot(source)
    expected = ref.build(0, target, **kwargs)
    reused = dst.update_block_table(source, target.block_table_tensor, target.slot_mapping)
    equal(snapshot(reused), snapshot(expected))
    equal(snapshot(source), before)
    assert reused is not source
    assert reused.has_initial_state is source.has_initial_state
    for name in ("spec_query_start_loc", "non_spec_query_start_loc", "chunk_indices", "num_accepted_tokens"):
        assert getattr(reused, name) is getattr(source, name)
    if source._reuse_context.spec_padded:
        assert reused.spec_state_indices_tensor.data_ptr() == dst.spec_state_indices_tensor.data_ptr()
    if source._reuse_context.decode_padded and not source._reuse_context.spec_padded:
        assert reused.non_spec_state_indices_tensor.data_ptr() == dst.non_spec_state_indices_tensor.data_ptr()


@pytest.mark.parametrize("device", ["cpu"])
def test_graph_buffers_refresh_and_clear_previous_spec_state(device):
    src, dst = [builder(device, True, "align") for _ in range(2)]
    pointers = []
    for offset in (100, 200):
        m = common(device, [4, 4], [35, 67], offset)
        source = src.build(
            0,
            m,
            num_accepted_tokens=torch.tensor([1, 3], device=device, dtype=torch.int32),
            num_decode_draft_tokens_cpu=torch.tensor([3, 3]),
        )
        result = dst.update_block_table(source, m.block_table_tensor + 1000, m.slot_mapping)
        pointers.append(result.spec_state_indices_tensor.data_ptr())
        assert result.spec_state_indices_tensor[0, 0].item() == offset + 1002
    assert pointers[0] == pointers[1]
    m = common(device, [1, 1], [40, 70], 400)
    source = src.build(0, m)
    dst.update_block_table(source, m.block_table_tensor + 1000, m.slot_mapping)
    assert torch.all(dst.spec_state_indices_tensor[:2] == -1).item()


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_runner_reuses_only_within_invocation(monkeypatch, capture, enabled):
    first = builder("cpu", capture, "none", reuse=enabled)
    second = builder("cpu", capture, "none", config=first.vllm_config)
    groups = [
        [SimpleNamespace(layer_names=[f"layer{i}"], get_metadata_builder=lambda _, b=b: b)]
        for i, b in enumerate((first, second))
    ]
    cfg = KVCacheConfig(
        num_blocks=20,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=[f"layer{i}"], kv_cache_spec=first.kv_cache_spec) for i in range(2)
        ],
    )
    kwargs = dict(
        attn_groups=groups,
        num_reqs=2,
        num_tokens=8,
        query_start_loc_gpu=torch.tensor([0, 4, 8]),
        query_start_loc_cpu=torch.tensor([0, 4, 8]),
        max_query_len=4,
        seq_lens=torch.tensor([8, 8]),
        max_seq_len=8,
        seq_lens_np=np.array([8, 8], dtype=np.int32),
        block_tables=[
            torch.arange(8, dtype=torch.int32).reshape(2, 4),
            torch.arange(8, 16, dtype=torch.int32).reshape(2, 4),
        ],
        slot_mappings=torch.arange(16).reshape(2, 8),
        kv_cache_config=cfg,
        for_cudagraph_capture=capture,
    )
    with (
        patch.object(second, "build", wraps=second.build) as build_call,
        patch.object(second, "update_block_table", wraps=second.update_block_table) as update_call,
    ):
        first_result = attn_utils.build_attn_metadata(**kwargs)
        second_result = attn_utils.build_attn_metadata(**kwargs)
        assert build_call.call_count == (0 if enabled else 2)
        assert update_call.call_count == (2 if enabled else 0)
        assert first_result["layer0"] is not second_result["layer0"]
        # Causal mismatch must fall back even when the KV spec matches.
        attn_utils.build_attn_metadata(**kwargs, causal={0: True, 1: False})
        assert build_call.call_count == (1 if enabled else 3)


@pytest.mark.parametrize("difference", ["spec", "type", "config", "kernel", "extra", "unsupported"])
def test_reuse_key_rejects_incompatible_groups(difference):
    first = builder("cpu", False, "none")
    other = builder("cpu", False, "none", config=first.vllm_config)
    extras = {}
    if difference == "spec":
        other.kv_cache_spec = dataclasses.replace(other.kv_cache_spec, block_size=32)
    elif difference == "type":

        class Derived(AscendGDNAttentionMetadataBuilder):
            pass

        other.__class__ = Derived
    elif difference == "config":
        other.vllm_config = builder("cpu", False, "none").vllm_config
    elif difference == "kernel":
        other.kernel_block_size = 32
    elif difference == "extra":
        extras = {"unknown": object()}
    else:
        other.supports_update_block_table = False
    assert metadata_reuse._metadata_reuse_key(first, True, {}, {}) != metadata_reuse._metadata_reuse_key(
        other, True, extras, {}
    )


def test_capability_excludes_derived_builders_and_legacy_runner():
    cfg = helpers._make_vllm_config()
    cfg.use_v2_model_runner = False
    kv = MambaSpec(block_size=16, shapes=((1,), (1,)), dtypes=(torch.float32,))
    assert not AscendGDNAttentionMetadataBuilder(kv, ["x"], cfg, torch.device("cpu")).supports_update_block_table
    cfg.use_v2_model_runner = True
    assert not AscendGDNAttentionMetadataBuilder(kv, ["x"], cfg, torch.device("cpu")).supports_update_block_table
    cfg.additional_config = {"reuse_kv_cache_groups": True}
    assert AscendGDNAttentionMetadataBuilder(kv, ["x"], cfg, torch.device("cpu")).supports_update_block_table

    class Derived(AscendGDNAttentionMetadataBuilder):
        pass

    assert not Derived(kv, ["x"], cfg, torch.device("cpu")).supports_update_block_table


@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("mode", ["none", "align"])
@pytest.mark.parametrize("queries,lengths", [([1, 1], [33, 65]), ([1, 8], [33, 24])])
def test_without_speculative_configuration(device, graph, mode, queries, lengths):
    src, dst, ref = [builder(device, graph, mode, num_spec=0) for _ in range(3)]
    m = common(device, queries, lengths)
    target = common(device, queries, lengths, 1000)
    source = src.build(0, m)
    reused = dst.update_block_table(source, target.block_table_tensor, target.slot_mapping)
    equal(snapshot(reused), snapshot(ref.build(0, target)))
