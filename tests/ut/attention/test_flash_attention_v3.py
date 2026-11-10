# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exercise the FA3 backend on CPU, replacing only framework/device boundaries."""

import ast
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def runtime(monkeypatch):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/flash_attention_v3.py"
    tree = ast.parse(source.read_text())
    # Keep production class bodies intact without importing the NPU runtime.
    tree.body = [
        node for node in tree.body if not (isinstance(node, ast.ImportFrom) and (node.module or "").startswith("vllm"))
    ]

    class Base:
        def __init__(self, *args):
            pass

        def __class_getitem__(cls, _item):
            return cls

    cache_write = Mock()
    notify = Mock()
    namespace = {
        "__name__": __name__,
        "VllmConfig": object,
        "CUDAGraphMode": SimpleNamespace(FULL="full"),
        "logger": Mock(),
        "AttentionBackend": Base,
        "AttentionCGSupport": SimpleNamespace(ALWAYS="always"),
        "AttentionImpl": Base,
        "AttentionLayer": object,
        "AttentionMetadata": Base,
        "AttentionMetadataBuilder": Base,
        "AttentionType": SimpleNamespace(DECODER="decoder"),
        "AttentionSpec": object,
        "AscendCommonAttentionMetadata": object,
        "DeviceOperator": SimpleNamespace(reshape_and_cache=cache_write),
        "notify_kv_cache_written": notify,
        "attention_transfer_window": nullcontext,
    }
    exec(compile(tree, str(source), "exec"), namespace)
    ops = SimpleNamespace(
        get_scheduler_metadata=Mock(side_effect=lambda **kwargs: torch.zeros(8, dtype=torch.uint8)),
        flash_attn_with_kvcache=Mock(side_effect=lambda query, *_args, **_kwargs: query + 1),
    )
    monkeypatch.setitem(sys.modules, "flash_attn_npu_3", ops)
    return SimpleNamespace(**namespace, ops=ops, cache_write=cache_write, notify=notify)


def make_builder(runtime, *, window=None, eager=False):
    impl = runtime.AscendFlashAttentionImpl(8, 192, 192**-0.5, num_kv_heads=2, sliding_window=window)
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(enforce_eager=eager),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(decode_mode=lambda: "full"),
            # A target bucket of 15 can round to a 16-token DFlash graph.
            cudagraph_capture_sizes=[15],
            static_forward_context={"draft.0": SimpleNamespace(impl=impl), "draft.1": SimpleNamespace(impl=impl)},
        ),
    )
    spec = SimpleNamespace(dtype=torch.bfloat16, block_size=128)
    builder = runtime.AscendFlashAttentionMetadataBuilder(spec, ["draft.0", "draft.1"], config, torch.device("cpu"))
    return builder, impl


def make_common():
    return SimpleNamespace(
        num_reqs=4,
        query_start_loc_cpu=torch.tensor([0, 4, 8, 12, 16], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 4, 8, 12, 16], dtype=torch.int32),
        seq_lens=torch.tensor([21, 31, 41, 51], dtype=torch.int32),
        block_table_tensor=torch.zeros(4, 8, dtype=torch.int32),
        num_input_tokens=16,
        num_actual_tokens=16,
        max_query_len=4,
        causal=False,
        slot_mapping=torch.arange(16, dtype=torch.int32),
    )


def test_replay_shrink_and_grow_refreshes_captured_tiling_without_reallocating(runtime):
    builder, _ = make_builder(runtime)
    common = make_common()
    first = builder.build(0, common)
    spec = next(iter(first.scheduler_metadata))
    captured = first.scheduler_metadata[spec]
    pointer = captured.data_ptr()

    for step, offsets in enumerate(([0, 4, 4, 4, 4], [0, 4, 8, 12, 12]), start=1):
        common.query_start_loc_cpu.copy_(torch.tensor(offsets))
        common.query_start_loc.copy_(common.query_start_loc_cpu)
        common.seq_lens.sub_(1)
        runtime.ops.get_scheduler_metadata.side_effect = lambda step=step, **kwargs: torch.full(
            (8,), step, dtype=torch.uint8
        )
        replay = builder.build(0, common)
        assert replay.scheduler_metadata[spec] is captured
        assert captured.data_ptr() == pointer
        assert torch.equal(captured, torch.full((8,), step, dtype=torch.uint8))
        args = runtime.ops.get_scheduler_metadata.call_args.kwargs
        assert args["batch_size"] == (1 if step == 1 else 3)
        assert args["cache_seqlens"].data_ptr() == common.seq_lens.data_ptr()
        assert args["cu_seqlens_q"].data_ptr() == common.query_start_loc.data_ptr()
        assert args["cu_seqlens_q"].numel() == args["batch_size"] + 1
        assert args["max_seqlen_k"] == 8 * 128
        assert args["max_seqlen_q"] == 4

    # Two equivalent layers use one tiling call per build.
    assert runtime.ops.get_scheduler_metadata.call_count == 3
    assert len(builder.scheduler_buffers) == 1


@pytest.mark.parametrize("change", ["bucket", "source", "causal", "query_width"])
def test_distinct_capture_contexts_do_not_overwrite_each_other(runtime, change):
    builder, _ = make_builder(runtime)
    common = make_common()
    captured = builder.build(0, common)
    if change == "bucket":
        common.num_input_tokens = 20
    elif change == "source":
        common.seq_lens = common.seq_lens.clone()
    elif change == "causal":
        common.causal = True
    else:
        common.max_query_len = 8
        common.query_start_loc_cpu.copy_(torch.tensor([0, 8, 16, 16, 16]))
        common.query_start_loc.copy_(common.query_start_loc_cpu)
    other = builder.build(0, common)
    spec = next(iter(captured.scheduler_metadata))
    assert captured.scheduler_metadata[spec].data_ptr() != other.scheduler_metadata[spec].data_ptr()


def test_dummy_batch_has_one_empty_request_and_layout_changes_fail_closed(runtime):
    builder, _ = make_builder(runtime)
    common = make_common()
    common.query_start_loc_cpu.zero_()
    common.query_start_loc.zero_()
    common.seq_lens.zero_()
    builder.build(0, common)
    assert runtime.ops.get_scheduler_metadata.call_args.kwargs["batch_size"] == 1
    runtime.ops.get_scheduler_metadata.side_effect = lambda **kwargs: torch.zeros(9, dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="layout changed"):
        builder.build(0, common)


def test_eager_builds_do_not_retain_tiling_for_every_batch_shape(runtime):
    builder, _ = make_builder(runtime, eager=True)
    common = make_common()
    first = builder.build(0, common)
    second = builder.build(0, common)
    spec = next(iter(first.scheduler_metadata))
    assert first.scheduler_metadata[spec] is not second.scheduler_metadata[spec]
    assert not builder.scheduler_buffers


@pytest.mark.parametrize("offsets", [[0, 3, 8, 12, 16], [0, 4, 4, 8, 12]])
def test_ragged_queries_preserve_packed_layout_and_offsets(runtime, offsets):
    builder, impl = make_builder(runtime)
    common = make_common()
    common.query_start_loc_cpu.copy_(torch.tensor(offsets))
    common.query_start_loc.copy_(common.query_start_loc_cpu)
    common.max_query_len = max(b - a for a, b in zip(offsets, offsets[1:]))
    metadata = builder.build(0, common)
    schedule_args = runtime.ops.get_scheduler_metadata.call_args.kwargs
    assert schedule_args["cu_seqlens_q"] is metadata.query_start_loc
    assert schedule_args["max_seqlen_q"] == common.max_query_len
    query = torch.randn(16, 8, 192, dtype=torch.bfloat16)
    output = torch.empty_like(query)
    layer = SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0)
    impl.forward(layer, query, None, None, (None, None), metadata, output)
    call = runtime.ops.flash_attn_with_kvcache.call_args
    assert call.args[0] is query
    assert call.kwargs["cu_seqlens_q"] is metadata.query_start_loc
    assert call.kwargs["max_seqlen_q"] == common.max_query_len
    torch.testing.assert_close(output, query + 1)


@pytest.mark.parametrize(
    "causal,window,expected", [(False, None, (-1, -1)), (False, 64, (63, 63)), (True, 64, (63, 0))]
)
def test_forward_preserves_scale_output_and_matching_device_schedule(runtime, causal, window, expected):
    builder, impl = make_builder(runtime, window=window)
    common = make_common()
    common.causal = causal
    common.num_actual_tokens = 8
    metadata = builder.build(0, common)
    query = torch.randn(16, 8, 192, dtype=torch.bfloat16)
    key = torch.randn(16, 2, 192, dtype=query.dtype)
    value = torch.randn_like(key)
    # The runner interleaves physical K/V pages; the backend must keep views.
    backing = torch.empty(2, 2, 128, 2, 192)
    cache = (backing[:, 0], backing[:, 1])
    layer = SimpleNamespace(layer_name="draft.0", _k_scale_float=1.0, _v_scale_float=1.0)
    output = torch.empty_like(query)
    assert impl.forward(layer, query, key, value, cache, metadata, output) is output
    torch.testing.assert_close(output, query + 1)
    cache_args = runtime.cache_write.call_args.kwargs
    assert cache_args["key"].shape == (8, 2, 192)
    assert cache_args["slot_mapping"].shape == (8,)
    assert cache_args["key_cache"] is cache[0]
    runtime.notify.assert_called_once_with("draft.0")
    args = runtime.ops.flash_attn_with_kvcache.call_args.kwargs
    assert runtime.ops.flash_attn_with_kvcache.call_args.args[1] is cache[0]
    assert runtime.ops.flash_attn_with_kvcache.call_args.args[2] is cache[1]
    assert args["cache_seqlens"] is metadata.seq_lens
    assert args["cu_seqlens_q"] is metadata.query_start_loc
    assert args["max_seqlen_q"] == common.max_query_len
    assert runtime.ops.flash_attn_with_kvcache.call_args.args[0] is query
    assert args["softmax_scale"] == 192**-0.5
    assert args["softcap"] == 0.0
    assert args["causal"] is causal
    assert args["window_size"] == expected
    assert runtime.ops.get_scheduler_metadata.call_args.kwargs["window_size"] == expected
    assert args["scheduler_metadata"] is next(iter(metadata.scheduler_metadata.values()))

    # DFlash context K/V can be written separately from the query attention.
    runtime.cache_write.reset_mock()
    impl.forward(layer, query, None, None, cache, metadata, output)
    runtime.cache_write.assert_not_called()


def test_shared_cache_and_profile_forward_skip_cache_writes(runtime):
    _, impl = make_builder(runtime)
    impl.kv_sharing_target_layer_name = "draft.owner"
    impl.do_kv_cache_update(None, None, None, None, None)
    runtime.cache_write.assert_not_called()
    layer = SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0)
    output = torch.ones(4, 8, 192)
    assert impl.forward(layer, None, None, None, None, None, output) is output
    assert torch.count_nonzero(output) == 0
    runtime.ops.flash_attn_with_kvcache.assert_not_called()
