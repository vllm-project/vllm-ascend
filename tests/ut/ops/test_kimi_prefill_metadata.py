# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-layer/step and stream-lifetime contracts for recipe PR #29's port."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tests.ut.ops import test_kimi_prefill_fused_kda as prefill_contracts


@pytest.fixture
def runtime(monkeypatch):
    ns = prefill_contracts.runtime.__wrapped__()
    timeline = []
    main_stream = SimpleNamespace(name="main")
    active = [main_stream]

    class Stream:
        name = "metadata"

        def __init__(self, **kwargs):
            pass

        def wait_event(self, event):
            timeline.append(("wait", self.name, event))

    main_stream.wait_event = lambda event: timeline.append(("wait", "main", event))

    class Event:
        def record(self):
            timeline.append(("record", active[0].name, self))

    @contextmanager
    def stream_context(stream):
        previous = active[0]
        active[0] = stream
        try:
            yield
        finally:
            active[0] = previous

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(Stream=Stream, Event=Event, stream=stream_context, current_stream=lambda: active[0]),
        raising=False,
    )
    monkeypatch.setattr(
        torch.Tensor, "record_stream", lambda tensor, stream: timeline.append(("record_stream", stream.name))
    )
    ns["timeline"] = timeline
    return ns


def _workspace(ns, *, max_batch=4, max_tokens=16):
    return ns["PrefillKDAMetadataWorkspace"](
        "cpu",
        capacity=lambda batch, tokens, **kwargs: batch + tokens,
        max_batch=max_batch,
        max_tokens=max_tokens,
    )


def _operators(ns):
    def metadata(q, v, initial_state, **kwargs):
        result = torch.zeros(initial_state.shape[0] + q.shape[0], dtype=torch.int32)
        result[: kwargs["cu_seqlens"].numel()].copy_(kwargs["cu_seqlens"])
        return result

    return ns["PrefillFusedKDAOperators"](
        MagicMock(side_effect=lambda q, k, v, **kwargs: (v.clone(), kwargs["initial_state"] + 3)),
        MagicMock(side_effect=metadata),
    )


def _invoke(ns, operator, context, *, heads=2, state=None):
    q, k, v, gate, beta = (tensor.squeeze(0).contiguous() for tensor in prefill_contracts._inputs())
    q, k, v, gate, beta = (tensor[:, :heads] for tensor in (q, k, v, gate, beta))
    if state is None:
        state = torch.zeros(2, heads, 4, 4)
    return ns["run_prefill_fused_kda"](
        operator,
        q,
        k,
        v,
        gate,
        beta,
        state,
        context.query_start_loc,
        torch.zeros(heads),
        torch.zeros(heads, 4),
        scale=0.5,
        lower_bound=-5.0,
        metadata_context=context,
    )


def test_ten_layers_and_next_forward_refresh_metadata(runtime):
    workspace = _workspace(runtime)
    operator = _operators(runtime)
    starts = torch.tensor([0, 2, 5], dtype=torch.int32)
    buffer_ptr = workspace.buffer.data_ptr()
    for step in range(2):
        # Same address, different lengths: schedule contents must be rebuilt.
        starts[1] = 2 + step
        context = workspace.new_context(starts)
        for layer in range(10):
            state = torch.full((2, 2, 4, 4), float(layer))
            _, final_state = _invoke(runtime, operator, context, state=state)
            torch.testing.assert_close(final_state, state + 3)
        assert operator.metadata.call_count == step + 1
        assert workspace.buffer.data_ptr() == buffer_ptr
        torch.testing.assert_close(context.metadata[:3], starts)
    assert operator.prefill.call_count == 20
    timeline = runtime["timeline"]
    # Every FlashKDA consumer is joined back to the allocation stream, including
    # cache hits, so the next forward cannot overwrite in-flight metadata.
    assert timeline.count(("record", "main", workspace.consumed)) == 20
    assert timeline.count(("wait", "metadata", workspace.consumed)) == 20
    builds = 2
    assert timeline.count(("record", "main", workspace.inputs_ready)) == builds
    assert timeline.count(("wait", "metadata", workspace.inputs_ready)) == builds
    assert timeline.count(("record", "metadata", workspace.metadata_ready)) == builds
    assert timeline.count(("wait", "main", workspace.metadata_ready)) == builds
    first_finish = timeline.index(("wait", "metadata", workspace.consumed))
    last_ready = len(timeline) - 1 - timeline[::-1].index(("record", "metadata", workspace.metadata_ready))
    assert first_finish < last_ready


def test_metadata_shorter_than_upper_bound_keeps_actual_length(runtime):
    # Reproduce the reported 219-token, 12-head request. The capacity helper
    # reports 5242 words while the operator returns only 1096 words.
    capacity = MagicMock(side_effect=[165419, 5242, 5242])
    workspace = runtime["PrefillKDAMetadataWorkspace"]("cpu", capacity=capacity, max_batch=128, max_tokens=16384)
    workspace.buffer.fill_(-1)
    buffer_ptr = workspace.buffer.data_ptr()
    starts = torch.tensor([0, 219], dtype=torch.int32)
    q, k, v, gate = (torch.zeros(219, 12, 128, dtype=torch.bfloat16) for _ in range(4))
    beta = torch.zeros(219, 12, dtype=torch.bfloat16)
    state = torch.zeros(1, 12, 128, 128)
    a_log = torch.zeros(12)
    dt_bias = torch.zeros(12, 128)
    results = [torch.arange(1096, dtype=torch.int32), torch.arange(512, dtype=torch.int32) + 1]
    operator = runtime["PrefillFusedKDAOperators"](
        MagicMock(side_effect=lambda q, k, v, **kwargs: (v, kwargs["initial_state"])),
        MagicMock(side_effect=results),
    )
    for step, expected in enumerate(results):
        context = workspace.new_context(starts)
        for _ in range(10):
            runtime["run_prefill_fused_kda"](
                operator,
                q,
                k,
                v,
                gate,
                beta,
                state,
                starts,
                a_log,
                dt_bias,
                scale=128**-0.5,
                lower_bound=-5.0,
                metadata_context=context,
            )
            passed_metadata = operator.prefill.call_args.kwargs["metadata"]
            assert passed_metadata.shape == expected.shape
            assert passed_metadata.data_ptr() == buffer_ptr
            torch.testing.assert_close(passed_metadata, expected)
        assert operator.metadata.call_count == step + 1
    assert operator.prefill.call_count == 20
    assert capacity.call_count == 3  # One allocation and one bound check per forward.
    torch.testing.assert_close(workspace.buffer[1096:], torch.full_like(workspace.buffer[1096:], -1))


@pytest.mark.parametrize("compact", [False, True])
def test_layers_keep_gather_scatter_and_share_compacted_boundaries(runtime, compact):
    workspace = _workspace(runtime)
    operator = _operators(runtime)
    starts = torch.tensor([0, 2, 2, 5] if compact else [0, 2, 5], dtype=torch.int32)
    keep = torch.tensor([True, False, True]) if compact else None
    chunk = SimpleNamespace(keep_meta=keep, flash_kda_metadata=workspace.new_context(starts, keep))
    for _ in range(3):
        layer, _, _ = prefill_contracts._paged_layer(runtime)
        prefill_contracts._install_prefill(layer, runtime)
        layer._prefill_fused_kda_operator = operator
        state = layer.kv_cache[1]
        state.fill_(9)
        before = state.clone()
        indices = torch.tensor([5, 2, 8] if compact else [5, 8], dtype=torch.int32)
        flags = torch.tensor([True, True, False] if compact else [True, False])
        layer._run_prefill_fused_kda(*prefill_contracts._inputs(), state, indices, flags, chunk, query_start_loc=starts)
        layer._ascend_kda_state_copy.gather.assert_called_once()
        layer._ascend_kda_state_copy.scatter.assert_called_once()
        torch.testing.assert_close(state[5], torch.full_like(state[5], 12))
        torch.testing.assert_close(state[8], torch.full_like(state[8], 3))
        torch.testing.assert_close(state[2], before[2])
    operator.metadata.assert_called_once()
    torch.testing.assert_close(
        operator.metadata.call_args.kwargs["cu_seqlens"], torch.tensor([0, 2, 5], dtype=torch.int32)
    )


def test_builder_attaches_fresh_shared_context(runtime):
    ns = runtime
    prefill_contracts.contracts._definitions(
        "vllm_ascend/ops/gdn_attn_builder.py", {"GDNCausalConv1dMetadata", "GDNPrefillMetadata"}, ns
    )
    prefill_contracts.contracts._definitions(
        "vllm_ascend/ops/gdn_attn_builder.py",
        {"AscendGDNAttentionMetadataBuilder"},
        ns,
        methods={"_attach_non_spec_prefill_metadata"},
        bases="Module",
    )
    builder = ns["AscendGDNAttentionMetadataBuilder"]()
    builder._USE_COMMON_KERNEL_METADATA = True
    builder.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(prefill_context_parallel_size=1))
    builder._flash_kda_metadata_workspace = _workspace(ns)
    starts = torch.tensor([0, 2, 2, 5], dtype=torch.int32)
    chunk = SimpleNamespace(keep_meta=torch.tensor([True, False, True]))
    contexts = []
    for _ in range(2):
        metadata = SimpleNamespace(
            num_prefills=3,
            non_spec_query_start_loc=starts,
            prefill_query_start_loc=starts,
            has_initial_state=torch.tensor([True, False, False]),
        )
        builder._attach_non_spec_prefill_metadata(metadata, chunk, torch.tensor([1, 2, 3]))
        contexts.append(metadata.non_spec_prefill_metadata.chunk.flash_kda_metadata)
    assert contexts[0] is not contexts[1]
    assert contexts[0].workspace is contexts[1].workspace
    torch.testing.assert_close(contexts[0].query_start_loc, torch.tensor([0, 2, 5], dtype=torch.int32))


def test_reuse_rejects_layout_or_operator_changes(runtime):
    context = _workspace(runtime).new_context(torch.tensor([0, 2, 5], dtype=torch.int32))
    operator = _operators(runtime)
    _invoke(runtime, operator, context)
    with pytest.raises(RuntimeError, match="different layouts"):
        _invoke(runtime, operator, context, heads=1)
    with pytest.raises(RuntimeError, match="different layouts"):
        _invoke(runtime, _operators(runtime), context)
    operator.metadata.assert_called_once()


def test_capacity_exhaustion_fails_before_launch(runtime):
    context = _workspace(runtime, max_batch=1, max_tokens=1).new_context(torch.tensor([0, 2, 5], dtype=torch.int32))
    operator = _operators(runtime)
    with pytest.raises(RuntimeError, match="fixed buffer"):
        _invoke(runtime, operator, context)
    operator.metadata.assert_not_called()
    operator.prefill.assert_not_called()


@pytest.mark.parametrize(
    "bad_result",
    [
        torch.zeros(7),
        torch.zeros(8, dtype=torch.int32),
        torch.zeros(21, dtype=torch.int32),
        torch.zeros(7, dtype=torch.int32, device="meta"),
    ],
)
def test_metadata_contract_is_checked_before_consumer(runtime, bad_result):
    context = _workspace(runtime).new_context(torch.tensor([0, 2, 5], dtype=torch.int32))
    operator = _operators(runtime)
    operator.metadata.side_effect = None
    operator.metadata.return_value = bad_result
    with pytest.raises(RuntimeError, match="capacity/dtype/device") as error:
        _invoke(runtime, operator, context)
    message = str(error.value)
    assert (
        f"actual(shape={tuple(bad_result.shape)}, numel={bad_result.numel()}, "
        f"dtype={bad_result.dtype}, device={bad_result.device})" in message
    )
    assert "expected(numel<=7, dtype=torch.int32, device=cpu)" in message
    assert "buffer_capacity=20" in message
    assert "q_shape=(5, 2, 4)" in message
    assert "initial_state_shape=(2, 2, 4, 4)" in message
    operator.prefill.assert_not_called()


def test_consumer_exception_still_joins_streams(runtime):
    workspace = _workspace(runtime)
    context = workspace.new_context(torch.tensor([0, 2, 5], dtype=torch.int32))
    operator = _operators(runtime)
    operator.prefill.side_effect = RuntimeError("kernel failed")
    with pytest.raises(RuntimeError, match="kernel failed"):
        _invoke(runtime, operator, context)
    assert runtime["timeline"][-1] == ("wait", "metadata", workspace.consumed)


def test_startup_requires_capacity_helper(runtime):
    with pytest.raises(ImportError, match="metadata_capacity_upper_bound"):
        runtime["PrefillKDAMetadataWorkspace"]("cpu")


def test_factory_enables_reuse_by_default_and_ignores_other_models(runtime):
    factory = runtime["create_prefill_kda_metadata_workspace"]
    vc = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={"other": object()}),
        scheduler_config=SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=16),
    )
    assert factory(vc, ["other"], "cpu") is None
    vc.compilation_config.static_forward_context["kimi"] = SimpleNamespace(_prefill_fused_kda_operator=object())
    runtime["import_module"].return_value = SimpleNamespace(metadata_capacity_upper_bound=lambda b, t, **kwargs: b + t)
    assert factory(vc, ["kimi"], "cpu").buffer.numel() == 20
