# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regressions for PLE context built from actual asynchronous decode inputs.

Import the helper directly so these tests also run without loading the Ascend
plugin or the model-runner's NPU/distributed dependencies.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "vllm_ascend" / "models" / "qwen4_exp" / "ple_history.py"
ADAPTER = HELPER.with_name("ple_inputs.py")
RUNNER = ROOT / "vllm_ascend" / "worker" / "model_runner_v1.py"
MAX_NUM_REQS = 4
MAX_MODEL_LEN = 64
MAX_NUM_TOKENS = 32
CONTEXT_LEN = 2
EOS_TOKEN_ID = 248046


def _new_history(device):
    spec = importlib.util.spec_from_file_location("qwen4_exp_ple_history", HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Qwen4ExpPLEHistory(
        max_num_reqs=MAX_NUM_REQS,
        max_model_len=MAX_MODEL_LEN,
        max_num_tokens=MAX_NUM_TOKENS,
        context_len=CONTEXT_LEN,
        eos_token_id=EOS_TOKEN_ID,
        device=device,
    )


@pytest.fixture
def history():
    return _new_history(torch.device("cpu"))


def _adapter_class(history):
    # Execute the actual adapter source, substituting its already-loaded helper
    # dependency so package __init__ does not initialize the Ascend platform.
    tree = ast.parse(ADAPTER.read_text())
    tree.body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.ImportFrom) and node.module == "vllm_ascend.models.qwen4_exp.ple_history")
    ]
    namespace = {"Qwen4ExpPLEHistory": type(history)}
    exec(compile(tree, str(ADAPTER), "exec"), namespace)
    return namespace["Qwen4ExpPLEInputs"]


def _runner_config(*, ple=True, asynchronous=True):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                ple_layer_ids=[2] if ple else [],
                ngram_size=CONTEXT_LEN + 1,
                eos_token_id=EOS_TOKEN_ID,
            )
        ),
        use_async_scheduling=asynchronous,
        num_spec_tokens=0,
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        max_num_reqs=MAX_NUM_REQS,
        max_model_len=MAX_MODEL_LEN,
        max_num_tokens=MAX_NUM_TOKENS,
        device=torch.device("cpu"),
        pin_memory=False,
    )


def _batch(req_ids, starts):
    return SimpleNamespace(
        req_ids=req_ids,
        num_reqs=len(req_ids),
        num_computed_tokens_cpu=np.array(starts, dtype=np.int32),
    )


def _scheduled(*, new=(), finished=(), snapshots=None, resumed=()):
    return SimpleNamespace(
        finished_req_ids=set(finished),
        scheduled_new_reqs=[SimpleNamespace(req_id=req_id) for req_id in new],
        scheduled_cached_reqs=SimpleNamespace(
            all_token_ids={} if snapshots is None else snapshots,
            resumed_req_ids=set(resumed),
        ),
    )


def _request(prompt, output=()):
    return SimpleNamespace(prompt_token_ids=prompt, output_token_ids=list(output))


def _adapter_prepare(adapter, batch, chunks):
    lengths = np.array([len(chunk) for chunk in chunks], dtype=np.int32)
    starts = batch.num_computed_tokens_cpu.tolist()
    device = adapter.history.token_ids.device
    adapter.prepare_inputs(
        batch,
        lengths,
        torch.tensor([token for chunk in chunks for token in chunk], dtype=torch.int32, device=device),
        torch.tensor(
            [position for start, length in zip(starts, lengths) for position in range(start, start + int(length))],
            dtype=torch.int64,
            device=device,
        ),
        torch.tensor([0, *np.cumsum(lengths).tolist()], dtype=torch.int32, device=device),
    )
    return adapter.ngram_context(MAX_NUM_REQS)


def _prepare(history, req_ids, starts, chunks, *, padded_tokens=(), context_buffer=None):
    """Model only repaired input_ids; CPU output placeholders are never copied."""
    query_lens = [len(chunk) for chunk in chunks]
    flat_tokens = [token for chunk in chunks for token in chunk]
    positions = [position for start, length in zip(starts, query_lens) for position in range(start, start + length)]
    query_start_loc = [0]
    for length in query_lens:
        query_start_loc.append(query_start_loc[-1] + length)
    if padded_tokens:
        query_start_loc.append(query_start_loc[-1] + len(padded_tokens))
    flat_tokens.extend(padded_tokens)
    positions.extend([-1] * len(padded_tokens))
    device = history.token_ids.device
    if context_buffer is None:
        context_buffer = torch.full((MAX_NUM_REQS, CONTEXT_LEN), -99, dtype=torch.int32, device=device)
    result = history.prepare_inputs(
        req_ids=req_ids,
        context_ends=starts,
        query_lens=query_lens,
        input_ids=torch.tensor(flat_tokens, dtype=torch.int32, device=device),
        positions=torch.tensor(positions, dtype=torch.int64, device=device),
        query_start_loc=torch.tensor(query_start_loc, dtype=torch.int32, device=device),
        context_buffer=context_buffer,
    )
    assert result.data_ptr() == context_buffer.data_ptr()
    assert result.shape == (len(req_ids), CONTEXT_LEN)
    return context_buffer


def _assert_context(context, expected):
    torch.testing.assert_close(context[: len(expected)], context.new_tensor(expected))
    if context.shape[0] > len(expected):
        torch.testing.assert_close(
            context[len(expected) :],
            torch.full_like(context[len(expected) :], EOS_TOKEN_ID),
        )


def test_async_decode_uses_real_inputs_when_cpu_history_is_placeholder(history):
    # The initial prompt is real, while every sampled output stays -1 on CPU.
    cpu_history = [11, 13, 17] + [-1] * 4
    history.initialize_request("decode", cpu_history, num_computed_tokens=3)
    for position, input_token, expected in (
        (3, 19, [13, 17]),
        (4, 23, [17, 19]),
        (5, 29, [19, 23]),
        (6, 31, [23, 29]),
    ):
        _assert_context(_prepare(history, ["decode"], [position], [[input_token]]), [expected])
    assert cpu_history == [11, 13, 17, -1, -1, -1, -1]
    # PLE needs history before the current input, never the current token.
    assert history.token_ids[history.req_slots["decode"], 3:7].tolist() == [19, 23, 29, 31]


@pytest.mark.parametrize("prompt_length", [0, 1, 2, 3, 9])
def test_sequence_start_pads_only_missing_context(history, prompt_length):
    prompt = list(range(10, 10 + prompt_length))
    history.initialize_request("start", prompt, num_computed_tokens=prompt_length)
    actual = _prepare(history, ["start"], [prompt_length], [[42]])
    expected = ([EOS_TOKEN_ID] * CONTEXT_LEN + prompt)[-CONTEXT_LEN:]
    _assert_context(actual, [expected])


def test_context_stays_correct_through_long_async_decode(history):
    prompt = [101, 103, 107]
    history.initialize_request("long", prompt + [-1] * 50, num_computed_tokens=len(prompt))
    real_history = list(prompt)
    for position in range(len(prompt), 53):
        _assert_context(_prepare(history, ["long"], [position], [[1000 + position]]), [real_history[-CONTEXT_LEN:]])
        real_history.append(1000 + position)


def test_chunked_prefill_commits_tokens_even_when_no_output_is_sampled(history):
    # No sampled-output callback between chunks: discarded prefill logits must
    # not prevent the next chunk or first decode from finding actual history.
    history.initialize_request("prefill", [], num_computed_tokens=0)
    _assert_context(_prepare(history, ["prefill"], [0], [[2, 3, 5, 7]]), [[EOS_TOKEN_ID, EOS_TOKEN_ID]])
    _assert_context(_prepare(history, ["prefill"], [4], [[11, 13, 17]]), [[5, 7]])
    _assert_context(_prepare(history, ["prefill"], [7], [[19]]), [[13, 17]])
    _assert_context(_prepare(history, ["prefill"], [8], [[23]]), [[17, 19]])


def test_request_reordering_and_removal_preserve_request_identity(history):
    history.initialize_request("a", [2, 3], num_computed_tokens=2)
    history.initialize_request("b", [11, 13, 17], num_computed_tokens=3)
    a_slot = history.req_slots["a"]
    b_slot = history.req_slots["b"]
    _assert_context(_prepare(history, ["a", "b"], [2, 3], [[5], [19]]), [[2, 3], [13, 17]])
    history.retain_requests(["b", "a"])
    _assert_context(_prepare(history, ["b", "a"], [4, 3], [[23], [7]]), [[17, 19], [3, 5]])
    assert history.req_slots["a"] == a_slot
    assert history.req_slots["b"] == b_slot
    history.retain_requests(["b"])
    assert not history.has_request("a")
    _assert_context(_prepare(history, ["b"], [5], [[29]]), [[19, 23]])


def test_slot_reuse_does_not_leak_finished_request_tokens(history):
    history.initialize_request("old", [101, 103, 107, 109], num_computed_tokens=4)
    _prepare(history, ["old"], [4], [[113]])
    history.retain_requests([])
    history.initialize_request("new", [2], num_computed_tokens=1)
    _assert_context(_prepare(history, ["new"], [1], [[3]]), [[EOS_TOKEN_ID, 2]])
    _assert_context(_prepare(history, ["new"], [2], [[5]]), [[2, 3]])
    assert not history.has_request("old")


def test_prefix_hit_initializes_actual_history_before_first_live_input(history):
    tokens = [2, 3, 5, 7, 11, 13, 17, 19]
    prefix_length = 6
    history.initialize_request("cached", tokens, num_computed_tokens=prefix_length)
    _assert_context(_prepare(history, ["cached"], [prefix_length], [[17, 19]]), [[11, 13]])
    _assert_context(_prepare(history, ["cached"], [8], [[23]]), [[17, 19]])


def test_rewind_overwrites_rejected_suffix_before_later_context(history):
    history.initialize_request("rewind", [2, 3, 5], num_computed_tokens=3)
    _prepare(history, ["rewind"], [3], [[7, 11, 13]])
    _assert_context(_prepare(history, ["rewind"], [4], [[101]]), [[5, 7]])
    _assert_context(_prepare(history, ["rewind"], [5], [[103]]), [[7, 101]])
    _assert_context(_prepare(history, ["rewind"], [6], [[107]]), [[101, 103]])


def test_resume_after_eviction_uses_real_token_snapshot(history):
    history.initialize_request("resume", [2, 3, 5], num_computed_tokens=3)
    _prepare(history, ["resume"], [3], [[7, 11]])
    history.retain_requests([])
    # A resumed request is rebuilt from the scheduler's actual snapshot.
    history.initialize_request("resume", [2, 3, 5, 7, 11, 13], num_computed_tokens=5)
    _assert_context(_prepare(history, ["resume"], [5], [[13]]), [[7, 11]])


def test_padding_tokens_do_not_extend_history_or_pollute_context(history):
    history.initialize_request("real", [2, 3], num_computed_tokens=2)
    _assert_context(_prepare(history, ["real"], [2], [[5]], padded_tokens=[997, 991, 983]), [[2, 3]])
    assert history.valid_ends["real"] == 3
    _assert_context(_prepare(history, ["real"], [3], [[7]]), [[3, 5]])


def test_persistent_context_buffer_keeps_same_address_and_refreshes_padding(history):
    history.initialize_request("a", [2, 3], num_computed_tokens=2)
    history.initialize_request("b", [11, 13], num_computed_tokens=2)
    buffer = torch.empty((MAX_NUM_REQS, CONTEXT_LEN), dtype=torch.int32, device=history.token_ids.device)
    address = buffer.data_ptr()
    _assert_context(_prepare(history, ["a", "b"], [2, 2], [[5], [17]], context_buffer=buffer), [[2, 3], [11, 13]])
    history.retain_requests(["a"])
    _assert_context(_prepare(history, ["a"], [3], [[7]], context_buffer=buffer), [[3, 5]])
    assert buffer.data_ptr() == address


def test_missing_history_gap_fails_instead_of_hashing_uninitialized_tokens(history):
    history.initialize_request("gap", [2, 3], num_computed_tokens=2)
    with pytest.raises((ValueError, RuntimeError), match="history|computed|valid|gap"):
        _prepare(history, ["gap"], [4], [[11]])


def test_rewind_invalidates_previously_committed_future_tokens(history):
    history.initialize_request("rewind", [2, 3, 5], num_computed_tokens=3)
    _prepare(history, ["rewind"], [3], [[7, 11, 13]])
    _prepare(history, ["rewind"], [3], [[101]])
    with pytest.raises((ValueError, RuntimeError), match="history|computed|valid|gap"):
        _prepare(history, ["rewind"], [6], [[103]])


def test_initialize_rejects_cpu_placeholders_inside_required_prefix(history):
    with pytest.raises(ValueError, match="negative|placeholder|token"):
        history.initialize_request("unknown", [2, -1, 5], num_computed_tokens=3)


def test_request_capacity_is_checked_without_evicting_live_history(history):
    for index in range(MAX_NUM_REQS):
        history.initialize_request(str(index), [index + 1], num_computed_tokens=1)
    with pytest.raises((ValueError, RuntimeError), match="capacity|slot|request"):
        history.initialize_request("overflow", [13], num_computed_tokens=1)
    _assert_context(_prepare(history, ["0"], [1], [[17]]), [[EOS_TOKEN_ID, 1]])


@pytest.mark.parametrize(
    ("prefix", "computed_tokens"),
    [([2], -1), ([2, 3], 3), (list(range(MAX_MODEL_LEN + 1)), MAX_MODEL_LEN + 1)],
)
def test_initialization_rejects_invalid_prefix_range(history, prefix, computed_tokens):
    with pytest.raises(ValueError, match="prefix|bounds"):
        history.initialize_request("invalid", prefix, num_computed_tokens=computed_tokens)
    assert not history.has_request("invalid")


def test_last_valid_position_does_not_allow_history_overflow(history):
    prefix = list(range(1, MAX_MODEL_LEN))
    history.initialize_request("limit", prefix, num_computed_tokens=MAX_MODEL_LEN - 1)
    _assert_context(_prepare(history, ["limit"], [MAX_MODEL_LEN - 1], [[MAX_MODEL_LEN]]), [prefix[-CONTEXT_LEN:]])
    with pytest.raises(ValueError, match="bounds"):
        _prepare(history, ["limit"], [MAX_MODEL_LEN], [[MAX_MODEL_LEN + 1]])


def test_failed_reinitialization_preserves_previous_real_prefix(history):
    history.initialize_request("valid", [2, 3, 5], num_computed_tokens=3)
    with pytest.raises(ValueError, match="placeholder|token"):
        history.initialize_request("valid", [2, -1, 5], num_computed_tokens=3)
    _assert_context(_prepare(history, ["valid"], [3], [[7]]), [[3, 5]])


@pytest.mark.parametrize(("ple", "asynchronous"), [(False, True), (True, False), (False, False)])
def test_adapter_keeps_non_ple_and_synchronous_runners_on_existing_path(history, ple, asynchronous):
    # Unsupported async features must not affect unrelated or sync models.
    runner = _runner_config(ple=ple, asynchronous=asynchronous)
    runner.num_spec_tokens = 3
    assert _adapter_class(history).from_runner(runner) is None


def test_adapter_is_created_for_original_async_ple_configuration(history):
    adapter = _adapter_class(history).from_runner(_runner_config())
    assert adapter is not None
    assert adapter.history.context_len == CONTEXT_LEN
    assert adapter.history.max_model_len == MAX_MODEL_LEN


@pytest.mark.parametrize(
    "feature",
    ["num_spec_tokens", "pipeline_parallel_size", "decode_context_parallel_size", "prefill_context_parallel_size"],
)
def test_adapter_rejects_unsupported_async_features_before_using_wrong_history(history, feature):
    runner = _runner_config()
    setattr(runner if feature == "num_spec_tokens" else runner.parallel_config, feature, 2)
    with pytest.raises(NotImplementedError):
        _adapter_class(history).from_runner(runner)


def test_adapter_continuous_decode_preserves_gpu_history_across_cpu_placeholders(history):
    adapter = _adapter_class(history)(history)
    batch = _batch(["decode"], [3])
    requests = {"decode": _request([2, 3, 5], [-1, -1, -1])}
    adapter.update_requests(batch, _scheduled(new=["decode"]), requests)
    _assert_context(_adapter_prepare(adapter, batch, [[7]]), [[3, 5]])
    batch.num_computed_tokens_cpu[0] = 4
    adapter.update_requests(batch, _scheduled(), requests)
    _assert_context(_adapter_prepare(adapter, batch, [[11]]), [[5, 7]])
    batch.num_computed_tokens_cpu[0] = 5
    adapter.update_requests(batch, _scheduled(), requests)
    _assert_context(_adapter_prepare(adapter, batch, [[13]]), [[7, 11]])
    assert requests["decode"].output_token_ids == [-1, -1, -1]


def test_adapter_resume_prefers_scheduler_snapshot_to_placeholder_request_state(history):
    adapter = _adapter_class(history)(history)
    requests = {"resume": _request([2, 3, 5], [-1, -1, -1])}
    batch = _batch(["resume"], [5])
    adapter.update_requests(batch, _scheduled(snapshots={"resume": [2, 3, 5, 7, 11, 13]}), requests)
    _assert_context(_adapter_prepare(adapter, batch, [[13]]), [[7, 11]])
    adapter.update_requests(_batch([], []), _scheduled(), requests)
    assert not history.has_request("resume")
    batch.num_computed_tokens_cpu[0] = 6
    adapter.update_requests(batch, _scheduled(snapshots={"resume": [2, 3, 5, 7, 11, 13, 17]}), requests)
    _assert_context(_adapter_prepare(adapter, batch, [[17]]), [[11, 13]])


def test_adapter_releases_finished_identity_before_same_id_is_reused(history):
    adapter = _adapter_class(history)(history)
    batch = _batch(["shared-id"], [3])
    adapter.update_requests(batch, _scheduled(new=["shared-id"]), {"shared-id": _request([101, 103, 107])})
    _adapter_prepare(adapter, batch, [[109]])
    batch.num_computed_tokens_cpu[0] = 1
    adapter.update_requests(
        batch,
        _scheduled(new=["shared-id"], finished=["shared-id"]),
        {"shared-id": _request([2])},
    )
    _assert_context(_adapter_prepare(adapter, batch, [[3]]), [[EOS_TOKEN_ID, 2]])


def test_adapter_preserves_history_for_same_batch_forced_resume(history):
    adapter = _adapter_class(history)(history)
    batch = _batch(["rewind"], [3])
    requests = {"rewind": _request([2, 3, 5], [-1, -1, -1])}
    adapter.update_requests(batch, _scheduled(new=["rewind"]), requests)
    _adapter_prepare(adapter, batch, [[7, 11, 13]])
    batch.num_computed_tokens_cpu[0] = 4
    adapter.update_requests(batch, _scheduled(resumed=["rewind"]), requests)
    _assert_context(_adapter_prepare(adapter, batch, [[101]]), [[5, 7]])
    batch.num_computed_tokens_cpu[0] = 5
    _assert_context(_adapter_prepare(adapter, batch, [[103]]), [[7, 101]])


def test_adapter_resume_fails_on_unrecoverable_placeholder_prefix(history):
    adapter = _adapter_class(history)(history)
    with pytest.raises(ValueError, match="placeholder"):
        adapter.update_requests(
            _batch(["missing"], [5]),
            _scheduled(),
            {"missing": _request([2, 3, 5], [-1, -1])},
        )


def _actual_runner(history):
    tree = ast.parse(RUNNER.read_text())
    runner_cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
    method = next(
        node
        for node in runner_cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_maybe_add_qwen4_exp_ple_inputs"
    )
    padding = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_pad_qwen4_exp_ple_graph_inputs"
    )
    isolated_cls = ast.ClassDef(name="IsolatedRunner", bases=[], keywords=[], body=[method], decorator_list=[])
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            padding,
            isolated_cls,
        ],
        type_ignores=[],
    )
    namespace = {"Any": Any, "np": np, "torch": torch}
    exec(compile(ast.fix_missing_locations(module), str(RUNNER), "exec"), namespace)
    runner = namespace["IsolatedRunner"]()
    config = _runner_config()
    runner.model_config = config.model_config
    runner.max_num_reqs = MAX_NUM_REQS
    runner.device = history.token_ids.device
    runner._qwen4_exp_ngram_context_buffer = None
    runner._qwen4_exp_query_start_loc_buffer = None
    runner._qwen4_exp_ple_inputs = _adapter_class(history)(history)
    runner.input_batch = _batch(["decode"], [3])
    runner.input_ids = SimpleNamespace(gpu=torch.tensor([7], dtype=torch.int32, device=runner.device))
    runner.query_start_loc = SimpleNamespace(gpu=torch.tensor([0, 1], dtype=torch.int32, device=runner.device))
    return runner


def _runner_context(runner, *, dummy=False, kwargs=None):
    kwargs = {} if kwargs is None else kwargs
    runner._maybe_add_qwen4_exp_ple_inputs(
        kwargs,
        num_reqs=1,
        num_reqs_padded=1,
        is_dummy=dummy,
    )
    return kwargs


def test_actual_runner_uses_gpu_history_without_reading_cpu_token_placeholders(history):
    runner = _actual_runner(history)
    adapter = runner._qwen4_exp_ple_inputs
    requests = {"decode": _request([2, 3, 5], [-1, -1])}
    adapter.update_requests(runner.input_batch, _scheduled(new=["decode"]), requests)
    _adapter_prepare(adapter, runner.input_batch, [[7]])
    runner.input_batch.num_computed_tokens_cpu[0] = 4
    _adapter_prepare(adapter, runner.input_batch, [[11]])
    runner.input_batch.token_ids_cpu = np.array([[2, 3, 5, -1, -1]], dtype=np.int32)
    # The previous CPU slice [5, -1] must become actual history [5, 7].
    result = _runner_context(runner)
    _assert_context(result["ngram_context"], [[5, 7]])
    np.testing.assert_array_equal(runner.input_batch.token_ids_cpu, [[2, 3, 5, -1, -1]])
    assert result["query_start_loc"].tolist() == [0, 1, 1, 1, 1]


def test_actual_runner_dummy_graph_inputs_leave_real_history_unchanged(history):
    runner = _actual_runner(history)
    adapter = runner._qwen4_exp_ple_inputs
    adapter.update_requests(runner.input_batch, _scheduled(new=["decode"]), {"decode": _request([2, 3, 5])})
    _adapter_prepare(adapter, runner.input_batch, [[7]])
    saved_context = history.context_buffer.clone()
    saved_end = history.valid_ends.copy()
    actual = _runner_context(runner)
    address = actual["ngram_context"].data_ptr()
    dummy = _runner_context(runner, dummy=True)
    torch.testing.assert_close(dummy["ngram_context"], torch.full_like(dummy["ngram_context"], EOS_TOKEN_ID))
    assert dummy["ngram_context"].data_ptr() == address
    torch.testing.assert_close(history.context_buffer, saved_context)
    assert history.valid_ends == saved_end
    _assert_context(_runner_context(runner)["ngram_context"], [[3, 5]])


def test_actual_runner_synchronous_ple_keeps_real_cpu_history_path(history):
    runner = _actual_runner(history)
    runner._qwen4_exp_ple_inputs = None
    runner.input_batch.token_ids_cpu = np.array([[2, 3, 5, 7]], dtype=np.int32)
    _assert_context(_runner_context(runner)["ngram_context"], [[3, 5]])


def test_actual_runner_non_ple_does_not_touch_inputs_or_history(history):
    runner = _actual_runner(history)
    runner.model_config.hf_text_config.ple_layer_ids = []
    # No initialized history and no CPU token history: neither is consulted.
    sentinel = torch.tensor([31], dtype=torch.int32)
    kwargs = {"existing_model_input": sentinel}
    result = _runner_context(runner, kwargs=kwargs)
    assert result == {"existing_model_input": sentinel}
    assert history.req_slots == {}


def test_actual_runner_preserves_explicit_model_context(history):
    runner = _actual_runner(history)
    external = torch.tensor([[41, 43]], dtype=torch.int32, device=runner.device)
    result = _runner_context(runner, kwargs={"ngram_context": external})
    _assert_context(result["ngram_context"], [[41, 43]])
