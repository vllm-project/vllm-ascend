# SPDX-License-Identifier: Apache-2.0
"""CPU control-flow tests executing scheduler source with fake cache/worker I/O.

The pinned vLLM checkout (or VLLM_SOURCE_ROOT) supplies the real base scheduler,
async methods, queues, and request status. Only imports and hardware/model
construction are excluded. These tests do not establish NPU or API support.
"""

import ast
import enum
import hashlib
import importlib.util
import json
import os
import sys
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "vllm_ascend/core/decoder_append_prefill_scheduler.py"
COPY_BASELINE_VLLM_REVISION = "544f182d9f0c9bfe8bd7645525951d5e3692a542"
# Generated independently from Scheduler.schedule in the declared copy baseline.
# The installed dependency can also be the supported v0.32.0 release, whose
# upstream schedule legitimately differs from that main-branch snapshot.
COPY_BASELINE_SCHEDULE_SHA256 = "0d0446b190ef0431a7b7862dd9728af2b27770c3d3b8dcb693a242cc9c75a145"
VLLM = Path(os.environ.get("VLLM_SOURCE_ROOT", ROOT.parent / "vllm")) / "vllm"
if not VLLM.is_dir():
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise RuntimeError("Install the target vLLM or set VLLM_SOURCE_ROOT to its checkout.")
    VLLM = Path(spec.origin).parent


class Output(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(outputs=[], has_structured_output_requests=False, pending_structured_output_tokens=False)
        self.__dict__.update(kwargs)


class TestRequest(SimpleNamespace):
    __test__ = False
    __hash__: ClassVar[Any] = object.__hash__

    def is_finished(self):
        return self.status.is_finished(self.status)

    def get_finished_reason(self):
        return "stop"

    def append_output_token_ids(self, token):
        self.num_output_tokens += 1
        self.num_tokens += 1

    def take_prefill_stats(self):
        return None

    def take_events(self):
        return ["queued"]


def execute_classes(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    assert len(nodes) == len(names)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


@pytest.fixture
def runtime(monkeypatch):
    namespace: dict[str, Any] = dict(
        __name__="admission_test_runtime",
        ABC=ABC,
        abstractmethod=abstractmethod,
        deque=deque,
        enum=enum,
        Enum=enum.Enum,
        time=time,
        defaultdict=defaultdict,
        dataclass=dataclass,
        field=field,
        SchedulerInterface=object,
        _SchedulerBase=object,
        SupportsHMA=type("SupportsHMA", (), {}),
        MULTIMODAL_REGISTRY=None,
        current_platform=SimpleNamespace(is_xpu=lambda: False),
        Request=TestRequest,
        EngineCoreOutputs=Output,
        EngineCoreOutput=Output,
        KVConnectorBlockState=SimpleNamespace,
        KV_FETCH_WAITING_TO_START="waiting_to_start",
        KV_FETCH_IN_PROGRESS="in_progress",
        KV_FETCH_COMPLETED_WAITING="completed_waiting",
        KV_FETCH_STAGES=("waiting_to_start", "in_progress", "completed_waiting"),
        KVCacheManager=lambda **kwargs: Mock(coordinator=SimpleNamespace(single_type_managers=[])),
        EventPublisherFactory=SimpleNamespace(create=lambda *args: Mock()),
        EncoderCacheManager=SimpleNamespace(create_manager=lambda **kwargs: Mock()),
        MambaSpec=type("MambaSpec", (), {}),
        logger=Mock(),
        check_stop=lambda request, max_model_len: False,
        record_function_or_nullcontext=lambda *args: nullcontext(),
        remove_all=lambda seq, removed: [req for req in seq if req not in removed],
        NewRequestData=SimpleNamespace(from_request=lambda request, *args, **kwargs: request),
    )
    # Dataclasses resolve postponed annotations through their defining module.
    module = ModuleType(namespace["__name__"])
    module.__dict__.update(namespace)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    namespace = module.__dict__
    execute_classes(VLLM / "v1/core/sched/output.py", {"SchedulerOutput"}, namespace)
    execute_classes(VLLM / "v1/request.py", {"RequestStatus"}, namespace)
    execute_classes(VLLM / "v1/core/sched/interface.py", {"PauseState"}, namespace)
    execute_classes(
        VLLM / "v1/core/sched/request_queue.py",
        {"SchedulingPolicy", "RequestQueue", "FCFSRequestQueue", "PriorityRequestQueue"},
        namespace,
    )
    namespace["create_request_queue"] = lambda policy: namespace["FCFSRequestQueue"]()
    execute_classes(VLLM / "v1/core/sched/scheduler.py", {"Scheduler"}, namespace)
    execute_classes(VLLM / "v1/core/sched/async_scheduler.py", {"AsyncScheduler"}, namespace)
    execute_classes(
        SOURCE,
        {"DecoderCacheAdmissionMixin", "DecoderAppendPrefillScheduler", "AsyncDecoderAppendPrefillScheduler"},
        namespace,
    )
    namespace.update(
        RecomputeSchedulerOutput=Output,
        RecomputeReqInfo=lambda **kwargs: SimpleNamespace(**kwargs),
        FinishReason=SimpleNamespace(STOP="stop"),
        get_ascend_config=lambda: SimpleNamespace(
            finegrained_tp_config=SimpleNamespace(oproj_tensor_parallel_size=1, mlp_tensor_parallel_size=1)
        ),
    )
    execute_functions(
        ROOT / "vllm_ascend/core/recompute_scheduler.py", {"_normalize_encoder_schedule_result"}, namespace
    )
    execute_classes(ROOT / "vllm_ascend/core/recompute_scheduler.py", {"RecomputeScheduler"}, namespace)
    execute_classes(
        SOURCE,
        {"DecoderAppendPrefillRecomputeScheduler", "AsyncDecoderAppendPrefillRecomputeScheduler"},
        namespace,
    )
    return SimpleNamespace(**namespace)


@pytest.fixture(
    params=[
        "DecoderAppendPrefillScheduler",
        "AsyncDecoderAppendPrefillScheduler",
        "DecoderAppendPrefillRecomputeScheduler",
        "AsyncDecoderAppendPrefillRecomputeScheduler",
    ]
)
def scheduler(runtime, request):
    return make_scheduler(runtime, request.param)


def make_scheduler(runtime, class_name, *, prefill_interval=1, lookahead_tokens=0, use_v2_model_runner=False):
    cls = getattr(runtime, class_name)
    # Execute the real installed Scheduler constructor. Only its cache/event
    # factories are faked, so release-only missing fields cannot be hidden by
    # populating an object created with __new__.
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_seqs=16,
            max_num_active_seqs=None,
            max_num_scheduled_tokens=None,
            max_num_batched_tokens=64,
            long_prefill_token_threshold=0,
            long_prefill_token_threshold_adaptive=False,
            enable_chunked_prefill=True,
            policy="fcfs",
            watermark=0.0,
            prefill_schedule_interval=prefill_interval,
            scheduler_reserve_full_isl=False,
        ),
        cache_config=SimpleNamespace(
            num_gpu_blocks=128,
            enable_prefix_caching=True,
            enable_mamba_shared_prefix_checkpoint=False,
            mamba_cache_mode="none",
        ),
        model_config=SimpleNamespace(
            uses_mrope=False,
            is_encoder_decoder=False,
            supports_multimodal_inputs=False,
            max_model_len=256,
            runner_type="generate",
            is_diffusion=False,
            return_sampling_mask=False,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            data_parallel_index=0,
        ),
        observability_config=SimpleNamespace(
            per_request_spec_decode_metrics=None,
            kv_cache_metrics=False,
            enable_mfu_metrics=False,
            enable_logging_iteration_details=False,
        ),
        ec_manager_config=SimpleNamespace(get_encoder_cache_manager_obj=lambda: None),
        aux_output_config=SimpleNamespace(enabled=False),
        kv_events_config=None,
        lora_config=None,
        kv_transfer_config=None,
        ec_transfer_config=None,
        speculative_config=None,
        num_speculative_tokens=0,
        num_lookahead_tokens=lookahead_tokens,
        num_prefill_lookahead_tokens=1,
        max_in_flight_tokens=0,
        is_mm_encoder_only=False,
        use_v2_model_runner=use_v2_model_runner,
    )
    scheduler = cls(
        vllm_config=config,
        kv_cache_config=SimpleNamespace(kv_cache_groups=[], has_mamba_layers=False, needs_kv_cache_zeroing=False),
        structured_output_manager=Mock(accept_tokens=Mock(return_value=True)),
        block_size=16,
        include_finished_set=True,
    )
    empty_blocks = SimpleNamespace(get_block_ids=lambda: ([],), blocks=([],))
    scheduler.__dict__.update(
        current_step=0,
        max_num_scheduled_tokens=64,
        _pause_state=runtime.PauseState.UNPAUSED,
        max_num_encoder_input_tokens=0,
        running=[],
        waiting=runtime.FCFSRequestQueue(),
        kv_holding_waiting=runtime.FCFSRequestQueue(),
        deferred_waiting=set(),
        policy=runtime.SchedulingPolicy.FCFS,
        kv_cache_manager=Mock(),
        prefill_capacity_bound=False,
        max_num_active_reqs=16,
        max_num_running_reqs=16,
        num_waiting_for_streaming_input=0,
        lora_config=None,
        connector=None,
        requires_kv_delivery=False,
        connector_prefix_cache_stats=None,
        needs_kv_cache_zeroing=False,
        ec_connector=None,
        aux_output_connector=None,
        prefix_replay_tokens=0,
        prefix_replay_group_ids=(),
        adaptive_long_prefill_threshold=False,
        _kv_fetch_counts={"waiting_to_start": 0, "in_progress": 0, "completed_waiting": 0},
        _kv_fetch_stages={},
        num_lookahead_tokens=lookahead_tokens,
        is_encoder_decoder=False,
        need_mamba_block_aligned_split=False,
        scheduler_reserve_full_isl=False,
        num_spec_tokens=0,
        num_sampled_tokens_per_step=1,
        num_prefill_lookahead=1,
        max_model_len=256,
        block_size=16,
        log_stats=False,
        return_sampling_mask=False,
        is_mm_encoder_only=False,
        structured_output_manager=Mock(accept_tokens=Mock(return_value=True)),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[None]),
        use_v2_model_runner=use_v2_model_runner,
        model_uses_mrope=False,
        prev_step_scheduled_req_ids=set(),
        reset_preempted_req_ids=set(),
        finished_req_ids=set(),
        finished_req_ids_dict=defaultdict(set),
        encoder_cache_manager=Mock(),
        dynamic_sd_lookup=None,
        defer_block_free=False,
        sched_step_seq=0,
        processed_step_seq=0,
        deferred_frees=deque(),
        _inflight_prefills=set(),
        requests={},
        _pending_decoder_cache_rejections={},
        _decoder_cache_rejection_batches={},
        perf_metrics=None,
        grammar_compile_error_reqs={},
        encoder_cache_mismatch_reqs=set(),
        kv_event_publisher=Mock(),
    )
    manager = scheduler.kv_cache_manager
    manager.get_blocks.return_value = empty_blocks
    manager.get_block_ids.return_value = ([],)
    manager.get_block_ids_for_computed_tokens.return_value = ([],)
    manager.empty_kv_cache_blocks = empty_blocks
    manager.get_computed_blocks.return_value = (empty_blocks, 0, 0)
    manager.allocate_slots.return_value = empty_blocks
    manager.get_num_common_prefix_blocks.return_value = [0]
    manager.take_boundary_state_offloads.return_value = {}
    manager.take_kv_cache_block_copies.return_value = ([], [])
    manager.take_events.return_value = None
    scheduler._make_cached_request_data = Mock(return_value=None)
    scheduler._get_new_block_ids_to_zero = Mock(return_value=None)
    scheduler.make_stats = Mock(return_value=None)
    return scheduler


def enqueue(runtime, scheduler, request_id="miss", *, marked=True, hits=0, output_tokens=0):
    request = TestRequest(
        request_id=request_id,
        sampling_params=SimpleNamespace(
            extra_args={"require_decoder_cache_hit": 1} if marked else {}, num_logprobs=None
        ),
        status=runtime.RequestStatus.PREEMPTED if output_tokens else runtime.RequestStatus.WAITING,
        num_computed_tokens=0,
        num_stale_output_tokens=0,
        num_in_flight_tokens=0,
        num_output_placeholders=0,
        next_decode_eligible_step=0,
        drop_stale_output=False,
        lora_request=None,
        shared_prefix_boundary=0,
        num_tokens=32,
        num_prompt_tokens=32,
        _all_token_ids=[0] * 32,
        prefill_stats=None,
        has_encoder_inputs=False,
        num_output_tokens=output_tokens,
        client_index=2,
        trace_headers={"trace": "test"},
        num_preemptions=int(bool(output_tokens)),
        is_prefill_chunk=True,
        use_structured_output=False,
        spec_token_ids=[],
        skip_reading_prefix_cache=False,
        last_sched_seq=0,
        pooling_params=None,
        stop_reason=None,
        num_nans_in_logits=0,
    )
    scheduler.requests[request_id] = request
    scheduler.waiting.add_request(request)
    scheduler.kv_cache_manager.get_computed_blocks.return_value = (
        scheduler.kv_cache_manager.empty_kv_cache_blocks,
        hits,
        0,
    )
    return request


def empty_model_output():
    return SimpleNamespace(
        sampled_token_ids=[],
        logprobs=None,
        prompt_logprobs_dict={},
        prompt_token_id_logprobs_dict={},
        pooler_output=None,
        num_nans_in_logits=None,
        kv_connector_output=None,
        ec_connector_output=None,
        cudagraph_stats=None,
    )


def test_miss_is_not_allocated_and_zero_token_batch_finishes(runtime, scheduler):
    req = enqueue(runtime, scheduler)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {}
    assert output.total_num_scheduled_tokens == 0
    scheduler.kv_cache_manager.allocate_slots.assert_not_called()
    results = scheduler.update_from_output(output, empty_model_output())
    terminal = results[2].outputs[0]
    assert (terminal.request_id, terminal.new_token_ids, terminal.stop_reason) == ("miss", [], "decoder_cache_miss")
    assert terminal.trace_headers == req.trace_headers
    assert terminal.events == ["queued"]
    assert results[2].finished_requests == {"miss"}
    assert not scheduler.requests and not scheduler.waiting and not scheduler.kv_holding_waiting
    assert not scheduler.deferred_waiting
    scheduler.kv_cache_manager.free.assert_called_once_with(req)
    assert not scheduler._pending_decoder_cache_rejections
    assert not scheduler._decoder_cache_rejection_batches


@pytest.mark.parametrize("marked,hits,expected_tokens", [(False, 0, 32), (True, 16, 16)])
def test_normal_or_resident_request_is_scheduled(runtime, scheduler, marked, hits, expected_tokens):
    req = enqueue(runtime, scheduler, marked=marked, hits=hits)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"miss": expected_tokens}
    assert scheduler.kv_cache_manager.allocate_slots.call_args.kwargs["num_new_computed_tokens"] == hits
    assert not scheduler._pending_decoder_cache_rejections
    assert req.num_in_flight_tokens == expected_tokens
    expected_placeholders = int(isinstance(scheduler, runtime.AsyncScheduler))
    assert req.num_output_placeholders == expected_placeholders


def test_connector_cannot_rescue_a_local_miss(runtime, scheduler):
    enqueue(runtime, scheduler)
    scheduler.connector = Mock(supports_divergent_local_hybrid_hits=True)
    scheduler.connector.get_num_new_matched_tokens.return_value = (32, True)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {}
    scheduler.connector.get_num_new_matched_tokens.assert_not_called()
    scheduler.kv_cache_manager.get_computed_blocks_for_connector.assert_not_called()
    scheduler.connector.update_state_after_alloc.assert_not_called()


def test_marked_hit_also_bypasses_remote_matching(runtime, scheduler):
    enqueue(runtime, scheduler, hits=16)
    scheduler.connector = Mock(supports_divergent_local_hybrid_hits=True)
    scheduler.connector_prefix_cache_stats = None
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"miss": 16}
    scheduler.connector.get_num_new_matched_tokens.assert_not_called()
    scheduler.connector.update_state_after_alloc.assert_called_once()


def test_allocation_retry_rechecks_residency(runtime, scheduler):
    enqueue(runtime, scheduler, hits=16)
    scheduler.kv_cache_manager.allocate_slots.return_value = None
    first = scheduler.schedule()
    assert first.num_scheduled_tokens == {}
    scheduler.kv_cache_manager.get_computed_blocks.return_value = (
        scheduler.kv_cache_manager.empty_kv_cache_blocks,
        0,
        0,
    )
    second = scheduler.schedule()
    assert scheduler.kv_cache_manager.allocate_slots.call_count == 1
    assert not scheduler.update_from_output(first, empty_model_output())
    assert scheduler.update_from_output(second, empty_model_output())[2].outputs[0].stop_reason == "decoder_cache_miss"


def test_pending_rejection_waits_for_its_own_batch(runtime, scheduler):
    first = scheduler.schedule()
    enqueue(runtime, scheduler)
    rejected = scheduler.schedule()
    later = scheduler.schedule()
    assert not scheduler.update_from_output(first, empty_model_output())
    assert "miss" in scheduler.requests
    assert later.num_scheduled_tokens == {}
    assert scheduler.kv_cache_manager.get_computed_blocks.call_count == 1
    assert (
        scheduler.update_from_output(rejected, empty_model_output())[2].outputs[0].stop_reason == "decoder_cache_miss"
    )
    assert not scheduler.update_from_output(later, empty_model_output())


@pytest.mark.parametrize("reuse_id", [False, True])
def test_abort_does_not_emit_rejection_or_finish_reused_id(runtime, scheduler, reuse_id):
    req = enqueue(runtime, scheduler)
    output = scheduler.schedule()
    scheduler.finish_requests("miss", runtime.RequestStatus.FINISHED_ABORTED)
    if reuse_id:
        enqueue(runtime, scheduler)
    results = scheduler.update_from_output(output, empty_model_output())
    assert all(not result.outputs for result in results.values())
    assert req.status == runtime.RequestStatus.FINISHED_ABORTED
    assert bool(scheduler.requests) == reuse_id


def test_post_output_miss_cannot_be_replayed(runtime, scheduler):
    enqueue(runtime, scheduler, output_tokens=3)
    output = scheduler.schedule()
    result = scheduler.update_from_output(output, empty_model_output())[2].outputs[0]
    assert result.stop_reason == "decoder_cache_lost_after_output"
    assert result.new_token_ids == []


def test_deliverable_stale_output_defers_admission(runtime, scheduler):
    req = enqueue(runtime, scheduler)
    req.num_stale_output_tokens = 1
    output = scheduler.schedule()
    scheduler.kv_cache_manager.get_computed_blocks.assert_not_called()
    assert not scheduler.update_from_output(output, empty_model_output())
    req.num_stale_output_tokens = 0
    req.num_output_tokens = 1
    output = scheduler.schedule()
    assert (
        scheduler.update_from_output(output, empty_model_output())[2].outputs[0].stop_reason
        == "decoder_cache_lost_after_output"
    )


def test_miss_does_not_block_an_unmarked_request(runtime, scheduler):
    enqueue(runtime, scheduler)
    enqueue(runtime, scheduler, "normal", marked=False)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"normal": 32}


def test_cache_miss_is_deferred_while_kv_holding_request_is_admitted(runtime, scheduler):
    miss = enqueue(runtime, scheduler)
    held = enqueue(runtime, scheduler, "held", marked=False)
    scheduler.waiting.remove_request(held)
    held.status = runtime.RequestStatus.PREEMPTED
    held.num_computed_tokens = 16
    scheduler.kv_holding_waiting.add_request(held)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {"held": 16}
    assert scheduler.running == [held]
    assert list(scheduler.waiting) == [miss]
    assert not scheduler.kv_holding_waiting
    assert scheduler.deferred_waiting == {miss}
    assert scheduler._pending_decoder_cache_rejections == {miss.request_id: miss}
    assert scheduler._decoder_cache_rejection_batches[id(output)] == (output, [miss])
    model_output = empty_model_output()
    model_output.req_id_to_index = {"held": 0}
    result = scheduler.update_from_output(output, model_output)
    assert [item.request_id for item in result[2].outputs] == ["miss"]
    assert not scheduler.waiting and not scheduler.kv_holding_waiting and not scheduler.deferred_waiting


@pytest.mark.parametrize("hits,rejected", [(0, True), (16, True), (32, False)])
def test_prefix_replay_requires_a_useful_local_prefix(runtime, scheduler, hits, rejected):
    enqueue(runtime, scheduler, hits=hits)
    scheduler.prefix_replay_tokens = 16
    output = scheduler.schedule()
    assert bool(scheduler._pending_decoder_cache_rejections) == rejected
    assert bool(output.num_scheduled_tokens) != rejected


def test_only_documented_admission_deltas_change_schedule():
    """Check the declared copy baseline, independently of the installed lane."""

    def schedule_tree(path, class_name):
        cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == class_name)
        return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "schedule")

    class StripAdmission(ast.NodeTransformer):
        def visit_AnnAssign(self, node):
            if isinstance(node.target, ast.Name) and node.target.id == "rejected_requests":
                return None
            return self.generic_visit(node)

        def visit_If(self, node):
            condition = ast.unparse(node.test)
            if "_pending_decoder_cache_rejections" in condition or "_should_reject_decoder_cache_miss" in condition:
                return None
            if condition == "hasattr(scheduler_output, 'skip_drafts')":
                expected = ast.parse("scheduler_output.skip_drafts = not drafts_needed").body[0]
                assert not node.orelse
                assert len(node.body) == 1 and ast.dump(node.body[0]) == ast.dump(expected)
                return None
            if "_requires_decoder_cache_hit" in condition:
                node.test = node.test.values[0]
            return self.generic_visit(node)

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == "SchedulerOutput":
                assert all(keyword.arg != "skip_drafts" for keyword in node.keywords)
                index = next(i for i, keyword in enumerate(node.keywords) if keyword.arg == "ec_manager_metadata")
                keyword = ast.keyword(arg="skip_drafts", value=ast.parse("not drafts_needed", mode="eval").body)
                node.keywords.insert(index, keyword)
            return self.generic_visit(node)

        def visit_Expr(self, node):
            if "self._record_decoder_cache_rejections(" in ast.unparse(node):
                return None
            return self.generic_visit(node)

    actual = StripAdmission().visit(schedule_tree(SOURCE, "DecoderAppendPrefillScheduler"))

    def canonical_ast(node):
        if isinstance(node, ast.AST):
            # Python 3.12 added empty type_params fields to FunctionDef. Ignore
            # that structural-only difference when checking the same source.
            return [
                type(node).__name__,
                [
                    (name, canonical_ast(value))
                    for name, value in ast.iter_fields(node)
                    if name != "type_params" or value
                ],
            ]
        if isinstance(node, list):
            return [canonical_ast(value) for value in node]
        return node

    docstring = ast.get_docstring(ast.parse(SOURCE.read_text()))
    assert docstring is not None and COPY_BASELINE_VLLM_REVISION in docstring
    encoded = json.dumps(canonical_ast(actual), separators=(",", ":")).encode()
    assert hashlib.sha256(encoded).hexdigest() == COPY_BASELINE_SCHEDULE_SHA256


@pytest.mark.parametrize("async_enabled", [False, True])
def test_constructor_preserves_async_setup(runtime, monkeypatch, async_enabled):
    def base_init(self, *args, **kwargs):
        self.num_spec_tokens = 2
        self.parallel_config = SimpleNamespace(pipeline_parallel_size=1)

    monkeypatch.setattr(runtime.Scheduler, "__init__", base_init)
    config = SimpleNamespace(scheduler_config=SimpleNamespace(async_scheduling=async_enabled))
    cls = runtime.AsyncDecoderAppendPrefillScheduler if async_enabled else runtime.DecoderAppendPrefillScheduler
    instance = cls(vllm_config=config)
    assert instance._pending_decoder_cache_rejections == {}
    assert instance._decoder_cache_rejection_batches == {}
    if async_enabled:
        assert instance._spec_token_placeholders == [-1, -1]


@pytest.mark.parametrize("async_enabled", [False, True])
@pytest.mark.parametrize("work", ["empty", "hit", "chunk", "miss"])
def test_native_constructor_and_output_preserve_version_contract(runtime, async_enabled, work):
    options = dict(prefill_interval=4, lookahead_tokens=1, use_v2_model_runner=True)
    native = make_scheduler(runtime, "Scheduler", **options)
    name = "AsyncDecoderAppendPrefillScheduler" if async_enabled else "DecoderAppendPrefillScheduler"
    scheduler = make_scheduler(runtime, name, **options)
    for attr, default in (
        ("can_skip_drafts", False),
        ("local_prefill_interval", 1),
        ("num_executed_steps", 0),
        ("last_prefill_step", -1),
    ):
        assert getattr(scheduler, attr) == getattr(native, attr, default)
    if work != "empty":
        request = enqueue(runtime, scheduler, hits=0 if work == "miss" else 16)
        request.max_tokens = 10
        if work == "chunk":
            scheduler.max_num_scheduled_tokens = 8

    output = scheduler.schedule()

    # This is the actual dependency's dataclass, not a kwargs-accepting fake.
    assert type(output) is runtime.SchedulerOutput
    expected_tokens = {"hit": {"miss": 16}, "chunk": {"miss": 8}}
    assert output.num_scheduled_tokens == expected_tokens.get(work, {})
    if "skip_drafts" in runtime.SchedulerOutput.__dataclass_fields__:
        assert output.skip_drafts is (work != "hit")
    else:
        assert not hasattr(output, "skip_drafts")
    if work == "miss":
        result = scheduler.update_from_output(output, empty_model_output())
        assert result[2].outputs[0].stop_reason == "decoder_cache_miss"


@pytest.mark.parametrize("async_enabled", [False, True])
def test_native_prefill_cadence_is_retained_only_when_upstream_supports_it(runtime, async_enabled):
    native = make_scheduler(runtime, "Scheduler", prefill_interval=3)
    name = "AsyncDecoderAppendPrefillScheduler" if async_enabled else "DecoderAppendPrefillScheduler"
    scheduler = make_scheduler(runtime, name, prefill_interval=3)
    decode = enqueue(runtime, scheduler, "decode", marked=False)
    scheduler.waiting.remove_request(decode)
    decode.status = runtime.RequestStatus.RUNNING
    decode.is_prefill_chunk = False
    decode.num_computed_tokens = 32
    decode.num_tokens = decode.num_tokens_with_spec = 33
    scheduler.running.append(decode)
    enqueue(runtime, scheduler, "append", hits=16)
    scheduler.num_executed_steps = scheduler.last_prefill_step = 1

    output = scheduler.schedule()

    assert output.num_scheduled_tokens["decode"] == 1
    if getattr(native, "local_prefill_interval", 1) > 1:
        assert "append" not in output.num_scheduled_tokens
        assert list(scheduler.waiting)[0].request_id == "append"
        assert scheduler.last_prefill_step == 1
    else:
        assert output.num_scheduled_tokens["append"] == 16
    assert scheduler.num_executed_steps == 2


def test_inflight_preempted_output_is_delivered_before_cache_loss(runtime, scheduler):
    req = enqueue(runtime, scheduler, hits=16)
    first = scheduler.schedule()
    scheduler.running.remove(req)
    scheduler._preempt_request(req, time.monotonic())
    scheduler.kv_cache_manager.get_computed_blocks.return_value = (
        scheduler.kv_cache_manager.empty_kv_cache_blocks,
        0,
        0,
    )
    waiting = scheduler.schedule()
    assert not scheduler._pending_decoder_cache_rejections
    model_output = empty_model_output()
    model_output.sampled_token_ids = [[42]]
    model_output.req_id_to_index = {"miss": 0}
    result = scheduler.update_from_output(first, model_output)
    assert result[2].outputs[0].new_token_ids == [42]
    assert req.num_stale_output_tokens == req.num_output_placeholders == 0
    assert not scheduler.update_from_output(waiting, empty_model_output())
    rejected = scheduler.schedule()
    result = scheduler.update_from_output(rejected, empty_model_output())
    assert result[2].outputs[0].stop_reason == "decoder_cache_lost_after_output"


def test_mixed_batch_delivers_normal_output_and_rejection(runtime, scheduler):
    enqueue(runtime, scheduler)
    normal = enqueue(runtime, scheduler, "normal", marked=False)
    output = scheduler.schedule()
    model_output = empty_model_output()
    model_output.sampled_token_ids = [[42]]
    model_output.req_id_to_index = {"normal": 0}
    result = scheduler.update_from_output(output, model_output)
    assert [(item.request_id, item.new_token_ids) for item in result[2].outputs] == [("normal", [42]), ("miss", [])]
    assert scheduler.running == [normal]
    assert set(scheduler.requests) == {"normal"}
    if isinstance(scheduler, runtime.AsyncScheduler):
        assert normal.num_output_placeholders == 0
        scheduler.kv_cache_manager.cache_blocks.assert_called_once_with(normal, 32)


@pytest.fixture(params=["DecoderAppendPrefillRecomputeScheduler", "AsyncDecoderAppendPrefillRecomputeScheduler"])
def combined(runtime, request):
    return make_scheduler(runtime, request.param)


def recovery_connector(*, pending=False, available=True):
    return Mock(
        has_preempted_request=Mock(return_value=available),
        get_num_new_matched_tokens=Mock(return_value=(None, False) if pending else (16, True)),
        requires_kv_delivery=False,
        request_finished=Mock(return_value=(False, None)),
        take_events=Mock(return_value=None),
    )


@pytest.mark.parametrize("pending", [False, True])
def test_recompute_restores_same_request_without_local_hit(runtime, combined, pending):
    req = enqueue(runtime, combined, output_tokens=1)
    combined.connector = recovery_connector(pending=pending)
    result = combined.schedule()
    assert not result.num_scheduled_tokens
    assert not combined._pending_decoder_cache_rejections
    combined.connector.get_num_new_matched_tokens.assert_called_once_with(req, 0)
    if pending:
        combined.kv_cache_manager.allocate_slots.assert_not_called()
    else:
        assert req.status == runtime.RequestStatus.WAITING_FOR_REMOTE_KVS
        assert combined.kv_cache_manager.allocate_slots.call_args.kwargs["delay_cache_blocks"]
        assert req.num_computed_tokens == 16


def test_recompute_does_not_treat_preemption_count_as_recovery_state(runtime, combined):
    enqueue(runtime, combined, output_tokens=1)
    combined.connector = recovery_connector(available=False)
    batch = combined.schedule()
    combined.connector.get_num_new_matched_tokens.assert_not_called()
    result = combined.update_from_output(batch, empty_model_output())
    assert result[2].outputs[0].stop_reason == "decoder_cache_lost_after_output"


@pytest.mark.parametrize("marked", [False, True])
@pytest.mark.parametrize("offload", ["saved", "full", "exception"])
def test_recompute_preemption_preserves_protocol_and_batch_order(runtime, combined, marked, offload):
    req = enqueue(runtime, combined, marked=marked, hits=16)
    earlier = combined.schedule()
    combined.running.remove(req)
    combined.connector = recovery_connector()
    hook = combined.connector.update_state_before_preempt
    hook.return_value = offload == "saved"
    if offload == "exception":
        hook.side_effect = RuntimeError("CPU offload failed")
    combined._recomputed_reqs = []
    combined._step_decoder_rejections = []
    saved = combined._preempt_or_recompute(req, 1.0)
    assert saved == (offload == "saved")
    if saved:
        assert req.status == runtime.RequestStatus.PREEMPTED
        assert not combined._step_decoder_rejections and not combined._recomputed_reqs
        return
    batch = Output(recomputed_reqs=combined._recomputed_reqs, num_scheduled_tokens={}, total_num_scheduled_tokens=0)
    combined._record_decoder_cache_rejections(batch, combined._step_decoder_rejections)
    if marked:
        # The original in-flight batch must still be delivered before rejection.
        output = empty_model_output()
        output.sampled_token_ids = [[42]]
        output.req_id_to_index = {req.request_id: 0}
        result = combined.update_from_output(earlier, output)
        assert result[2].outputs[0].new_token_ids == [42]
        assert not combined._recomputed_reqs
    result = combined.update_from_output(batch, empty_model_output())
    assert [out.stop_reason for out in result[2].outputs] == [
        "decoder_cache_lost_after_output" if marked else "recomputed"
    ]
    assert req.request_id not in combined.requests


@pytest.mark.parametrize("outputs", [0, 1])
def test_failed_restore_terminates_without_scheduling_local_recompute(runtime, combined, outputs):
    req = enqueue(runtime, combined, output_tokens=outputs)
    req.status = runtime.RequestStatus.WAITING_FOR_REMOTE_KVS
    req.num_preemptions = 1
    combined.waiting.remove_request(req)
    combined.kv_holding_waiting.add_request(req)
    combined.deferred_waiting.add(req)
    combined.connector = recovery_connector(available=False)
    combined.failed_recving_kv_req_ids = {req.request_id}
    combined.finished_recving_kv_req_ids = {req.request_id}
    combined.needs_kv_cache_zeroing = False
    batch = combined.schedule()
    assert not batch.num_scheduled_tokens
    result = combined.update_from_output(batch, empty_model_output())
    assert result[2].outputs[0].stop_reason == ("decoder_cache_lost_after_output" if outputs else "decoder_cache_miss")
    combined.kv_cache_manager.allocate_slots.assert_not_called()
    assert not combined.waiting and not combined.kv_holding_waiting and not combined.deferred_waiting


@pytest.mark.parametrize("decode_first", [False, True])
def test_combined_mixed_batch_does_not_pad_decode_in_either_order(runtime, combined, decode_first):
    combined.num_spec_tokens = 2
    combined._spec_token_placeholders = [-1, -1]
    names = ["decode", "append"] if decode_first else ["append", "decode"]
    for name in names:
        req = enqueue(runtime, combined, name, hits=16)
        if name == "decode":
            req.num_tokens = req.num_prompt_tokens = 17
    batch = combined.schedule()
    assert batch.num_scheduled_tokens == {"decode": 1, "append": 16}
    # Async scheduling may add draft placeholders after scheduling; they must
    # not turn this first decode into a padded three-token allocation.
    allocations = {
        call.args[0].request_id: call.args[1] for call in combined.kv_cache_manager.allocate_slots.call_args_list
    }
    assert allocations == {"decode": 1, "append": 16}


def test_plain_recompute_still_pads_first_decode(runtime):
    scheduler = make_scheduler(runtime, "RecomputeScheduler")
    scheduler.num_spec_tokens = 2
    req = enqueue(runtime, scheduler, marked=False, hits=16)
    req.num_tokens = req.num_prompt_tokens = 17
    assert scheduler.schedule().num_scheduled_tokens == {req.request_id: 3}


def execute_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    # Exclude heavyweight imports inside platform validation, whose objects
    # are supplied explicitly by the fixture instead.
    for node in nodes:
        node.body = [statement for statement in node.body if not isinstance(statement, (ast.Import, ast.ImportFrom))]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


@pytest.fixture
def platform_runtime():
    ns = {
        "logger": Mock(),
        "_validate_kv_load_failure_policy": lambda config: None,
        "refresh_block_size": Mock(),
        "_downgrade_swa_bounded_replay": Mock(),
        "get_current_hardware_profile": lambda: SimpleNamespace(supports=lambda _: False),
        "HardwareCapability": SimpleNamespace(AUTO_ENABLE_CUSTOM_OPS="auto_enable_custom_ops"),
    }
    ns["RecomputeSchedulerConfig"] = SimpleNamespace(
        initialize_from_config=lambda config: SimpleNamespace(**vars(config.scheduler_config))
    )
    execute_functions(
        ROOT / "vllm_ascend/platform.py",
        {
            "_is_decoder_append_prefill_scheduler",
            "_get_recompute_scheduler_cls",
            "_get_decoder_append_prefill_scheduler_cls",
            "_get_dyntra_lb_scheduler_cls",
            "_check_ascend_config",
            "_setup_worker_and_scheduler",
            "_validate_decoder_append_prefill_policies",
        },
        ns,
    )
    return SimpleNamespace(**ns)


def make_platform_config(
    *, async_enabled=False, role="kv_consumer", selected=None, recompute=False, dyntra=False, append_prefill=False
):
    config = SimpleNamespace(
        additional_config={},
        kv_transfer_config=None if role is None else SimpleNamespace(kv_role=role, has_connector=lambda _: False),
        scheduler_config=SimpleNamespace(scheduler_cls=selected, async_scheduling=async_enabled, policy="fcfs"),
        parallel_config=SimpleNamespace(
            data_parallel_size=2 if dyntra else 1, nnodes_within_dp=1, worker_cls="test.Worker"
        ),
    )
    policies = SimpleNamespace(
        enable_balance_scheduling=False,
        recompute_scheduler_enable=recompute,
        decoder_append_prefill_enable=append_prefill,
        **{
            key: SimpleNamespace(enabled=False)
            for key in (
                "dyntra_lb_config",
                "short_request_first_config",
                "profiling_chunk_config",
                "batch_job_sched_config",
            )
        },
    )
    policies.dyntra_lb_config.enabled = dyntra
    return config, SimpleNamespace(
        scheduler_config=policies,
        finegrained_tp_config=SimpleNamespace(oproj_tensor_parallel_size=1, mlp_tensor_parallel_size=1),
    )


@pytest.mark.parametrize("async_enabled", [False, True])
@pytest.mark.parametrize("as_type", [False, True])
@pytest.mark.parametrize("combined_selected", [False, True])
def test_platform_combines_explicit_append_scheduler_and_rejects_overriding_policies(
    platform_runtime, async_enabled, as_type, combined_selected
):
    name = "DecoderAppendPrefillRecomputeScheduler" if combined_selected else "DecoderAppendPrefillScheduler"
    if async_enabled:
        name = f"Async{name}"
    module = "vllm_ascend.core.decoder_append_prefill_scheduler"
    selected = type(name, (), {"__module__": module}) if as_type else f"{module}.{name}"
    config, ascend_config = make_platform_config(async_enabled=async_enabled, selected=selected, recompute=True)
    policies = ascend_config.scheduler_config
    platform_runtime._check_ascend_config(config, ascend_config)
    assert config.scheduler_config.scheduler_cls.endswith(
        ".AsyncDecoderAppendPrefillRecomputeScheduler" if async_enabled else ".DecoderAppendPrefillRecomputeScheduler"
    )
    for key, reason in (
        ("dyntra_lb_config", "decode-only"),
        ("short_request_first_config", "PD-disaggregated Decoders"),
        ("profiling_chunk_config", "PP>1"),
        ("batch_job_sched_config", "long-input, short-output agentic"),
    ):
        getattr(policies, key).enabled = True
        with pytest.raises(ValueError, match=reason):
            platform_runtime._check_ascend_config(config, ascend_config)
        getattr(policies, key).enabled = False
    policies.enable_balance_scheduling = True
    with pytest.raises(ValueError, match="PD-mixed"):
        platform_runtime._check_ascend_config(config, ascend_config)
    policies.enable_balance_scheduling = False
    policies.recompute_scheduler_enable = False
    with pytest.raises(ValueError, match="requires recompute_scheduler_enable"):
        platform_runtime._check_ascend_config(config, ascend_config)
    policies.recompute_scheduler_enable = True
    config.scheduler_config.async_scheduling = not async_enabled
    with pytest.raises(ValueError, match="matching async_scheduling"):
        platform_runtime._check_ascend_config(config, ascend_config)
    config.scheduler_config.async_scheduling = async_enabled
    config.kv_transfer_config.kv_role = "kv_both"
    with pytest.raises(ValueError, match="D nodes"):
        platform_runtime._check_ascend_config(config, ascend_config)


@pytest.mark.parametrize("composite", [False, True])
def test_combined_scheduler_reports_actual_recovery_state(combined, cache_runtime, composite):
    manager = cache_runtime.PreemptOffloadScheduler.__new__(cache_runtime.PreemptOffloadScheduler)
    manager._preempted_req_states = {"A": SimpleNamespace(ready=False)}
    provider = SimpleNamespace(has_preempted_request=Mock(wraps=manager.has_preempted_request))
    connector = provider
    if composite:
        ns: dict[str, Any] = {
            "KVConnectorBase_V1": type("KVConnectorBase_V1", (), {}),
            "SupportsHMA": type("SupportsHMA", (), {}),
        }
        execute_classes(VLLM / "distributed/kv_transfer/kv_connector/v1/multi_connector.py", {"MultiConnector"}, ns)
        connector = ns["MultiConnector"].__new__(ns["MultiConnector"])
        connector._connectors = [object(), provider]
    combined.connector = connector

    request = SimpleNamespace(request_id="A", num_preemptions=1)
    # Recovery state may exist while its D2H store is still pending.
    assert combined._is_restoring_decoder_request(request)
    request.request_id = "B"
    assert not combined._is_restoring_decoder_request(request)

    request.request_id = "A"
    request.num_preemptions = 0
    provider.has_preempted_request.reset_mock()
    assert not combined._is_restoring_decoder_request(request)
    provider.has_preempted_request.assert_not_called()

    request.num_preemptions = 1
    manager._preempted_req_states.clear()
    assert not combined._is_restoring_decoder_request(request)
    combined.connector = object()
    assert not combined._is_restoring_decoder_request(request)


@pytest.fixture
def cache_runtime():
    """Load real CPU block-pool/offload logic without importing torch/NPU."""
    import contextlib
    import itertools
    from collections.abc import Sequence
    from dataclasses import dataclass, field

    ns = dict(
        __name__=__name__,
        dataclass=dataclass,
        field=field,
        ABC=ABC,
        abstractmethod=abstractmethod,
        logger=Mock(),
        MEDIUM_GPU="GPU",
        BlockHash=bytes,
        BlockHashWithGroupId=bytes,
        defaultdict=defaultdict,
        contextlib=contextlib,
        itertools=itertools,
        Sequence=Sequence,
        cdiv=lambda a, b: (a + b - 1) // b,
        FullAttentionSpec=type("FullAttentionSpec", (), {}),
        ChunkedLocalAttentionSpec=type("ChunkedLocalAttentionSpec", (), {}),
        UniformTypeKVCacheSpecs=type("UniformTypeKVCacheSpecs", (), {}),
        CircularBufferSpec=type("CircularBufferSpec", (), {}),
    )
    utils = VLLM / "v1/core/kv_cache_utils.py"
    execute_classes(utils, {"KVCacheBlock", "FreeKVCacheBlockQueue"}, ns)
    execute_functions(utils, {"make_block_hash_with_group_id", "get_block_hash", "resolve_block_hashes"}, ns)
    execute_functions(utils, {"resolve_dcp_kv_block_size"}, ns)
    execute_functions(ROOT / "vllm_ascend/core/kv_cache_interface.py", {"is_circular_kv_cache_spec"}, ns)
    execute_classes(VLLM / "v1/core/block_pool.py", {"BlockHashToBlockMap", "BlockPool"}, ns)
    execute_classes(
        VLLM / "v1/core/single_type_kv_cache_manager.py", {"SingleTypeKVCacheManager", "FullAttentionManager"}, ns
    )
    execute_classes(
        ROOT / "vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/preempt_offload/manager.py",
        {"TransferMeta", "PreemptedRequestState", "PreemptOffloadScheduler"},
        ns,
    )
    return SimpleNamespace(**ns)


@pytest.mark.parametrize("evict_before_b", [False, True])
def test_restored_a_finishes_and_b_reuses_new_npu_blocks(runtime, combined, cache_runtime, evict_before_b):
    """Exercise real offload metadata, hash registration and eviction; no device I/O."""
    cr = cache_runtime
    pool = cr.BlockPool(6, enable_caching=True, hash_block_size=16)
    cpu_pool = cr.BlockPool(6, enable_caching=True, hash_block_size=16)
    req = enqueue(runtime, combined, "A", hits=16)
    req.block_hashes = [b"prefix-0", b"prefix-1"]
    req.num_tokens = req.num_prompt_tokens = 33
    old = pool.get_new_blocks(2)
    pool.cache_full_blocks(req, old, 0, 2, 16, 0)
    manager = cr.PreemptOffloadScheduler.__new__(cr.PreemptOffloadScheduler)
    spec = cr.FullAttentionSpec()
    spec.block_size = 16
    spec.dcp_sharded = False
    manager.__dict__.update(
        _gpu_block_pool=pool,
        cpu_block_pool=cpu_pool,
        enable_offload_prefix_caching=True,
        cpu_kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)]),
        _group_is_mamba=[False],
        _group_is_sliding_window=[False],
        _group_has_mamba_conv=[False],
        dcp_world_size=1,
        num_spec_tokens=0,
        _preempted_req_states={},
        _pending_hash_blocks={},
        _preempt_store_event_to_blocks={},
        _preempt_store_event_to_reqs={},
        _preempt_load_event_to_reqs={},
    )
    tables = {"A": old}
    combined.kv_cache_manager.free.side_effect = lambda request: pool.free_blocks(tables.pop(request.request_id, []))
    combined.kv_cache_manager.get_block_ids.side_effect = lambda request_id: ([b.block_id for b in tables[request_id]],)
    combined.waiting.pop_request()
    req.status = runtime.RequestStatus.RUNNING
    req.num_computed_tokens = 32
    combined.connector = recovery_connector()
    combined.connector.update_state_before_preempt.side_effect = manager.update_state_before_preempt
    combined.connector.has_preempted_request.side_effect = manager.has_preempted_request
    assert combined._preempt_or_recompute(req, 1.0)
    state = manager._preempted_req_states["A"]
    # Complete the simulated D2H event before allowing the source to be reused.
    manager._preempt_store_event_to_blocks[0] = state.store_transfer_meta
    manager._preempt_store_event_to_reqs[0] = ["A"]
    manager._process_preempt_store_event(0)
    assert state.ready
    unrelated = pool.get_new_blocks(pool.get_num_free_blocks())
    assert pool.get_cached_block(req.block_hashes[0], [0]) is None
    # Release different blocks for restore, retaining A's old IDs elsewhere.
    reusable = [block for block in unrelated if block not in old][:2]
    pool.free_blocks(reusable)
    restored = pool.get_new_blocks(2)
    assert not {b.block_id for b in old} & {b.block_id for b in restored}
    tables["A"] = restored
    assert manager.get_num_new_matched_tokens(req, 0) == (32, True)
    manager.update_state_after_alloc(req, SimpleNamespace(get_block_ids=lambda: ([b.block_id for b in restored],)), 32)
    assert all(block.ref_cnt == 2 for block in restored)  # Request + H2D hold.
    state.load_event = 1
    manager._preempt_load_event_to_reqs[1] = ["A"]
    manager._cleanup_preempt_load_request("A")
    assert all(block.ref_cnt == 1 for block in restored)
    assert not manager.has_preempted_request("A")
    # Real Scheduler restore path must register the newly allocated NPU blocks.
    combined.kv_cache_manager.cache_blocks.side_effect = lambda request, tokens: pool.cache_full_blocks(
        request, tables[request.request_id], 0, tokens // 16, 16, 0
    )
    req.status = runtime.RequestStatus.WAITING_FOR_REMOTE_KVS
    req.num_computed_tokens = 32
    combined.finished_recving_kv_req_ids = {"A"}
    combined.failed_recving_kv_req_ids = set()
    combined._step_decoder_rejections = []
    combined._update_waiting_for_remote_kv(req)
    assert pool.get_cached_block(req.block_hashes[0], [0]) == [restored[0]]
    # Terminal decode output follows the real Scheduler completion path.
    combined.waiting.pop_request()
    req.status = runtime.RequestStatus.RUNNING
    req.resumable = False
    req.spec_decode_metrics = None
    req.num_in_flight_tokens = 1
    req.num_output_placeholders = int(isinstance(combined, runtime.AsyncScheduler))
    combined.running = [req]
    # The restored prefix is already registered; subsequent cache calls need
    # only track full blocks beyond it (none in this one-token completion).
    combined.kv_cache_manager.cache_blocks.side_effect = None
    old_check_stop = runtime.Scheduler._update_request_with_output.__globals__["check_stop"]

    def stop(request, max_model_len):
        request.status = runtime.RequestStatus.FINISHED_STOPPED
        return True

    runtime.Scheduler._update_request_with_output.__globals__["check_stop"] = stop
    output = empty_model_output()
    output.sampled_token_ids = [[42]]
    output.req_id_to_index = {"A": 0}
    try:
        combined.update_from_output(
            Output(
                num_scheduled_tokens={"A": 1},
                total_num_scheduled_tokens=1,
                recomputed_reqs=None,
                scheduled_spec_decode_tokens={},
            ),
            output,
        )
    finally:
        runtime.Scheduler._update_request_with_output.__globals__["check_stop"] = old_check_stop
    assert "A" not in combined.requests
    assert all(block.ref_cnt == 0 for block in restored)
    if evict_before_b:
        pool.get_new_blocks(pool.get_num_free_blocks())
    # B's actual APC lookup traverses real hashes and the real block pool.
    blocks, hits = cr.FullAttentionManager.find_longest_cache_hit(req.block_hashes, 32, [0], pool, spec, False, 16)
    assert hits == (0 if evict_before_b else 32)
    b = enqueue(runtime, combined, "B", hits=hits)
    combined.kv_cache_manager.get_computed_blocks.return_value = (SimpleNamespace(blocks=blocks), hits, 0)
    b.num_tokens = b.num_prompt_tokens = 48
    result = combined.schedule()
    assert result.num_scheduled_tokens == ({} if evict_before_b else {"B": 16})
    assert bool(combined._pending_decoder_cache_rejections) == evict_before_b


@pytest.mark.parametrize("async_enabled", [False, True])
def test_combined_constructor_initializes_state(runtime, monkeypatch, async_enabled):
    def init(self, *args, **kwargs):
        self.num_spec_tokens = 2
        self.parallel_config = SimpleNamespace(pipeline_parallel_size=1)

    monkeypatch.setattr(runtime.Scheduler, "__init__", init)
    cls = (
        runtime.AsyncDecoderAppendPrefillRecomputeScheduler
        if async_enabled
        else runtime.DecoderAppendPrefillRecomputeScheduler
    )
    config = SimpleNamespace(scheduler_config=SimpleNamespace(async_scheduling=async_enabled))
    obj = cls(vllm_config=config)
    assert obj._pending_decoder_cache_rejections == obj._decoder_cache_rejection_batches == {}
    if async_enabled:
        assert obj._spec_token_placeholders == [-1, -1]


@pytest.fixture
def upstream_scheduler_selector(runtime, monkeypatch):
    source = VLLM / "config/scheduler.py"
    definition = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "SchedulerConfig"
    )
    method = next(
        node for node in definition.body if isinstance(node, ast.FunctionDef) and node.name == "get_scheduler_cls"
    )
    namespace: dict[str, Any] = {}
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    for module_name, class_name in (
        ("vllm.v1.core.sched.scheduler", "Scheduler"),
        ("vllm.v1.core.sched.async_scheduler", "AsyncScheduler"),
    ):
        imported_module = ModuleType(module_name)
        setattr(imported_module, class_name, getattr(runtime, class_name))
        monkeypatch.setitem(sys.modules, module_name, imported_module)
    return namespace["get_scheduler_cls"]


@pytest.mark.parametrize("async_enabled", [False, True])
@pytest.mark.parametrize("role", [None, "kv_producer", "kv_consumer", "kv_both"])
def test_unspecified_scheduler_uses_upstream_default(
    runtime, platform_runtime, upstream_scheduler_selector, async_enabled, role
):
    config, ascend_config = make_platform_config(async_enabled=async_enabled, role=role)
    platform_runtime._check_ascend_config(config, ascend_config)
    platform_runtime._setup_worker_and_scheduler(config, ascend_config)
    assert config.scheduler_config.scheduler_cls is None
    assert upstream_scheduler_selector(config.scheduler_config) is (
        runtime.AsyncScheduler if async_enabled else runtime.Scheduler
    )


@pytest.mark.parametrize("async_enabled", [False, True])
@pytest.mark.parametrize("role", [None, "kv_producer", "kv_consumer", "kv_both"])
@pytest.mark.parametrize("as_type", [False, True])
def test_custom_scheduler_without_recompute_is_preserved(platform_runtime, async_enabled, role, as_type):
    selected = type("Scheduler", (), {"__module__": "custom"}) if as_type else "custom.Scheduler"
    config, ascend_config = make_platform_config(async_enabled=async_enabled, role=role, selected=selected)
    platform_runtime._check_ascend_config(config, ascend_config)
    platform_runtime._setup_worker_and_scheduler(config, ascend_config)
    assert config.scheduler_config.scheduler_cls is selected


@pytest.mark.parametrize("async_enabled", [False, True])
@pytest.mark.parametrize("as_type", [False, True])
def test_explicit_append_scheduler_without_recompute_is_preserved(platform_runtime, async_enabled, as_type):
    name = "AsyncDecoderAppendPrefillScheduler" if async_enabled else "DecoderAppendPrefillScheduler"
    module = "vllm_ascend.core.decoder_append_prefill_scheduler"
    selected = type(name, (), {"__module__": module}) if as_type else f"{module}.{name}"
    config, ascend_config = make_platform_config(async_enabled=async_enabled, selected=selected)
    platform_runtime._check_ascend_config(config, ascend_config)
    platform_runtime._setup_worker_and_scheduler(config, ascend_config)
    assert config.scheduler_config.scheduler_cls is selected


@pytest.mark.parametrize(
    "name",
    ["RecomputeScheduler", "AsyncRecomputeScheduler", "DyntraLBRecomputeScheduler", "AsyncDyntraLBRecomputeScheduler"],
)
@pytest.mark.parametrize("selection", ["default", "path", "class"])
def test_existing_recompute_configuration_does_not_enable_admission(platform_runtime, name, selection):
    module = "vllm_ascend.core.recompute_scheduler"
    selected: str | type | None = None
    if selection == "path":
        selected = f"{module}.{name}"
    elif selection == "class":
        selected = type(name, (), {"__module__": module})
    config, ascend_config = make_platform_config(
        async_enabled=name.startswith("Async"), selected=selected, recompute=True, dyntra="DyntraLB" in name
    )
    platform_runtime._check_ascend_config(config, ascend_config)
    platform_runtime._setup_worker_and_scheduler(config, ascend_config)
    assert config.scheduler_config.scheduler_cls == f"{module}.{name}"
    assert not platform_runtime._is_decoder_append_prefill_scheduler(config.scheduler_config.scheduler_cls)


@pytest.mark.parametrize("async_enabled", [False, True])
@pytest.mark.parametrize("as_type", [False, True])
@pytest.mark.parametrize("dyntra", [False, True])
def test_recompute_preserves_baseline_custom_scheduler_override(platform_runtime, async_enabled, as_type, dyntra):
    selected = type("Scheduler", (), {"__module__": "custom"}) if as_type else "custom.Scheduler"
    config, ascend_config = make_platform_config(
        async_enabled=async_enabled, selected=selected, recompute=True, dyntra=dyntra
    )
    platform_runtime._check_ascend_config(config, ascend_config)
    platform_runtime._setup_worker_and_scheduler(config, ascend_config)
    name = "DyntraLBRecomputeScheduler" if dyntra else "RecomputeScheduler"
    if async_enabled:
        name = f"Async{name}"
    assert config.scheduler_config.scheduler_cls == f"vllm_ascend.core.recompute_scheduler.{name}"
    assert not platform_runtime._is_decoder_append_prefill_scheduler(config.scheduler_config.scheduler_cls)


@pytest.mark.parametrize("async_enabled", [False, True])
@pytest.mark.parametrize("as_type", [False, True])
def test_prefill_ignores_recompute_and_preserves_custom_scheduler(platform_runtime, async_enabled, as_type):
    selected = type("Scheduler", (), {"__module__": "custom"}) if as_type else "custom.Scheduler"
    config, ascend_config = make_platform_config(
        async_enabled=async_enabled, role="kv_producer", selected=selected, recompute=True
    )
    platform_runtime._check_ascend_config(config, ascend_config)
    platform_runtime._setup_worker_and_scheduler(config, ascend_config)
    assert config.scheduler_config.scheduler_cls is selected
    assert config.additional_config["scheduler_config"]["recompute_scheduler_enable"] is False


@pytest.mark.parametrize("async_enabled", [False, True])
@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("selection", ["default", "path", "class"])
def test_append_prefill_config_selects_scheduler_like_dyntra_lb(platform_runtime, async_enabled, recompute, selection):
    selected: str | type | None = None
    if selection == "path":
        selected = "custom.Scheduler"
    elif selection == "class":
        selected = type("Scheduler", (), {"__module__": "custom"})
    for feature in ("append_prefill", "dyntra"):
        config, ascend_config = make_platform_config(
            async_enabled=async_enabled, selected=selected, recompute=recompute, **{feature: True}
        )
        platform_runtime._check_ascend_config(config, ascend_config)
        platform_runtime._setup_worker_and_scheduler(config, ascend_config)
        if feature == "append_prefill":
            module = "vllm_ascend.core.decoder_append_prefill_scheduler"
            name = "DecoderAppendPrefillRecomputeScheduler" if recompute else "DecoderAppendPrefillScheduler"
        elif recompute:
            module = "vllm_ascend.core.recompute_scheduler"
            name = "DyntraLBRecomputeScheduler"
        else:
            module = "vllm_ascend.core.dyntra_lb_scheduler"
            name = "DyntraLBScheduler"
        if async_enabled:
            name = f"Async{name}"
        assert config.scheduler_config.scheduler_cls == f"{module}.{name}"


@pytest.mark.parametrize("role", [None, "kv_producer", "kv_both"])
@pytest.mark.parametrize("recompute", [False, True])
def test_append_prefill_config_rejects_non_decoder_nodes(platform_runtime, role, recompute):
    config, ascend_config = make_platform_config(role=role, recompute=recompute, append_prefill=True)
    with pytest.raises(ValueError, match="kv_consumer"):
        platform_runtime._check_ascend_config(config, ascend_config)
    assert config.scheduler_config.scheduler_cls is None


@pytest.mark.parametrize(
    ("policy", "reason"),
    [
        ("enable_balance_scheduling", "PD-mixed"),
        ("dyntra_lb_config", "decode-only"),
        ("short_request_first_config", "PD-disaggregated Decoders"),
        ("profiling_chunk_config", "PP>1"),
        ("batch_job_sched_config", "long-input, short-output agentic"),
    ],
)
@pytest.mark.parametrize("recompute", [False, True])
def test_append_prefill_config_rejects_conflicting_policy_before_selection(platform_runtime, policy, reason, recompute):
    config, ascend_config = make_platform_config(recompute=recompute, append_prefill=True)
    policies = ascend_config.scheduler_config
    if policy == "enable_balance_scheduling":
        policies.enable_balance_scheduling = True
    else:
        getattr(policies, policy).enabled = True
    with pytest.raises(ValueError, match=reason):
        platform_runtime._check_ascend_config(config, ascend_config)
    assert config.scheduler_config.scheduler_cls is None
