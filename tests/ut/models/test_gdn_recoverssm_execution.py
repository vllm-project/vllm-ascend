# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduling, ownership and graph metadata contracts; no NPU execution."""

import inspect
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.ascend_config import GDNRecoverSSMExecutionConfig
from vllm_ascend.attention import gdn_recoverssm_execution as execution
from vllm_ascend.attention import recoverssm_gdn as binding
from vllm_ascend.attention.gdn_recoverssm_metadata import AscendGDNRecoverSSMMetadata
from vllm_ascend.worker import worker as worker_module
from vllm_ascend.worker.v2 import model_runner as runner_module
from vllm_ascend.worker.worker import NPUWorker


def job(indices=None, *, align=False, context=None):
    return execution.GDNCommitJob(
        context=context or NS(commit=Mock(), commit_plan=NS(capacity=8)),
        indices=indices if indices is not None else torch.tensor([1, 2, 3], dtype=torch.int32),
        offsets=torch.tensor([0, 4, 8, 12], dtype=torch.int32),
        mapping=None,
        table=torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32) if align else None,
        computed=torch.tensor([7, 15, 23], dtype=torch.int32) if align else None,
        block_size=8 if align else None,
    )


def config(**kwargs):
    return GDNRecoverSSMExecutionConfig(**kwargs)


def vconfig():
    return NS(
        parallel_config=NS(pipeline_parallel_size=1, prefill_context_parallel_size=1, decode_context_parallel_size=1),
        cache_config=NS(use_replayssm=True),
        model_config=NS(architecture="Qwen4ExpForConditionalGeneration"),
        speculative_config=NS(method="mtp"),
    )


def test_typed_config_defaults_and_validation():
    assert not config().enabled
    assert not config(async_commit="false").async_commit
    config(async_commit=True, capture_commit=True).validate(vconfig())
    for name in ("pipeline_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size"):
        vc = vconfig()
        setattr(vc.parallel_config, name, 2)
        with pytest.raises(ValueError, match="PP=PCP=DCP"):
            config(capture_commit=True).validate(vc)
    vc = vconfig()
    vc.cache_config.use_replayssm = False
    with pytest.raises(ValueError, match="use-replayssm"):
        config(async_commit=True).validate(vc)
    vc = vconfig()
    vc.speculative_config.method = "eagle3"
    with pytest.raises(ValueError, match="MTP drafter"):
        config(async_commit=True).validate(vc)


def test_internal_mtp_config_does_not_install_target_recovery_policy():
    vc = vconfig()
    draft = NS(architecture="Qwen4ExpMTP")
    vc.speculative_config.draft_model_config = draft
    vc.model_config = draft
    cfg = config(async_commit=True, capture_commit=True)
    assert cfg.is_draft_config(vc)
    cfg.validate(vc)
    # Architecture names alone cannot bypass target validation.
    vc.model_config = NS(architecture="Qwen4ExpMTP")
    assert not cfg.is_draft_config(vc)
    with pytest.raises(ValueError, match="Qwen4Exp MTP only"):
        cfg.validate(vc)


@pytest.mark.parametrize("batch,capacity,expected", [(1, 192, 1), (3, 192, 4), (125, 192, 128), (160, 192, 192)])
def test_bounded_graph_buckets(batch, capacity, expected):
    assert execution.commit_bucket(batch, capacity) == expected


def test_distinct_page_buffers_cannot_be_merged_even_when_values_match():
    left = job()
    assert execution._same_inputs(left, replace(left, context=NS(commit_plan=NS(capacity=8))))
    right = replace(left, indices=left.indices.clone())
    assert not execution._same_inputs(left, right)
    runner = execution.GDNCommitExecutor(config(capture_commit=True), 8)
    assert runner._merge_jobs((left, right)) == (left, right)


def test_draft_with_recurrent_state_is_rejected():
    runner = execution.GDNCommitExecutor(config(async_commit=True), 8)
    with pytest.raises(ValueError, match="without recurrent"):
        runner.configure_draft(NS(modules=lambda: [NS(use_gdn_recoverssm=True)]))
    runner.configure_draft(NS(modules=lambda: [NS()]))
    assert runner.overlap_safe


def test_draft_attention_attribute_names_do_not_imply_recurrent_state():
    runner = execution.GDNCommitExecutor(config(async_commit=True), 8)
    runner.configure_draft(NS(modules=lambda: [NS(num_k_heads=8, kv_cache=[])]))
    assert runner.overlap_safe
    with pytest.raises(ValueError, match="without recurrent"):
        runner.configure_draft(NS(modules=lambda: [Mock(spec=execution.MambaBase)]))


def fake_npu(monkeypatch, log):
    class Stream:
        def __init__(self, name="side"):
            self.name = name

        def wait_event(self, event):
            log.append((self.name, "wait", event.name))

        def synchronize(self):
            log.append((self.name, "synchronize"))

    class Event:
        count = 0

        def __init__(self):
            self.name = ("ready", "done")[Event.count % 2]
            Event.count += 1

        def record(self, stream):
            log.append((stream.name, "record", self.name))

    @contextmanager
    def stream(value):
        log.append((value.name, "enter"))
        yield

    main = Stream("main")
    monkeypatch.setattr(torch.npu, "Stream", Stream)
    monkeypatch.setattr(torch.npu, "Event", Event)
    monkeypatch.setattr(torch.npu, "current_stream", lambda: main)
    monkeypatch.setattr(torch.npu, "stream", stream)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, value: log.append((value.name, "retain")))


def test_async_submission_does_not_wait_until_draft_or_target_fence(monkeypatch):
    log = []
    fake_npu(monkeypatch, log)
    runner = execution.GDNCommitExecutor(config(async_commit=True, merge_groups=False), 8)
    runner.configure_draft(NS(modules=lambda: [NS()]))
    runner._run = Mock(side_effect=lambda *args: log.append(("side", "commit")))
    runner.submit((job(),), torch.tensor([2, 3, 1], dtype=torch.int32))
    assert runner.pending
    assert ("main", "wait", "done") not in log
    assert log.index(("main", "record", "ready")) < log.index(("side", "wait", "ready"))
    assert log.index(("side", "commit")) < log.index(("side", "record", "done"))
    runner.wait()
    assert log[-1] == ("main", "wait", "done")
    assert not runner.pending
    before = len(log)
    runner.wait()
    assert len(log) == before
    runner.close()
    assert log[-1] == ("side", "synchronize")


def test_align_keeps_existing_boundary_copy_order(monkeypatch):
    log = []
    fake_npu(monkeypatch, log)
    runner = execution.GDNCommitExecutor(config(async_commit=True, merge_groups=False), 8)
    runner.configure_draft(NS(modules=lambda: [NS()]))
    runner._run = Mock()
    runner.submit((job(align=True),), torch.tensor([2, 3, 1], dtype=torch.int32))
    assert not runner.pending
    assert log[-1] == ("main", "wait", "done")


def test_capture_warmup_cannot_apply_real_transition(monkeypatch):
    source = job()
    calls = []
    source.context.commit.side_effect = lambda accepted, indices, *args: calls.append(indices.clone())
    graph = execution.GDNCommitGraph((source,), torch.tensor([2, 1, 3], dtype=torch.int32), 4)

    @contextmanager
    def capture(*args, **kwargs):
        yield

    monkeypatch.setattr(torch.npu, "NPUGraph", Mock)
    monkeypatch.setattr(torch.npu, "graph", capture)
    graph.capture(object())
    assert len(calls) == 2
    assert all(not indices.count_nonzero() for indices in calls)


def test_state_submits_all_groups_without_losing_their_page_indices():
    executor = NS(submit=Mock())
    state = execution.AscendGDNRecoverSSMState(executor)
    metas = []
    for first in (1, 11, 21):
        meta = AscendGDNRecoverSSMMetadata.__new__(AscendGDNRecoverSSMMetadata)
        meta.num_spec_decodes = 2
        meta.recoverssm_context = object()
        meta.recoverssm_commit = NS(
            state_indices=torch.tensor([[first], [first + 1]], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
            request_indices=None,
            block_table=None,
            num_computed_tokens=None,
            block_size=None,
        )
        metas.append(meta)
    state._step = tuple(metas)
    sampled = torch.tensor([2, 3], dtype=torch.int32)
    state.commit_step(sampled, torch.arange(2), state_indices=None, num_accepted_tokens=torch.ones(2))
    jobs, actual = executor.submit.call_args.args
    assert actual is sampled
    assert len(jobs) == 3
    assert [j.indices.tolist() for j in jobs] == [[1, 2], [11, 12], [21, 22]]
    assert state._step is None


def cache_context():
    conv = torch.zeros((4, 6, 6), dtype=torch.bfloat16)
    checkpoint = torch.zeros((4, 1, 2, 2), dtype=torch.bfloat16)
    replay = torch.zeros((4, 1, 4, 5), dtype=torch.float32)
    cache = binding.GDNRecoverSSMLayerCache(
        conv,
        conv.data_ptr(),
        conv.stride(0),
        conv.stride(1),
        conv.stride(2),
        checkpoint,
        checkpoint.data_ptr(),
        checkpoint.stride(0),
        replay,
        replay.data_ptr(),
        replay.stride(0),
        1,
    )
    caches = (cache,)
    return binding.GDNRecoverSSMCommitContext(
        caches,
        binding.GDNRecoverSSMCommitPlan.allocate(8, torch.device("cpu")),
        4,
        3,
        conv.stride(1),
        binding._create_conv_cache_address_stride_table(caches),
        binding._create_state_commit_cache_address_stride_table(caches),
    )


def test_compatible_groups_merge_all_layers_and_reuse_pointer_views(monkeypatch):
    first = job(context=cache_context())
    second = replace(first, context=cache_context())
    third = replace(first, context=cache_context())
    runner = execution.GDNCommitExecutor(config(capture_commit=True), 8)
    merged = runner._merge_jobs((first, second, third))
    assert len(merged) == 1
    ctx = merged[0].context
    assert len(ctx.layer_caches) == 3
    assert ctx._state_commit_cache_address_stride_table.shape == (2, 2, 3)
    assert runner._merge_jobs((first, second, third))[0].context is ctx
    compact = Mock()
    monkeypatch.setattr(binding, "prepare_commit_plan", Mock())
    monkeypatch.setattr(binding, "compact_conv_state", compact)
    monkeypatch.setattr(binding, "gdn_recoverssm_commit", Mock())
    for _ in range(2):
        ctx.commit(torch.ones(3, dtype=torch.int32), first.indices, first.offsets)
    for call in compact.call_args_list:
        assert all(a is b for a, b in zip(call.args[1:5], ctx._conv_table_views))


def test_draft_exit_fences_before_connector_finalize(monkeypatch):
    log = []
    executor = NS(wait=lambda: log.append("commit_join"))
    runner = runner_module.NPUModelRunner.__new__(runner_module.NPUModelRunner)
    runner.model_state = NS(recoverssm=execution.AscendGDNRecoverSSMState(executor))

    @contextmanager
    def parent_context(self):
        yield
        log.append("draft_context_exit")

    monkeypatch.setattr(runner_module.GPUModelRunner, "draft_context", parent_context)
    with runner.draft_context():
        log.append("draft")
    log.append("kv_finalize")
    assert log == ["draft", "commit_join", "draft_context_exit", "kv_finalize"]


def test_target_entry_fences_before_preemption_or_input_reuse():
    runner = runner_module.NPUModelRunner.__new__(runner_module.NPUModelRunner)
    runner._wait_gdn_recovery = Mock(side_effect=RuntimeError("entry fence"))
    with pytest.raises(RuntimeError, match="entry fence"):
        inspect.unwrap(runner_module.NPUModelRunner.execute_model)(runner, object())
    runner._wait_gdn_recovery.assert_called_once()


def test_async_launch_exception_is_not_hidden_and_done_is_recorded(monkeypatch):
    log = []
    fake_npu(monkeypatch, log)
    runner = execution.GDNCommitExecutor(config(async_commit=True, merge_groups=False), 8)
    runner.configure_draft(NS(modules=lambda: [NS()]))
    runner._run = Mock(side_effect=RuntimeError("commit failure"))
    with pytest.raises(RuntimeError, match="commit failure"):
        runner.submit((job(),), torch.ones(3, dtype=torch.int32))
    assert runner.pending
    assert ("side", "record", "done") in log
    runner.wait()
    assert log[-1] == ("main", "wait", "done")


def test_graph_does_not_hide_short_sampled_counts(monkeypatch):
    source = job()
    graph = execution.GDNCommitGraph((source,), torch.ones(3, dtype=torch.int32), 4)
    stage = Mock()
    monkeypatch.setattr(execution, "stage_gdn_commit", stage)
    with pytest.raises(ValueError, match="do not cover"):
        graph.update((source,), torch.ones(2, dtype=torch.int32))
    stage.assert_not_called()


def test_graph_stages_shared_accepted_only_once(monkeypatch):
    jobs = tuple(job() for _ in range(3))
    graph = execution.GDNCommitGraph(jobs, torch.ones(3, dtype=torch.int32), 4)
    stage = Mock()
    monkeypatch.setattr(execution, "stage_gdn_commit", stage)
    graph.update(jobs, torch.ones(3, dtype=torch.int32))
    assert [call.kwargs["copy_accepted"] for call in stage.call_args_list] == [True, False, False]


@pytest.mark.parametrize("capacity", [1, 8, 160, 192])
def test_startup_covers_every_dispatch_bucket(capacity):
    buckets = execution.commit_buckets(capacity)
    assert buckets[0] == capacity
    assert set(buckets) == {execution.commit_bucket(n, capacity) for n in range(1, capacity + 1)}


def test_graph_miss_never_captures_during_a_request(monkeypatch):
    runner = execution.GDNCommitExecutor(config(capture_commit=True, merge_groups=False), 8)
    monkeypatch.setattr(execution, "GDNCommitGraph", Mock(side_effect=AssertionError("live capture")))
    source = job()
    runner._run((source,), torch.ones(3, dtype=torch.int32))
    source.context.commit.assert_called_once()
    assert not runner.graphs


@pytest.mark.parametrize("align", [False, True])
def test_startup_capture_uses_null_metadata_then_runtime_only_replays(monkeypatch, align):
    fake_npu(monkeypatch, [])
    captures = []

    def stage(source, target, accepted, out_accepted, **kwargs):
        assert not source.indices.count_nonzero() and not source.offsets.count_nonzero()
        assert source.indices.data_ptr() != target.indices.data_ptr()
        assert source.offsets.data_ptr() != target.offsets.data_ptr()
        if source.table is not None:
            assert source.table.data_ptr() != target.table.data_ptr()

    monkeypatch.setattr(execution, "stage_gdn_commit", stage)

    def capture(graph, stream):
        assert all(not job.indices.count_nonzero() and not job.offsets.count_nonzero() for job in graph.jobs)
        assert not graph.accepted.count_nonzero()
        graph.graph = NS(replay=Mock())
        captures.append(graph)

    monkeypatch.setattr(execution.GDNCommitGraph, "capture", capture)
    runner = execution.GDNCommitExecutor(config(capture_commit=True, merge_groups=False), 8)
    source = job(align=align)
    runner.prepare((source,))
    assert len(captures) == 2 * len(execution.commit_buckets(8))
    monkeypatch.setattr(execution, "stage_gdn_commit", Mock())
    runner._run((source,), torch.ones(3, dtype=torch.int32))
    assert sum(graph.graph.replay.call_count for graph in captures) == 1
    source.context.commit.assert_not_called()
    runner.close()
    assert not runner.graphs


def test_worker_deducts_measured_reserve_and_rejects_exhausted_budget(monkeypatch):
    cfg = config(capture_commit=True)
    monkeypatch.setattr(worker_module, "get_ascend_config", lambda: NS(gdn_recoverssm_execution_config=cfg))
    worker = NPUWorker.__new__(NPUWorker)
    worker.vllm_config = None
    worker.model_runner = NS(profile_gdn_recovery_memory=Mock(return_value=30))
    assert worker._reserve_gdn_recovery_memory(100) == 70
    assert worker.gdn_recovery_graph_memory_bytes == 30
    with pytest.raises(ValueError, match="leaves no KV cache"):
        worker._reserve_gdn_recovery_memory(30)
    worker.model_runner = NS()
    assert worker._reserve_gdn_recovery_memory(100) == 100
    cfg = config()
    worker.model_runner = NS(profile_gdn_recovery_memory=Mock(side_effect=AssertionError("disabled probe")))
    assert worker._reserve_gdn_recovery_memory(100) == 100


def memory_api(monkeypatch, free, peak=30):
    monkeypatch.setattr(torch.npu, "synchronize", Mock())
    monkeypatch.setattr(torch.npu, "empty_cache", Mock())
    monkeypatch.setattr(torch.npu, "reset_peak_memory_stats", Mock())
    monkeypatch.setattr(torch.npu, "mem_get_info", Mock(side_effect=[(n, 1000) for n in free]))
    monkeypatch.setattr(torch.npu, "memory_allocated", Mock(return_value=100))
    monkeypatch.setattr(torch.npu, "max_memory_allocated", Mock(return_value=100 + peak))


def test_real_capture_requires_pre_kv_budget_and_checks_peak(monkeypatch):
    runner = runner_module.NPUModelRunner.__new__(runner_module.NPUModelRunner)
    runner.ascend_config = NS(gdn_recoverssm_execution_config=config(capture_commit=True))
    with pytest.raises(RuntimeError, match="profiled before"):
        runner.capture_gdn_recovery_graphs()
    executor = NS(prepare=Mock())
    runner._gdn_recovery_executor = lambda: executor
    monkeypatch.setattr(runner_module, "startup_gdn_commit_jobs", lambda runner: (job(),))
    runner.gdn_recovery_graph_memory_bytes = 40
    memory_api(monkeypatch, [100, 80])
    assert runner.capture_gdn_recovery_graphs() == 30
    memory_api(monkeypatch, [100, 80], peak=41)
    with pytest.raises(RuntimeError, match="exceeding the profiled reserve"):
        runner.capture_gdn_recovery_graphs()


@pytest.mark.parametrize("fails", [False, True])
def test_memory_probe_cleans_temporary_context(monkeypatch, fails):
    from vllm.v1.core import kv_cache_config_builder, kv_cache_planning
    from vllm.v1.worker.gpu import cudagraph_utils

    runner = runner_module.NPUModelRunner.__new__(runner_module.NPUModelRunner)
    runner.ascend_config = NS(gdn_recoverssm_execution_config=config(capture_commit=True))
    runner.vllm_config = object()
    runner.cache_config = NS(num_gpu_blocks=None)
    runner.get_kv_cache_spec = Mock(return_value={})
    runner.initialize_kv_cache = Mock()
    runner.capture_gdn_recovery_graphs = Mock(return_value=50)
    if fails:
        runner.capture_gdn_recovery_graphs.side_effect = RuntimeError("capture failure")
    executor = NS(close=Mock())
    runner._gdn_recovery_executor = lambda: executor
    planner = NS(get_kv_cache_config_from_groups=Mock(return_value=NS(num_blocks=1)))
    monkeypatch.setattr(kv_cache_config_builder, "get_kv_cache_config_builder", lambda _: planner)
    monkeypatch.setattr(kv_cache_planning, "get_kv_cache_groups_from_builder", lambda *args: [])
    cleanup = Mock()
    monkeypatch.setattr(cudagraph_utils, "_teardown_profiling_state", cleanup)
    memory_api(monkeypatch, [100, 65])
    if fails:
        with pytest.raises(RuntimeError, match="capture failure"):
            runner.profile_gdn_recovery_memory()
        assert not hasattr(runner, "gdn_recovery_graph_memory_bytes")
    else:
        assert runner.profile_gdn_recovery_memory() == 50
        assert runner.profile_gdn_recovery_memory() == 50
    executor.close.assert_called_once()
    cleanup.assert_called_once_with(runner)
    assert runner.initialize_kv_cache.call_args.kwargs == {"is_profiling": True}
    assert planner.get_kv_cache_config_from_groups.call_args.kwargs == {"num_blocks": 1}


def test_recovery_capture_follows_model_graphs_in_startup_only(monkeypatch):
    events = []
    worker = NS(
        vllm_config=NS(
            compilation_config=NS(
                compile_sizes=[],
                cudagraph_mode=runner_module.CUDAGraphMode.NONE,
                get_compile_ranges=lambda: [],
                compilation_time=0.0,
            )
        ),
        model_config=NS(enforce_eager=False, seed=0),
        cache_config=NS(kv_cache_memory_bytes=1),
        model_runner=NS(
            capture_model=lambda: events.append("target_and_draft_graphs") or 10,
            capture_gdn_recovery_graphs=lambda: events.append("recovery_graphs") or 5,
        ),
    )
    monkeypatch.setattr(
        worker_module,
        "get_ascend_config",
        lambda: NS(
            enable_cpu_binding=False,
            gdn_recoverssm_execution_config=config(capture_commit=True),
        ),
    )
    monkeypatch.setattr(worker_module, "get_current_hardware_profile", lambda: NS(supports=lambda _: False))
    monkeypatch.setattr(worker_module, "set_random_seed", Mock())
    with patch(
        "vllm_ascend.model_executor.warmup.kernel_warmup.kernel_warmup",
        side_effect=lambda _: events.append("kernel_warmup"),
    ):
        NPUWorker.compile_or_warm_up_model(worker)
    assert events == ["kernel_warmup", "target_and_draft_graphs", "recovery_graphs"]


def test_startup_jobs_use_bound_groups_and_separate_null_page_tables():
    builders = []
    for _ in range(3):
        builder = execution.AscendGDNRecoverSSMMetadataBuilder.__new__(execution.AscendGDNRecoverSSMMetadataBuilder)
        builder.spec_state_indices_tensor = torch.tensor([[17]], dtype=torch.int32)
        builder.kv_cache_spec = NS(block_size=8)
        builder._get_context = Mock(return_value=object())
        builders.append(builder)
    runner = NS(
        max_num_reqs=8,
        device=torch.device("cpu"),
        cache_config=NS(mamba_cache_mode="align"),
        attn_groups=[[NS(get_metadata_builder=lambda index, builder=builder: builder)] for builder in builders],
        block_tables=NS(block_tables=[NS(gpu=torch.ones((9, 7), dtype=torch.int32)) for _ in builders]),
    )
    jobs = execution.startup_gdn_commit_jobs(runner)
    assert len(jobs) == 3
    assert len({job.indices.data_ptr() for job in jobs}) == 3
    for builder, job in zip(builders, jobs):
        assert job.context is builder._get_context.return_value
        assert job.table.shape == (8, 7)
        assert not job.indices.count_nonzero() and not job.table.count_nonzero()
        assert builder.spec_state_indices_tensor[0, 0] == 17
