# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU composition checks; these do not measure or execute NPU kernels."""

import ast
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
INDEXER = "vllm_ascend/attention/indexer.py"
DEVICE = "vllm_ascend/device/device_op.py"


def test_a3_inline_metadata_receives_global_bounds_and_geometry(runtime):
    runtime.scope["uses_native_int8_sparse_adaptor"] = lambda _: True
    runtime.scope["get_dcp_group"] = lambda: SimpleNamespace(world_size=16, rank_in_group=7)
    producer = Mock(return_value=torch.empty(1024, dtype=torch.int32))
    runtime.scope["DeviceOperator"] = SimpleNamespace(build_indexer_local_metadata=producer)
    metadata = make_metadata()
    metadata.seq_lens = torch.tensor([10000, 20000], dtype=torch.int32)
    layer = make_layer(runtime, "first")
    layer._dcp_interleave = 768
    layer._get_local_qli_inputs(metadata)
    kw = producer.call_args.kwargs
    assert kw["causal_seqlens_kv_global"] is metadata.seq_lens
    assert (kw["cp_world"], kw["cp_rank"], kw["cp_kv_cache_interleave_size"]) == (16, 7, 768)


def test_a3_deferred_metadata_receives_global_bounds_and_geometry(deferred_runtime):
    env = deferred_runtime
    env.runtime.scope["uses_native_int8_sparse_adaptor"] = lambda _: True
    env.runtime.scope["get_dcp_group"] = lambda: SimpleNamespace(world_size=16, rank_in_group=7)
    producer = Mock(return_value=torch.empty(1024, dtype=torch.int32))
    env.runtime.scope["DeviceOperator"] = SimpleNamespace(build_indexer_local_metadata=producer)
    env.builder.vllm_config.parallel_config = SimpleNamespace(cp_kv_cache_interleave_size=768)
    metadata = make_metadata()
    metadata.seq_lens = torch.tensor([10000, 20000], dtype=torch.int32)
    env.builder._defer_local_qli_metadata(metadata)
    tasks = env.builder.take_device_metadata_tasks()
    assert producer.call_count == 0
    tasks[0].run()
    kw = producer.call_args.kwargs
    assert kw["causal_seqlens_kv_global"] is metadata.seq_lens
    assert (kw["cp_world"], kw["cp_rank"], kw["cp_kv_cache_interleave_size"]) == (16, 7, 768)


def load_methods(path, owner, names, scope):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner)
    cls.bases = []
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in cls.body} == set(names)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[])), path, "exec"), scope)
    return scope[owner]


@pytest.fixture
def runtime(monkeypatch):
    state = SimpleNamespace(context=SimpleNamespace())
    scope = {"torch": torch, "dataclass": dataclass, "get_forward_context": lambda: state.context}
    scope.update(
        uses_native_int8_sparse_adaptor=lambda _: False,
        get_current_hardware_profile=lambda: None,
    )
    adaptor = load_methods(
        DEVICE,
        "A5DeviceAdaptor",
        ["build_indexer_local_metadata", "indexer_select_local_candidates", "indexer_select_dcp_candidates"],
        scope,
    )
    scope["DeviceOperator"] = adaptor
    backend = load_methods(INDEXER, "AscendSFAIndexerBackend", ["_get_local_qli_inputs"], scope)

    def metadata_kernel(*args, **kwargs):
        # Encode the input lengths so reuse of an old result is observable.
        return torch.cat((kwargs["cu_seqlens_q"].clone(), kwargs["seqused_k"].clone()))

    kernel = Mock(side_effect=metadata_kernel)
    candidates = Mock(return_value=(torch.empty(0), torch.empty(0)))
    monkeypatch.setattr(torch.ops._C_ascend, "npu_quant_lightning_indexer_v2_metadata", kernel, raising=False)
    monkeypatch.setattr(torch.ops._C_ascend, "npu_quant_lightning_indexer_local_candidates", candidates, raising=False)
    return SimpleNamespace(
        scope=scope, state=state, backend=backend, adaptor=adaptor, kernel=kernel, candidates=candidates
    )


def make_layer(runtime, name, heads=32, head_dim=128, topk=2048):
    layer = runtime.backend()
    layer.n_head, layer.head_dim, layer.topk_tokens = heads, head_dim, topk
    layer.k_cache = SimpleNamespace(prefix=name)
    return layer


def make_metadata(query_starts=(0, 2, 4), local_lengths=(31, 65), dtype=torch.int32):
    return SimpleNamespace(
        query_start_loc=torch.tensor(query_starts, dtype=dtype),
        actual_seq_lengths_key=torch.tensor(local_lengths, dtype=dtype),
        seq_lens=torch.tensor(local_lengths, dtype=dtype),
        qli_metadata=None,
        qli_metadata_group_id=None,
        qli_dcp_args=None,
    )


def test_24_layers_share_metadata_and_no_pad(runtime, monkeypatch):
    metadata = make_metadata()
    per_layer_metadata = {f"layer{i}": metadata for i in range(24)}
    pad = Mock(side_effect=AssertionError("query_start_loc already contains zero"))
    monkeypatch.setattr(torch.nn.functional, "pad", pad)
    outputs = [make_layer(runtime, name)._get_local_qli_inputs(per_layer_metadata[name]) for name in per_layer_metadata]
    runtime.kernel.assert_called_once()
    for query, seq, plan in outputs:
        assert query is metadata.query_start_loc and seq is metadata.actual_seq_lengths_key
        assert plan is outputs[0][2]
    assert runtime.kernel.call_args.args == (32, 1, 128, 2048, 1)
    assert runtime.kernel.call_args.kwargs | {} == {
        "cu_seqlens_q": metadata.query_start_loc,
        "seqused_k": metadata.actual_seq_lengths_key,
        "batch_size": 2,
        "max_seqlen_q": -1,
        "max_seqlen_k": -1,
        "layout_q": "TND",
        "layout_k": "PA_BBND",
        "mask_mode": 0,
        "cmp_ratio": 1,
    }
    pad.assert_not_called()


def test_warmup_then_capture_same_context_keeps_metadata_generation(runtime):
    metadata = make_metadata()
    first, second = make_layer(runtime, "first"), make_layer(runtime, "second")
    old = first._get_local_qli_inputs(metadata)[2]
    assert second._get_local_qli_inputs(metadata)[2] is old
    metadata.actual_seq_lengths_key.add_(8)
    new = first._get_local_qli_inputs(metadata)[2]
    assert second._get_local_qli_inputs(metadata)[2] is new
    assert runtime.kernel.call_count == 2 and new is not old
    torch.testing.assert_close(new[-2:], torch.tensor([39, 73], dtype=torch.int32))


def test_new_forward_skipping_previous_owner_does_not_reuse_stale_metadata(runtime):
    metadata = make_metadata()
    first, second = make_layer(runtime, "first"), make_layer(runtime, "second")
    old = first._get_local_qli_inputs(metadata)[2]
    runtime.state.context = SimpleNamespace()
    metadata.actual_seq_lengths_key.zero_()
    new = second._get_local_qli_inputs(metadata)[2]
    assert new is not old and runtime.kernel.call_count == 2
    torch.testing.assert_close(new[-2:], torch.zeros(2, dtype=torch.int32))


@pytest.mark.parametrize("change", ["heads", "head_dim", "topk", "query", "key", "draft"])
def test_distinct_contracts_do_not_share(runtime, change):
    metadata = make_metadata()
    first, second = make_layer(runtime, "first"), make_layer(runtime, "second")
    first._get_local_qli_inputs(metadata)
    if change == "heads":
        second.n_head = 64
    elif change == "head_dim":
        second.head_dim = 64
    elif change == "topk":
        second.topk_tokens = 1024
    elif change == "query":
        metadata.query_start_loc = torch.tensor([0, 1, 4], dtype=torch.int32)
    elif change == "key":
        metadata.actual_seq_lengths_key = torch.tensor([33, 0], dtype=torch.int32)
    else:
        # Same shapes and input storage, but a separate MTP metadata object.
        metadata = SimpleNamespace(**vars(metadata))
    second._get_local_qli_inputs(metadata)
    assert runtime.kernel.call_count == 2


@pytest.mark.parametrize("starts,lengths", [([0, 1, 2, 2], [31, 65, 0]), ([0, 2, 5], [128, 0]), ([0], [])])
def test_speculative_padding_and_empty_dcp_lengths(runtime, starts, lengths):
    metadata = make_metadata(starts, lengths, dtype=torch.int64)
    inputs = make_layer(runtime, "first")._get_local_qli_inputs(metadata)
    assert all(tensor.dtype == torch.int32 for tensor in inputs)
    torch.testing.assert_close(inputs[0], metadata.query_start_loc.to(torch.int32))
    torch.testing.assert_close(inputs[1], metadata.actual_seq_lengths_key.to(torch.int32))


def test_candidate_consumer_uses_shared_inputs_without_regeneration(runtime):
    metadata = make_metadata()
    query_starts, local_lengths, plan = make_layer(runtime, "first")._get_local_qli_inputs(metadata)
    visible = torch.tensor([[7], [8], [0], [2]], dtype=torch.int32)
    cache = torch.zeros(2, 128, 1, 128)
    blocks = torch.tensor([[0], [1]], dtype=torch.int32)
    runtime.adaptor.indexer_select_local_candidates(
        torch.zeros(4, 32, 128),
        torch.zeros(4, 32),
        cache,
        torch.zeros(2, 128, 1, 1),
        blocks,
        query_starts,
        local_lengths,
        torch.ones(4, 32),
        visible,
        qli_metadata=plan,
    )
    runtime.kernel.assert_called_once()
    args = runtime.candidates.call_args.args
    assert args[5] is query_starts and args[6] is local_lengths and args[7] is blocks and args[9] is plan
    torch.testing.assert_close(args[8], visible)


@pytest.fixture
def deferred_runtime(runtime, monkeypatch):
    """Compose the production builder, executor and consumer with CPU streams."""
    calls = []

    class Stream:
        def __init__(self, name="metadata"):
            self.name = name

        def wait_event(self, event):
            calls.append((self.name, "wait", event))

    class Event:
        def record(self, stream):
            calls.append((stream.name, "record", self))

    current = SimpleNamespace(stream=Stream("main"))

    @contextmanager
    def stream_scope(stream):
        previous, current.stream = current.stream, stream
        try:
            yield
        finally:
            current.stream = previous

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(Stream=Stream, Event=Event, current_stream=lambda: current.stream, stream=stream_scope),
        raising=False,
    )
    path = ROOT / "vllm_ascend/worker/device_metadata.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [
        node for node in tree.body if not (isinstance(node, ast.ImportFrom) and node.module == "vllm.forward_context")
    ]
    module = ModuleType("qli_device_metadata_cpu")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    module.__dict__.update(
        BatchDescriptor=object,
        get_forward_context=lambda: runtime.state.context,
        is_forward_context_available=lambda: True,
    )
    exec(compile(tree, str(path), "exec"), module.__dict__)
    runtime.scope.update(
        contextmanager=contextmanager,
        DeviceMetadataTask=module.DeviceMetadataTask,
        DeviceMetadataStage=module.DeviceMetadataStage,
        wait_for_device_metadata=module.wait_for_device_metadata,
    )
    builder_cls = load_methods(
        INDEXER,
        "AscendSFAIndexerMetadataBuilder",
        ["defer_device_metadata", "take_device_metadata_tasks", "_defer_local_qli_metadata"],
        runtime.scope,
    )
    builder = builder_cls()
    builder.use_sharded_dcp = True
    builder._device_metadata_enabled = False
    builder._device_metadata_tasks = ()
    builder.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(index_n_heads=32, index_head_dim=128, index_topk=2048)
        )
    )
    runtime.kernel.side_effect = lambda *args, **kwargs: (
        calls.append((current.stream.name, "kernel")),
        torch.cat((kwargs["cu_seqlens_q"].clone(), kwargs["seqused_k"].clone())),
    )[1]
    target = module.TargetDeviceMetadata()
    groups = [[SimpleNamespace(get_metadata_builder=lambda _: builder)]]

    def prepare(metadata, *, full_graph=False):
        def build(**kwargs):
            assert builder._device_metadata_enabled
            builder._defer_local_qli_metadata(metadata)
            return {f"layer{i}": metadata for i in range(24)}

        result = target.run_build(build, attn_groups=groups, full_graph_mode=full_graph)
        runtime.state.context = SimpleNamespace(attn_metadata=result)
        return result

    return SimpleNamespace(runtime=runtime, builder=builder, target=target, prepare=prepare, calls=calls, module=module)


@pytest.mark.parametrize("full_graph", [False, True])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_v2_deferred_producer_and_single_consumer_wait(deferred_runtime, full_graph, dtype):
    env = deferred_runtime
    metadata = make_metadata(dtype=dtype)
    env.prepare(metadata, full_graph=full_graph)
    if full_graph:
        assert metadata.qli_metadata is None
        assert env.runtime.kernel.call_count == 0
        env.target.begin_forward()
    assert not env.builder._device_metadata_enabled
    assert env.builder.take_device_metadata_tasks() == ()
    assert ("metadata", "kernel") in env.calls
    assert not any(call[:2] == ("main", "wait") for call in env.calls)
    outputs = [make_layer(env.runtime, f"layer{i}")._get_local_qli_inputs(metadata) for i in range(24)]
    assert all(output[2] is metadata.qli_metadata for output in outputs)
    assert all(t.dtype == torch.int32 for t in outputs[0])
    env.runtime.kernel.assert_called_once()
    # Consumption and final join use the same frontier; no per-layer wait or
    # additional output copy is submitted.
    env.target.finish()
    assert len([call for call in env.calls if call[:2] == ("main", "wait")]) == 1
    torch.testing.assert_close(outputs[0][2], torch.tensor([0, 2, 4, 31, 65], dtype=torch.int32))
    assert not hasattr(env.runtime.state.context, "sfa_qli_metadata_cache")


def test_v2_warmup_capture_and_replay_do_not_resubmit_on_host(deferred_runtime):
    env = deferred_runtime
    query, keys = torch.tensor([0, 2, 4], dtype=torch.int32), torch.tensor([31, 65], dtype=torch.int32)
    for step in range(3):
        metadata = make_metadata()
        metadata.query_start_loc, metadata.actual_seq_lengths_key = query, keys
        keys.add_(8)
        env.prepare(metadata, full_graph=True)
        env.target.begin_forward()
        # The previous first consumer may be absent from a later forward.
        output = make_layer(env.runtime, f"first{step}")._get_local_qli_inputs(metadata)[2]
        torch.testing.assert_close(output[-2:], keys)
        env.target.finish()
    assert env.runtime.kernel.call_count == 3
    assert len(env.target.executor._stage_ready) == 1
    env.prepare(make_metadata(), full_graph=True)
    env.target.finish_replay()
    assert env.runtime.kernel.call_count == 3
    assert env.target._tasks == ()


def test_v2_unused_producer_is_joined_and_draft_scope_restored(deferred_runtime):
    env = deferred_runtime
    env.prepare(make_metadata(), full_graph=True)
    env.target.begin_forward()
    env.target.finish()  # e.g. a profiling path did not execute QLI
    assert len([call for call in env.calls if call[:2] == ("main", "wait")]) == 1
    with pytest.raises(ValueError, match="build failed"), env.builder.defer_device_metadata():
        assert env.builder._device_metadata_enabled
        raise ValueError("build failed")
    assert not env.builder._device_metadata_enabled
    env.builder.use_sharded_dcp = False
    with env.builder.defer_device_metadata():
        assert not env.builder._device_metadata_enabled


@pytest.mark.parametrize(
    "a5,architecture,sharded,kv_dtype,enabled",
    [
        (True, "KimiK31ForConditionalGeneration", True, "fp8", True),
        (True, "KimiK31ForConditionalGeneration", False, "fp8", False),
        (True, "KimiK31ForConditionalGeneration", True, "fp4", False),
        (True, "KimiK31ForConditionalGeneration", True, "auto", False),
        (False, "KimiK31ForConditionalGeneration", True, "fp8", False),
        (True, "KimiK3ForConditionalGeneration", True, "fp8", False),
        (True, "KimiK3ForCausalLM", True, "fp8", False),
        (True, "KimiLinearForCausalLM", True, "fp8", False),
        (True, "DeepseekV41ForCausalLM", False, "fp8", True),
        (True, "DeepseekV41ForCausalLM", False, "auto", True),
        (False, "DeepseekV41ForCausalLM", False, "fp8", False),
        (True, "DeepseekV3ForCausalLM", False, "fp8", False),
        (True, "DeepseekV3ForCausalLM", True, "fp8", False),
    ],
)
def test_v2_runner_only_enables_supported_metadata_producers(a5, architecture, sharded, kv_dtype, enabled):
    path = ROOT / "vllm_ascend/worker/v2/model_runner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(ast.unparse(target) == "self.model_state.device_metadata" for target in node.targets)
    )
    marker = object()
    runner = SimpleNamespace(
        model_state=SimpleNamespace(),
        model_config=SimpleNamespace(architecture=architecture),
        vllm_config=SimpleNamespace(attention_config=SimpleNamespace(indexer_kv_dtype=kv_dtype)),
    )
    sharded_check = Mock(return_value=sharded)
    scope = {
        "self": runner,
        "uses_a5_packed_cache": lambda: a5,
        "enable_sfa_dcp_sharded_indexer": sharded_check,
        "TargetDeviceMetadata": lambda: marker,
    }
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(path), "exec"), scope)
    assert runner.model_state.device_metadata is (marker if enabled else None)
    if a5 and architecture == "KimiK31ForConditionalGeneration":
        sharded_check.assert_called_once_with(runner.vllm_config)
    else:
        sharded_check.assert_not_called()


@pytest.mark.parametrize("rank", [0, 7])
def test_a5_native_dcp_metadata_and_consumer_share_causal_contract(runtime, monkeypatch, rank):
    metadata = make_metadata((0, 4, 8), (768, 1536))
    global_lens = torch.tensor([6190, 12320], dtype=torch.int32)
    metadata.qli_dcp_args = dict(global_seq_lens=global_lens, cp_world=8, cp_rank=rank, cp_interleave=768)
    layers = [make_layer(runtime, f"layer{i}") for i in range(24)]
    shared = [layer._get_local_qli_inputs(metadata) for layer in layers]
    runtime.kernel.assert_called_once()
    producer = runtime.kernel.call_args.kwargs
    assert producer["causal_seqlens_kv_global"] is global_lens
    assert producer["enable_dcp"] is True and producer["mask_mode"] == 3

    key = torch.zeros(2, 16896, dtype=torch.uint8).view(torch.float8_e4m3fn)
    key = key.as_strided((2, 128, 1, 128), (16896, 128, 128, 1))
    scale = torch.ones(2, 4224).as_strided((2, 128, 1, 1), (4224, 1, 1, 1))
    output = (torch.zeros(12, 1, 2048, dtype=torch.int32), torch.zeros(12, 1, 2048, dtype=torch.bfloat16))
    native = Mock(return_value=output)
    monkeypatch.setattr(torch.ops._C_ascend, "npu_quant_lightning_indexer_v2", native, raising=False)
    result = runtime.adaptor.indexer_select_dcp_candidates(
        torch.zeros(12, 32, 128, dtype=torch.float8_e4m3fn),
        torch.ones(12, 32),
        key,
        scale,
        torch.zeros(2, 1, dtype=torch.int32),
        shared[0][0],
        shared[0][1],
        torch.ones(12, 32),
        qli_metadata=shared[0][2],
        **metadata.qli_dcp_args,
    )
    assert result is output  # No standalone BF16 -> FP32 conversion.
    args, consumer = native.call_args
    assert args[1] is key and args[4].stride(0) == scale.stride(0)
    assert args[4].data_ptr() == scale.data_ptr()
    for name in (
        "enable_dcp",
        "cp_world",
        "cp_rank",
        "cp_kv_cache_interleave_size",
        "causal_seqlens_kv_global",
        "mask_mode",
    ):
        assert consumer[name] is producer[name] or consumer[name] == producer[name]
    assert consumer["return_value"] == 1 and consumer["output_idx_offset"] is None
    runtime.candidates.assert_not_called()


def test_native_dcp_deferred_producer_refreshes_global_lengths(deferred_runtime):
    env = deferred_runtime
    global_lens = torch.tensor([6144, 12288], dtype=torch.int32)
    seen = []
    original = env.runtime.kernel.side_effect

    def observe(*args, **kwargs):
        seen.append(kwargs["causal_seqlens_kv_global"].clone())
        assert kwargs["enable_dcp"] and kwargs["cp_rank"] == 7 and kwargs["mask_mode"] == 3
        return original(*args, **kwargs)

    env.runtime.kernel.side_effect = observe
    for _ in range(2):
        metadata = make_metadata()
        metadata.qli_dcp_args = dict(global_seq_lens=global_lens, cp_world=8, cp_rank=7, cp_interleave=768)
        global_lens.add_(4)
        env.prepare(metadata, full_graph=True)
        env.target.begin_forward()
        for i in range(24):
            make_layer(env.runtime, f"layer{i}")._get_local_qli_inputs(metadata)
        env.target.finish()
    assert env.runtime.kernel.call_count == 2
    torch.testing.assert_close(seen[1] - seen[0], torch.tensor([4, 4], dtype=torch.int32))


@pytest.mark.parametrize("native,a3", [(False, False), (True, False), (False, True)])
def test_metadata_builder_selects_row_lengths_or_native_dcp(runtime, monkeypatch, native, a3):
    module = ModuleType("vllm_ascend.ops.triton.sharded_indexer")
    module.build_local_key_lengths = Mock(side_effect=lambda seq, out, *args: out.copy_(seq))
    module.build_local_lengths = Mock(side_effect=lambda seq, ends, out, visible, *args: out.copy_(seq))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    runtime.scope.update(
        uses_native_int8_sparse_adaptor=lambda _: a3,
        uses_pcp_sharded_layout=lambda common, use_pcp: False,
        get_dcp_group=lambda: SimpleNamespace(world_size=8, rank_in_group=7),
        get_cos_and_sin_mla=lambda positions, **kwargs: (positions, positions),
        AscendSFAIndexerMetadata=SimpleNamespace,
    )
    builder = load_methods(INDEXER, "AscendSFAIndexerMetadataBuilder", ["_build"], runtime.scope)()
    builder.use_dcp = builder.use_pcp = builder._device_metadata_enabled = False
    builder.use_sharded_dcp, builder.use_native_dcp_qli = True, native
    builder._mask_lim_slot_mapping = Mock()
    builder._sharded_seq_buffers, builder._slot_capacity = {}, 16
    builder.device, builder.kernel_block_size, builder.use_rope = torch.device("cpu"), 128, False
    builder.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=2),
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=768),
    )
    common = SimpleNamespace(
        num_reqs=2,
        num_input_tokens=12,
        num_actual_tokens=8,
        slot_mapping=torch.zeros(12, dtype=torch.int32),
        positions=torch.zeros(12, dtype=torch.int32),
        query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
        seq_lens=torch.tensor([6144, 12288], dtype=torch.int32),
        block_table_tensor=torch.zeros(2, 1, dtype=torch.int32),
    )
    metadata = builder._build(common, "target", False, False)
    if native:
        assert metadata.local_visible_lengths is None
        assert metadata.qli_dcp_args["global_seq_lens"].data_ptr() == common.seq_lens.data_ptr()
        assert metadata.qli_dcp_args["cp_rank"] == 7
        assert builder._sharded_seq_buffers["target"][1].numel() == 0
        module.build_local_key_lengths.assert_called_once()
        module.build_local_lengths.assert_not_called()
    elif a3:
        assert metadata.qli_dcp_args is None and metadata.local_visible_lengths is None
        module.build_local_key_lengths.assert_called_once()
        module.build_local_lengths.assert_not_called()
    else:
        assert metadata.qli_dcp_args is None and metadata.local_visible_lengths.shape == (12, 1)
        module.build_local_key_lengths.assert_not_called()
        module.build_local_lengths.assert_called_once()
