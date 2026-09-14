# SPDX-License-Identifier: Apache-2.0
"""Execute DSpark graph orchestration on CPU without importing the NPU stack."""

import ast
import copy
import sys
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import numpy as np
import pytest
import torch


class GraphMode(Enum):
    NONE = 0
    FULL = 1


class ParentProposer:
    def __init__(self, config, device, runner=None):
        self.draft_model_config = config.speculative_config.draft_model_config
        self.num_speculative_tokens = 7
        self.max_batch_size = 2
        self.max_num_tokens = 16
        self.dtype = torch.float32
        self.device = device
        self.use_cuda_graph = True


class MetadataProvider:
    pass


def load_methods(filename, class_name, methods, **extra):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/spec_decode" / filename
    tree = ast.parse(source.read_text(encoding="utf-8"))
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    definition.bases = [ast.Name(id="ParentProposer", ctx=ast.Load())]
    definition.body = [node for node in definition.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    namespace = {
        "torch": torch,
        "copy": copy,
        "F": torch.nn.functional,
        "envs": SimpleNamespace(VLLM_ASCEND_ENABLE_FLASH_MLA=False),
        "CUDAGraphMode": GraphMode,
        "ParentProposer": ParentProposer,
        "DeviceMetadataTaskProvider": MetadataProvider,
        "DynamicSpecScheduler": MagicMock(),
        "get_ascend_config": lambda: SimpleNamespace(dynamic_spec_config=SimpleNamespace(method="", method_params={})),
        "logger": MagicMock(),
        "BreakableACLGraphWrapper": type("BreakableACLGraphWrapper", (), {}),
        "_HIDDEN_STATE_DRAFTER_TYPES": (SimpleNamespace,),
        **extra,
    }
    exec(compile("from __future__ import annotations\n" + ast.unparse(definition), str(source), "exec"), namespace)
    return namespace[class_name]


@pytest.mark.parametrize("dynamic", [False, True])
def test_static_graph_and_dynamic_eager_capacity(dynamic):
    proposer_type = load_methods(
        "dspark_proposer.py",
        "AscendDSparkProposer",
        {"__init__"},
        get_ascend_config=lambda: SimpleNamespace(
            dynamic_spec_config=SimpleNamespace(method="dspark" if dynamic else "", method_params={})
        ),
    )
    draft_config = SimpleNamespace(hf_config=SimpleNamespace(sample_from_anchor=True), get_hidden_size=lambda: 4)
    proposer = proposer_type(
        SimpleNamespace(speculative_config=SimpleNamespace(draft_model_config=draft_config)), "cpu"
    )
    assert proposer.use_cuda_graph is (not dynamic)
    assert proposer.num_query_per_req == 7
    assert proposer.max_query_tokens == 16
    assert proposer.positions.shape == proposer._slot_mapping_buffer.shape == (16,)


@pytest.mark.parametrize(
    "width,real_reqs,graph_reqs,tokens,expected",
    [
        (7, 1, 2, 16, [0, 7, 14, 16]),
        (5, 2, 8, 48, [0, 5, 10, 15, 20, 25, 30, 35, 40, 48]),
        (7, 2, 2, 16, [0, 7, 14, 16]),
    ],
)
def test_query_padding_including_full_request_bucket(width, real_reqs, graph_reqs, tokens, expected):
    proposer_type = load_methods("dspark_proposer.py", "AscendDSparkProposer", {"pad_query_start_loc_for_graph"})
    offsets = SimpleNamespace(np=np.zeros(graph_reqs + 2, dtype=np.int32), copy_to_gpu=MagicMock())
    offsets.np[: real_reqs + 1] = np.arange(real_reqs + 1) * width
    count = proposer_type.pad_query_start_loc_for_graph(
        SimpleNamespace(num_query_per_req=width), offsets, tokens, real_reqs, graph_reqs
    )
    assert count == len(expected) - 1
    assert offsets.np[: len(expected)].tolist() == expected
    offsets.copy_to_gpu.assert_called_once_with()


def test_block_table_reserves_and_clears_extra_metadata_row():
    proposer_type = load_methods("dspark_proposer.py", "AscendDSparkProposer", {"set_per_group_attn_metadata"})
    proposer = SimpleNamespace(
        max_batch_size=2, _per_group_block_tables={}, _per_group_slot_mappings={}, _per_group_block_table_buffers={}
    )
    table = torch.arange(8, dtype=torch.int32).reshape(2, 4)
    proposer_type.set_per_group_attn_metadata(proposer, 0, table, torch.arange(16))
    buffer = proposer._per_group_block_table_buffers[0]
    assert buffer.shape == (3, 4)
    assert torch.equal(buffer[:2], table)
    assert torch.count_nonzero(buffer[2:]) == 0
    buffer[1:].fill_(99)
    proposer_type.set_per_group_attn_metadata(proposer, 0, table[:1], torch.arange(8))
    assert proposer._per_group_block_table_buffers[0] is buffer
    assert torch.count_nonzero(buffer[1:]) == 0


def test_graph_dispatch_uses_target_verification_width():
    proposer_type = load_methods("llm_base_proposer.py", "AscendSpecDecodeBaseProposer", {"_propose"})
    proposer = proposer_type.__new__(proposer_type)
    proposer.method = "dspark"
    proposer.model = SimpleNamespace(combine_hidden_states=lambda value: value)
    proposer.hidden_size = 4
    proposer.use_cuda_graph = True
    metadata = MagicMock()
    metadata.batch_size.return_value = 16
    proposer.set_inputs_first_pass = MagicMock(return_value=(112, torch.arange(112), metadata, None))
    proposer.runner = MagicMock()
    proposer.runner.dcp_manager = None
    proposer.runner.input_batch.lora_id_to_lora_request = {}
    proposer.runner.cudagraph_dispatcher.dispatch.return_value = (GraphMode.FULL, SimpleNamespace(num_tokens=128))
    proposer.runner._sync_metadata_across_dp.side_effect = RuntimeError("dispatch observed")
    with pytest.raises(RuntimeError, match="dispatch observed"):
        proposer._propose(
            7,
            torch.zeros(112),
            torch.zeros(112),
            torch.zeros(112, 4),
            torch.zeros(16),
            torch.arange(112),
            metadata,
            SimpleNamespace(uniform=True),
            MagicMock(),
        )
    proposer.runner.cudagraph_dispatcher.dispatch.assert_called_once_with(
        num_tokens=128, uniform_decode=True, has_lora=False
    )


@pytest.mark.parametrize("mode", [GraphMode.NONE, GraphMode.FULL])
@pytest.mark.parametrize("dcp_size", [1, 2])
@pytest.mark.parametrize("flash_enabled", [False, True])
def test_metadata_waits_before_graph_and_preserves_eager_overlap(mode, dcp_size, flash_enabled):
    proposer_type = load_methods(
        "llm_base_proposer.py",
        "AscendSpecDecodeBaseProposer",
        {"build_draft_attn_metadata"},
        envs=SimpleNamespace(VLLM_ASCEND_ENABLE_FLASH_MLA=flash_enabled),
    )
    tasks = [SimpleNamespace(stage="attention", group_id=gid) for gid in (7, 9)]
    groups = []
    for gid, task in enumerate(tasks):
        provider = MetadataProvider()
        provider.take_device_metadata_tasks = lambda task=task: (task,)
        provider.build_for_drafting = lambda *args, **kwargs: SimpleNamespace()
        groups.append(
            SimpleNamespace(
                kv_cache_group_id=gid, layer_names=[str(gid)], get_metadata_builder=lambda provider=provider: provider
            )
        )
    executor = MagicMock()
    proposer = SimpleNamespace(
        draft_attn_groups=groups,
        use_compress=False,
        method="dspark",
        dcp_size=dcp_size,
        runner=SimpleNamespace(device_metadata_executor=executor, dcp_manager=MagicMock()),
        num_query_per_req=1,
        sliding_window=None,
        vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(prefill_context_parallel_size=1)),
        _per_group_block_table_buffers={},
        _per_group_query_slot_mapping_buffers={0: None, 1: None},
    )
    proposer_type.build_draft_attn_metadata(proposer, SimpleNamespace(num_reqs=1), 1, 1, aclgraph_runtime_mode=mode)
    expected = [call.submit(tasks)] if dcp_size == 1 or flash_enabled else []
    if expected and mode is GraphMode.FULL:
        expected += [call.wait(task.stage, task.group_id) for task in tasks]
    assert executor.mock_calls == expected


@pytest.mark.parametrize("dcp_size", [1, 2])
def test_dummy_graph_waits_outside_context_and_disables_all_cache_writes(monkeypatch, dcp_size):
    events = []
    captured = []
    task = SimpleNamespace(stage="attention", group_id=0)
    provider = MetadataProvider()
    provider.take_device_metadata_tasks = lambda: (task,)
    provider.build_for_graph_capture = lambda metadata, state: captured.append(metadata) or SimpleNamespace()

    @contextmanager
    def forward_context(*args, **kwargs):
        events.append("enter")
        yield
        events.append("exit")

    proposer_type = load_methods(
        "dspark_proposer.py",
        "AscendDSparkProposer",
        {"dummy_run"},
        AscendCommonAttentionMetadata=SimpleNamespace,
        AscendAttentionState=SimpleNamespace(SpecDecoding="spec"),
        set_ascend_forward_context=forward_context,
        get_forward_context=lambda: SimpleNamespace(cudagraph_runtime_mode=GraphMode.FULL),
        _EXTRA_CTX=SimpleNamespace(capturing=True),
        envs=SimpleNamespace(VLLM_ASCEND_ENABLE_FLASH_MLA=True),
    )
    executor = SimpleNamespace(
        submit=lambda tasks: events.append("submit"),
        wait=lambda *args: events.append("wait"),
        release=lambda: events.append("release"),
    )
    proposer = proposer_type.__new__(proposer_type)
    proposer.runner = SimpleNamespace(
        device_metadata_executor=executor,
        dynamic_eplb=False,
        optimistic_seq_lens_cpu=torch.tensor([7, 7]),
        seq_lens=torch.tensor([7, 7]),
        _sync_metadata_across_dp=lambda n, **kwargs: (n, None, None),
        dcp_manager=load_dcp_manager(monkeypatch, dcp_size, 0, 1),
    )
    proposer.num_query_per_req = proposer.num_speculative_tokens = 7
    proposer.dcp_size = dcp_size
    proposer.max_query_tokens = 16
    proposer.use_cuda_graph = True
    proposer.device = "cpu"
    proposer._context_positions_buffer = proposer.positions = torch.arange(16)
    proposer.hidden_states = torch.zeros(16, 4)
    proposer.token_indices_to_sample = torch.zeros(16, dtype=torch.int32)
    proposer.token_arange_np = np.arange(4)
    proposer.vllm_config = SimpleNamespace()
    proposer.draft_attn_groups = [
        SimpleNamespace(kv_cache_group_id=0, layer_names=["draft"], get_metadata_builder=lambda: provider)
    ]
    proposer._per_group_query_slot_mapping_buffers = {0: torch.arange(16)}
    proposer._per_group_block_table_buffers = {0: torch.zeros(3, 4)}
    proposer.model = SimpleNamespace(get_draft_attn_causal=lambda: [False])
    proposer._adjust_tensor = lambda tensor, size: torch.nn.functional.pad(tensor, (0, max(0, size - len(tensor))))[
        :size
    ]
    proposer._get_positions = lambda size: proposer.positions[:size]
    proposer.input_ids = torch.arange(16)
    proposer.parallel_drafting_token_id = 0
    proposer._slot_mapping_buffer = torch.arange(16)
    proposer._per_group_context_slot_mapping_buffers = {0: torch.arange(16)}
    base_type = load_methods("llm_base_proposer.py", "AscendSpecDecodeBaseProposer", {"_pad_draft_buffers"})
    proposer._pad_draft_buffers = base_type._pad_draft_buffers.__get__(proposer)
    proposer._runnable = lambda **kwargs: events.append("graph")
    proposer.dummy_run(
        16, num_reqs=2, aclgraph_runtime_mode=GraphMode.FULL, batch_descriptor=SimpleNamespace(num_tokens=16)
    )
    assert events == ["submit", "wait", "enter", "graph", "exit", "release"]
    metadata = captured[0]
    assert metadata.query_start_loc_cpu.tolist() == [0, 7, 14, 16]
    assert metadata.seq_lens.tolist() == metadata.seq_lens_cpu.tolist() == [7, 7, 0]
    assert metadata.block_table_tensor.shape[0] == metadata.num_reqs == 3
    assert torch.all(metadata.slot_mapping == -1)
    assert torch.all(proposer._per_group_context_slot_mapping_buffers[0] == -1)
    if dcp_size > 1:
        cp = metadata.context_parallel_metadata
        assert cp.num_computed_tokens_of_dcp.tolist() == [[0, 0], [0, 0], [0, 0]]
        assert cp.draft_cp_seq_len.tolist() == [4, 4, 0]
        assert cp.query_lens_cpu.tolist() == [7, 7, 2]


def load_dcp_manager(monkeypatch, size, rank, interleave):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/context_parallel/common_cp.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    func = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_dcp_local_seq_lens"
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[func], type_ignores=[]), str(source), "exec"), namespace)
    monkeypatch.setitem(sys.modules, "vllm_ascend.attention.utils", SimpleNamespace(AscendDCPMetadata=SimpleNamespace))
    manager_type = load_methods(
        "../worker/dcp_utils.py",
        "DCPManager",
        {"_get_dcp_local_seq_lens", "prepare_dspark_first_pass_cp_metadata"},
        get_dcp_local_seq_lens=namespace[func.name],
    )
    manager = manager_type.__new__(manager_type)
    manager.dcp_world_size = size
    manager.dcp_world_rank = rank
    manager.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=interleave))
    return manager


@pytest.mark.parametrize("size,rank", [(2, 0), (2, 1), (4, 3)])
@pytest.mark.parametrize("interleave", [1, 128])
def test_dspark_dcp_history_and_total_lengths_use_device_epoch(monkeypatch, size, rank, interleave):
    manager = load_dcp_manager(monkeypatch, size, rank, interleave)
    common = SimpleNamespace(
        num_reqs=3,
        seq_lens=torch.tensor([255, 259, 0], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([-99, -99, -99], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 3, 6, 16], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 3, 6, 16], dtype=torch.int32),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("DCP Flash metadata must not read back device lengths")

    with monkeypatch.context() as no_readback:
        for name in ("cpu", "numpy", "tolist"):
            no_readback.setattr(torch.Tensor, name, forbidden)
        manager.prepare_dspark_first_pass_cp_metadata(common, 3, use_device_seq_lens=True)

    cp = common.context_parallel_metadata
    expected_history = [
        [sum(position // interleave % size == r for position in range(length)) for r in range(size)]
        for length in (252, 256, 0)
    ]
    expected_total = [
        sum(position // interleave % size == rank for position in range(length)) for length in (255, 259, 0)
    ]
    assert cp.num_computed_tokens_of_dcp.tolist() == expected_history
    assert cp.draft_cp_seq_len.tolist() == expected_total
    assert cp.query_lens_cpu.tolist() == [3, 3, 10]
    assert cp.dcp_mtp_attn_mask is None


@pytest.mark.parametrize("real_reqs", [1, 2])
def test_dcp_propose_pads_metadata_without_mtp_table_clone(monkeypatch, real_reqs):
    proposer_type = load_methods(
        "llm_base_proposer.py",
        "AscendSpecDecodeBaseProposer",
        {"_propose", "_adjust_tensor", "_pad_draft_buffers", "build_draft_attn_metadata"},
        envs=SimpleNamespace(VLLM_ASCEND_ENABLE_FLASH_MLA=True),
    )
    draft_type = load_methods("dspark_proposer.py", "AscendDSparkProposer", {"pad_query_start_loc_for_graph"})
    proposer = proposer_type.__new__(proposer_type)
    proposer.method = "dspark"
    proposer.dcp_size = 2
    proposer.num_query_per_req = proposer.num_speculative_tokens = 3
    proposer.decode_threshold = 4
    proposer.parallel_drafting = proposer.use_cuda_graph = True
    proposer.supports_mm_inputs = proposer.uses_mrope = proposer.use_compress = False
    proposer.draft_window_size = proposer.sliding_window = None
    proposer.block_table_tensor_clone = None
    proposer.hidden_size = 4
    proposer.model = SimpleNamespace(combine_hidden_states=lambda value: value)
    proposer.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(prefill_context_parallel_size=1))
    proposer.pad_query_start_loc_for_graph = draft_type.pad_query_start_loc_for_graph.__get__(proposer)
    starts = torch.zeros(4, dtype=torch.int32)
    proposer.query_start_loc = SimpleNamespace(cpu=starts, gpu=starts, np=starts.numpy(), copy_to_gpu=lambda: None)
    proposer.query_start_loc_group = [torch.zeros(4, dtype=torch.int32)]
    proposer.seq_lens_group = [torch.zeros(3, dtype=torch.int32)]
    proposer.slot_mapping_group = [torch.full((8,), -1, dtype=torch.int32)]
    proposer.input_ids = torch.zeros(8, dtype=torch.int32)
    proposer.positions = torch.arange(8)
    proposer.token_indices_to_sample = torch.zeros(8, dtype=torch.int32)
    proposer.parallel_drafting_token_id = 0
    proposer._dflash_num_context = 2
    proposer._slot_mapping_buffer = torch.arange(8)
    proposer._per_group_query_slot_mapping_buffers = {0: torch.arange(8)}
    proposer._per_group_context_slot_mapping_buffers = {0: torch.arange(8)}
    proposer._per_group_block_table_buffers = {0: torch.zeros(3, 4, dtype=torch.int32)}
    proposer._context_slot_mapping_buffers = None
    captured = []
    provider = MetadataProvider()
    provider.take_device_metadata_tasks = lambda: ()
    provider.build_for_drafting = lambda metadata, **kwargs: captured.append(metadata) or SimpleNamespace()
    proposer.draft_attn_groups = [
        SimpleNamespace(
            kv_cache_group_id=0,
            layer_names=["draft"],
            get_metadata_builder=lambda: provider,
        )
    ]
    descriptor = SimpleNamespace(num_tokens=8, num_reqs=2)
    proposer.runner = SimpleNamespace(
        dcp_manager=load_dcp_manager(monkeypatch, 2, 0, 1),
        device_metadata_executor=None,
        input_batch=SimpleNamespace(lora_id_to_lora_request={}),
        cudagraph_dispatcher=SimpleNamespace(dispatch=lambda **kwargs: (GraphMode.FULL, descriptor)),
        _sync_metadata_across_dp=lambda n, **kwargs: (n, None, None),
    )
    common = SimpleNamespace(
        batch_size=lambda: real_reqs,
        num_reqs=real_reqs,
        query_start_loc=torch.arange(real_reqs + 1, dtype=torch.int32) * 3,
        query_start_loc_cpu=torch.arange(real_reqs + 1, dtype=torch.int32) * 3,
        seq_lens=torch.tensor([255, 259][:real_reqs], dtype=torch.int32),
        seq_lens_cpu=None,
        _seq_lens_cpu=None,
        num_computed_tokens_cpu=None,
        is_prefilling=torch.zeros(real_reqs, dtype=torch.bool),
        block_table_tensor=torch.ones(real_reqs, 4, dtype=torch.int32),
        slot_mapping=torch.arange(real_reqs * 3),
    )
    proposer.set_inputs_first_pass = lambda **kwargs: (real_reqs * 3, torch.arange(real_reqs * 3), common, None)

    def stop_at_context_update(*args):
        raise RuntimeError("context update reached")

    proposer.build_model_inputs_first_pass = stop_at_context_update
    with pytest.raises(RuntimeError, match="context update reached"):
        proposer._propose(
            3,
            torch.zeros(real_reqs * 3),
            torch.arange(real_reqs * 3),
            torch.zeros(real_reqs * 3, 4),
            torch.zeros(real_reqs),
            torch.arange(real_reqs * 3),
            common,
            SimpleNamespace(uniform=True),
            SimpleNamespace(),
        )
    metadata = captured[0]
    assert proposer.block_table_tensor_clone is None
    assert common.block_table_tensor.shape == metadata.block_table_tensor.shape == (3, 4)
    assert metadata.query_start_loc.tolist() == [0, 3, 6, 8]
    assert metadata.seq_lens.tolist() == [*([255, 259][:real_reqs]), *([0] * (3 - real_reqs))]
    assert len(metadata.is_prefilling) == metadata.num_reqs == 3
    assert torch.all(metadata.slot_mapping[real_reqs * 3 :] == -1)
    cp = metadata.context_parallel_metadata
    assert cp.query_lens_cpu.tolist() == [3, 3, 2]
    assert cp.num_computed_tokens_of_dcp[real_reqs:].count_nonzero() == 0
    assert cp.draft_cp_seq_len.tolist() == [*([128, 130][:real_reqs]), *([0] * (3 - real_reqs))]


@pytest.mark.parametrize("dcp_size", [2, 4])
@pytest.mark.parametrize("interleave", [1, 3])
def test_dcp_query_slots_keep_owner_and_strided_block_table(dcp_size, interleave):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton/spec_decode/utils.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "copy_and_expand_dflash_and_dspark_inputs_kernel"
    )
    # Execute the production query-slot arithmetic with CPU tensors. The
    # pointer shim checks physical row stride without a Triton/NPU launch.
    statements = None
    for node in ast.walk(kernel):
        for body in (getattr(node, "body", []), getattr(node, "orelse", [])):
            for index, statement in enumerate(body):
                if isinstance(statement, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "query_kv_slot_pos" for target in statement.targets
                ):
                    end = next(
                        j
                        for j in range(index, len(body))
                        if isinstance(body[j], ast.Expr)
                        and isinstance(body[j].value, ast.Call)
                        and isinstance(body[j].value.func, ast.Attribute)
                        and body[j].value.func.attr == "store"
                    )
                    statements = body[index:end]
                    break
    assert statements is not None
    code = compile(ast.Module(body=statements, type_ignores=[]), str(source), "exec")
    backing = torch.full((24,), -99, dtype=torch.int32)
    table = backing.as_strided((2, 5), (9, 1), 2)
    table.copy_(torch.tensor([[9, 2, 11, 5, 7], [3, 13, 4, 1, 15]]))
    req_indices = torch.arange(2).repeat_interleave(5)
    query_indices = torch.arange(5).repeat(2)
    effective_lengths = torch.tensor([7, 15]).repeat_interleave(5)
    actual_by_rank = []
    for rank in range(dcp_size):
        namespace = dict(
            effective_seq_len=effective_lengths,
            q_idx=query_indices,
            req_idx=req_indices,
            DCP_SIZE=dcp_size,
            DCP_RANK=rank,
            CP_INTERLEAVE_SIZE=interleave,
            block_size=4,
            block_table_ptr=table.storage_offset(),
            block_table_stride=table.stride(0),
            mask=torch.ones(10, dtype=torch.bool),
            tl=SimpleNamespace(int64=torch.int64, where=torch.where, load=lambda ptr, **kwargs: backing[ptr]),
        )
        exec(code, namespace)
        actual = namespace["slot_q"]
        expected = []
        for req, length in enumerate([7, 15]):
            for query in range(5):
                position = length + query
                if position // interleave % dcp_size != rank:
                    expected.append(-1)
                    continue
                local_position = sum(p // interleave % dcp_size == rank for p in range(position))
                expected.append(int(table[req, local_position // 4]) * 4 + local_position % 4)
        assert actual.tolist() == expected
        actual_by_rank.append(actual)
    assert torch.all(torch.stack(actual_by_rank).ge(0).sum(dim=0) == 1)
