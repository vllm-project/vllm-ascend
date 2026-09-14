# SPDX-License-Identifier: Apache-2.0
"""Execute DSpark graph orchestration on CPU without importing the NPU stack."""

import ast
import copy
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any
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
    take_device_metadata_tasks: Any
    build_for_drafting: Any
    build_for_graph_capture: Any


def load_methods(filename, class_name, methods, **extra):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/spec_decode" / filename
    tree = ast.parse(source.read_text(encoding="utf-8"))
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    definition.bases = [ast.Name(id="ParentProposer", ctx=ast.Load())]
    definition.body = [node for node in definition.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    namespace = {
        "torch": torch,
        "copy": copy,
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
def test_metadata_waits_before_graph_and_preserves_eager_overlap(mode, dcp_size):
    proposer_type = load_methods("llm_base_proposer.py", "AscendSpecDecodeBaseProposer", {"build_draft_attn_metadata"})
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
        runner=SimpleNamespace(device_metadata_executor=executor),
        sliding_window=None,
        vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(prefill_context_parallel_size=1)),
        _per_group_block_table_buffers={},
        _per_group_query_slot_mapping_buffers={0: None, 1: None},
    )
    proposer_type.build_draft_attn_metadata(proposer, SimpleNamespace(num_reqs=1), 1, 1, aclgraph_runtime_mode=mode)
    expected = [] if dcp_size != 1 else [call.submit(tasks)]
    if dcp_size == 1 and mode is GraphMode.FULL:
        expected += [call.wait(task.stage, task.group_id) for task in tasks]
    assert executor.mock_calls == expected


def test_dummy_graph_waits_outside_context_and_disables_all_cache_writes():
    events = []
    captured = []
    task = SimpleNamespace(stage="attention", group_id=0)
    provider = MetadataProvider()
    provider.take_device_metadata_tasks = lambda: (task,)

    def build_for_graph_capture(metadata, state):
        captured.append(metadata)
        return SimpleNamespace()

    provider.build_for_graph_capture = build_for_graph_capture

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
    )
    proposer.num_query_per_req = proposer.num_speculative_tokens = 7
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
