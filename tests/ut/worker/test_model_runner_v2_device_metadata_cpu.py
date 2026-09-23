# SPDX-License-Identifier: Apache-2.0
"""CPU checks for MRv2 metadata ownership and graph replay integration."""

import ast
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).parents[3]


def load_functions(names, **namespace):
    path = ROOT / "vllm_ascend/worker/v2/attn_utils.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace.update(contextmanager=contextmanager)
    exec(
        compile("from __future__ import annotations\n" + "\n".join(map(ast.unparse, definitions)), str(path), "exec"),
        namespace,
    )
    return SimpleNamespace(**namespace)


class Executor:
    def __init__(self, name, log):
        self.name = name
        self.log = log
        self.submission_in_flight = False

    def submit(self, tasks):
        assert not self.submission_in_flight
        self.submission_in_flight = True
        self.log.append((self.name, "submit"))
        for task in tasks:
            task.run()

    def wait(self, stage, group_id):
        assert self.submission_in_flight
        self.log.append((self.name, "wait", stage, group_id))

    def release(self):
        assert self.submission_in_flight
        self.log.append((self.name, "release"))
        self.submission_in_flight = False


def test_nested_target_and_draft_restore_owner_and_release_after_consumer():
    current = ContextVar("test_executor", default=None)
    functions = load_functions({"device_metadata_context"}, _device_metadata_executor=current)
    log = []
    target, draft = Executor("target", log), Executor("draft", log)
    with functions.device_metadata_context(target):
        target.submit([])
        with functions.device_metadata_context(target):
            assert current.get() is target
        assert target.submission_in_flight
        log.append(("target", "consumer"))
        with pytest.raises(ValueError), functions.device_metadata_context(draft):
            draft.submit([])
            log.append(("draft", "consumer"))
            raise ValueError("consumer failed after enqueue")
        assert current.get() is target
        assert target.submission_in_flight
    assert current.get() is None
    assert log == [
        ("target", "submit"),
        ("target", "consumer"),
        ("draft", "submit"),
        ("draft", "consumer"),
        ("draft", "release"),
        ("target", "release"),
    ]


def test_build_waits_outside_graph_and_retains_buffers_until_consumed():
    current = ContextVar("test_executor", default=None)
    log = []
    executor = Executor("target", log)
    tasks = [SimpleNamespace(stage=2, group_id=4, run=lambda: log.append(("tiling", "run")))]

    class Provider:
        def enable_device_metadata(self):
            log.append(("builder", "enable"))

        def build(self, *, common_prefix_len, common_attn_metadata):
            return SimpleNamespace(common=common_attn_metadata)

        def take_device_metadata_tasks(self):
            return tasks

    capturing = False
    functions = load_functions(
        {"device_metadata_context", "build_attn_metadata"},
        _device_metadata_executor=current,
        np=np,
        torch=SimpleNamespace(
            from_numpy=torch.from_numpy, npu=SimpleNamespace(is_current_stream_capturing=lambda: capturing)
        ),
        DeviceMetadataTaskProvider=Provider,
        AscendCommonAttentionMetadata=lambda **kwargs: SimpleNamespace(**kwargs),
        GDNAttentionMetadataBuilder=type("GDN", (), {}),
        AscendDSAMetadataBuilder=type("DSA", (), {}),
        AscendSFAMetadataBuilder=type("SFA", (), {}),
    )
    provider = Provider()
    common = dict(
        attn_groups=[[SimpleNamespace(get_metadata_builder=lambda _: provider, layer_names=["target.0", "target.1"])]],
        num_reqs=2,
        num_tokens=7,
        query_start_loc_gpu=torch.tensor([0, 7, 7]),
        query_start_loc_cpu=torch.tensor([0, 7, 7]),
        seq_lens=torch.tensor([20, 0]),
        max_query_len=7,
        max_seq_len=20,
        block_tables=[torch.ones((2, 1), dtype=torch.int32)],
        slot_mappings=[torch.arange(7)],
        positions=torch.arange(7),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[object()]),
    )
    with functions.device_metadata_context(executor):
        metadata = functions.build_attn_metadata(**common)
        assert metadata["target.0"] is metadata["target.1"]
        assert executor.submission_in_flight
        assert ("target", "release") not in log
        assert log[-1] == ("target", "wait", 2, 4)
        log.append(("target", "consumer"))
        functions.build_attn_metadata(**common)
        assert log.index(("target", "release")) > log.index(("target", "consumer"))
        log.append(("target", "consumer2"))
    assert log[-1] == ("target", "release")
    capturing = True
    with functions.device_metadata_context(executor), pytest.raises(AssertionError):
        functions.build_attn_metadata(**common)
    assert not executor.submission_in_flight


def test_flash_builder_refinement_preserves_scheduler_groups_and_query_heads():
    class Group:
        def __init__(self, backend, names, spec, group_id):
            self.backend, self.layer_names, self.kv_cache_spec, self.kv_cache_group_id = backend, names, spec, group_id
            self.metadata_builders = [object(), object()]

        def create_metadata_builders(self, **kwargs):
            self.created_with = kwargs
            self.metadata_builders = [
                SimpleNamespace(names=self.layer_names) for _ in range(kwargs["num_metadata_builders"])
            ]

    layers = {
        "mla.rope.0": SimpleNamespace(impl=SimpleNamespace(use_flash_mla=True, use_mla_rope=True, scale=0.4)),
        "mla.plain": SimpleNamespace(impl=SimpleNamespace(use_flash_mla=True, use_mla_rope=False, scale=0.2)),
        "mla.rope.1": SimpleNamespace(impl=SimpleNamespace(use_flash_mla=True, use_mla_rope=True, scale=0.4)),
        "gqa.heads8": SimpleNamespace(impl=SimpleNamespace(scale=0.125)),
        "gqa.heads4": SimpleNamespace(impl=SimpleNamespace(scale=0.125)),
    }
    groups = [
        [
            Group("mla", ["mla.rope.1", "mla.plain", "mla.rope.0"], "latent", 0),
            Group("gqa", ["gqa.heads8"], "gqa", 0),
            Group("gqa", ["gqa.heads4"], "gqa", 0),
        ],
        [],
        [],
        [],
    ]
    scheduler_groups = [object() for _ in range(4)]
    cache = SimpleNamespace(kv_cache_groups=scheduler_groups)
    functions = load_functions(
        {"init_attn_backend"},
        _upstream_init_attn_backend=lambda *args, **kwargs: (groups, "support", [128, 128, 128, 128]),
        ascend_envs=SimpleNamespace(VLLM_ASCEND_ENABLE_FLASH_MLA=True),
        get_layers_from_vllm_config=lambda *args: layers,
        AttentionLayerBase=object,
        AttentionGroup=Group,
    )
    result, support, sizes = functions.init_attn_backend(cache, "config", "cpu")
    assert cache.kv_cache_groups is scheduler_groups
    assert len(result) == 4
    assert [group.layer_names for group in result[0]] == [
        ["mla.rope.0", "mla.rope.1"],
        ["mla.plain"],
        ["gqa.heads8"],
        ["gqa.heads4"],
    ]
    assert support == "support" and sizes == [128] * 4
    assert all(group.kv_cache_group_id == 0 for group in result[0])
    assert all(len(group.metadata_builders) == 2 for group in result[0])


def test_flash_draft_graph_replay_does_not_rebuild_metadata_or_update_fia():
    path = ROOT / "vllm_ascend/worker/v2/spec_decode/dflash/aclgraph.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "run_fullgraph"]
    cls.bases = [ast.Name(id="Parent", ctx=ast.Load())]
    namespace = {
        "Parent": type("Parent", (), {"run_fullgraph": lambda self, desc: ("replayed", desc)}),
        "ascend_envs": SimpleNamespace(VLLM_ASCEND_ENABLE_FLASH_MLA=True),
    }
    exec(compile("from __future__ import annotations\n" + ast.unparse(cls), str(path), "exec"), namespace)
    manager = namespace[cls.name]()
    manager.speculator = SimpleNamespace(attn_architecture="MLA")
    assert manager.run_fullgraph("bucket") == ("replayed", "bucket")


def load_dspark_methods(names, **namespace):
    path = ROOT / "vllm_ascend/worker/v2/spec_decode/dspark/speculator.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    cls.bases = []
    namespace.update(torch=torch, contextmanager=contextmanager)
    exec(compile("from __future__ import annotations\n" + ast.unparse(cls), str(path), "exec"), namespace)
    return namespace[cls.name]


def test_dense_mla_draft_capture_publishes_decode_state_and_restores_factory():
    original = object()
    module = SimpleNamespace(build_attn_metadata=original)
    cls = load_dspark_methods(
        {"draft_capture_context"},
        dflash_cudagraph=module,
        ascend_envs=SimpleNamespace(VLLM_ASCEND_ENABLE_FLASH_MLA=True),
        AscendAttentionState=SimpleNamespace(SpecDecoding="spec"),
        build_attn_metadata=lambda **kwargs: kwargs,
    )
    speculator = cls()
    speculator.attn_architecture = "MLA"
    speculator.input_buffers = SimpleNamespace(positions=torch.arange(28))
    with pytest.raises(ValueError), speculator.draft_capture_context():
        metadata = module.build_attn_metadata(num_reqs=4, num_tokens=28)
        assert metadata["attn_state"] == "spec"
        assert metadata["is_prefilling"].dtype == torch.bool
        assert metadata["is_prefilling"].tolist() == [False] * 4
        assert metadata["positions"].data_ptr() == speculator.input_buffers.positions.data_ptr()
        raise ValueError("capture failed")
    assert module.build_attn_metadata is original


@pytest.mark.parametrize("architecture", ["MLA", None])
def test_draft_padding_preserves_device_flash_cu_and_updates_only_legacy_metadata(architecture):
    cls = load_dspark_methods(
        {"_update_draft_attn_metadata"},
        AscendAttentionState=SimpleNamespace(SpecDecoding="spec"),
    )
    speculator = cls()
    speculator.attn_architecture = architecture
    speculator.num_query_per_req = 7
    flash = SimpleNamespace(flash=SimpleNamespace(cu=torch.tensor([0, 7, 7, 28])), actual_seq_lengths_q=None)
    legacy = SimpleNamespace(decode=SimpleNamespace(actual_seq_lengths_q=None), actual_seq_lengths_q=None)
    entries = {"flash": flash, "legacy": legacy}
    assert speculator._update_draft_attn_metadata(entries, 4) is entries
    assert flash.flash.cu.tolist() == [0, 7, 7, 28]
    assert flash.actual_seq_lengths_q is None
    actual = legacy.decode if architecture == "MLA" else legacy
    assert actual.actual_seq_lengths_q == [7, 14, 21, 28]
