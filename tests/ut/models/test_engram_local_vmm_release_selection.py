# SPDX-License-Identifier: Apache-2.0
"""Check local VMM selection without importing the full vLLM runtime."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


def resolve(config, vllm_config):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/ascend_config.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendConfig")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "derive_and_validate")
    # Run the real selection statements, stopping before unrelated platform logic.
    stop = next(
        i
        for i, node in enumerate(method.body)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "self.enable_force_eplb"
    )
    function = ast.parse("def select(self, vllm_config):\n    pass\n").body[0]
    assert isinstance(function, ast.FunctionDef)
    function.body = method.body[:stop]
    code = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    scope = {}
    exec(compile(code, str(source), "exec"), scope)
    scope["select"](config, vllm_config)
    return config.engram_local_vmm


@pytest.mark.parametrize(
    "external,explicit,cpu_offload,expected",
    [
        (None, None, True, True),
        (None, False, True, False),
        (None, True, True, True),
        (None, None, False, False),
        ("pool", None, True, False),
        ("rdma", None, True, False),
        ("pool", False, True, False),
        ("rdma", False, True, False),
    ],
)
def test_local_vmm_selection_preserves_existing_backends(external, explicit, cpu_offload, expected):
    config = SimpleNamespace(
        engram_local_vmm=explicit if explicit is not None else True,
        engram_shared_host_dir="/tmp/pool" if external == "pool" else None,
        shared_engram_rdma=object() if external == "rdma" else None,
    )
    vc = SimpleNamespace(
        engram_config=SimpleNamespace(cpu_offload=cpu_offload, dp_shared_memory=True),
        model_config=SimpleNamespace(architecture="DeepseekV41ForCausalLM", enable_sleep_mode=False),
        load_config=SimpleNamespace(load_format="auto"),
        use_v2_model_runner=False,
        additional_config={} if explicit is None else {"engram_local_vmm": explicit},
    )
    assert resolve(config, vc) is expected


@pytest.mark.parametrize("external", ["pool", "rdma"])
def test_explicit_local_vmm_rejects_external_backend(external):
    config = SimpleNamespace(
        engram_local_vmm=True,
        engram_shared_host_dir="/tmp/pool" if external == "pool" else None,
        shared_engram_rdma=object() if external == "rdma" else None,
    )
    vc = SimpleNamespace(engram_config=SimpleNamespace(cpu_offload=True), additional_config={"engram_local_vmm": True})
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve(config, vc)


def test_implicit_local_vmm_does_not_change_non_engram_models():
    config = SimpleNamespace(engram_local_vmm=True, engram_shared_host_dir=None, shared_engram_rdma=None)
    assert not resolve(config, SimpleNamespace(engram_config=None, additional_config={}))


def test_implicit_local_vmm_preserves_private_host_tables():
    config = SimpleNamespace(engram_local_vmm=True, engram_shared_host_dir=None, shared_engram_rdma=None)
    vc = SimpleNamespace(
        engram_config=SimpleNamespace(cpu_offload=True, dp_shared_memory=False),
        additional_config={},
        model_config=SimpleNamespace(architecture="DeepseekV41ForCausalLM"),
        use_v2_model_runner=False,
    )
    assert not resolve(config, vc)
