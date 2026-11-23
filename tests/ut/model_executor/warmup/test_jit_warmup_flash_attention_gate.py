# SPDX-License-Identifier: Apache-2.0
"""Output-gate compile-key coverage, registration and shared launch contract."""

import ast
import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from tests.ut.model_executor.warmup.helpers import make_mock_worker, make_registry
from vllm_ascend.ops.triton import flash_attention_output as gate_ops


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("hidden,blocks", [(1536, {512, 2048, 4096, 8192}), (65536, {4096, 8192}), (0, set())])
@pytest.mark.parametrize("projected_aligned,gate_aligned", [(True, True), (True, False), (False, True), (False, False)])
def test_key_golden_set(dtype, hidden, blocks, projected_aligned, gate_aligned):
    owner = gate_ops._FLASH_ATTENTION_GATE_KERNEL
    keys = owner.get_warmup_keys(
        dtype=dtype,
        hidden=hidden,
        projected_stride=hidden + 16,
        gate_stride=hidden + 1,
        projected_aligned=projected_aligned,
        gate_aligned=gate_aligned,
    )
    assert set(keys) == {
        owner.CompileKey(dtype, hidden, hidden + 16, hidden + 1, block, projected_aligned, gate_aligned)
        for block in blocks
    }


@pytest.mark.parametrize("prefix", [0, 576, 1])
def test_registration_is_scoped_deduplicated_and_metadata_only(prefix):
    registry = make_registry()
    register = gate_ops.register_flash_attention_gate_warmup
    with patch.object(torch, "empty", side_effect=AssertionError("warmup must not allocate tensors")):
        register(torch.bfloat16, 1536, prefix)
        assert len(registry) == 0
        with registry.activate():
            register(torch.bfloat16, 1536, prefix)
            register(torch.bfloat16, 1536, prefix)
        assert len(registry) == 1
        owner = gate_ops._FLASH_ATTENTION_GATE_KERNEL
        args, kwargs = registry._registrations[owner][0]
        keys = owner.get_warmup_keys(*args, **kwargs)
        for key in keys:
            inputs = owner.warmup_inputs(key)
            assert inputs["projected"].stride(0) == 1536
            assert inputs["gate"].stride(0) == 1536 + prefix
            assert inputs["gate"].data_ptr() == (0 if prefix % 8 == 0 else 1)
            assert inputs["BLOCK"] == key.block


@pytest.mark.parametrize("tokens,block", [(5, 512), (6, 2048), (42, 2048), (43, 4096), (682, 4096), (683, 8192)])
@pytest.mark.parametrize("strided", [False, True])
def test_shared_launcher_preserves_runtime_grid_and_compile_options(tokens, block, strided):
    owner = gate_ops.FlashAttentionGateKernel()
    kernel = Mock()
    kernel.arg_names = list(inspect.signature(getattr(owner.kernel, "fn", owner.kernel)).parameters)
    launch = Mock()
    kernel.__getitem__ = Mock(return_value=launch)
    owner.kernel = kernel
    hidden = 1536
    a = torch.empty(tokens, hidden + (16 if strided else 0))[:, :hidden]
    g = torch.empty(tokens, hidden + (8 if strided else 0))[:, :hidden]
    keys = owner.get_warmup_keys(dtype=a.dtype, hidden=hidden, projected_stride=a.stride(0), gate_stride=g.stride(0))
    for key in keys:
        owner.compile(key)
    assert kernel.warmup.call_count == 4
    kernel.__getitem__.assert_not_called()
    assert {call.kwargs["BLOCK"] for call in kernel.warmup.call_args_list} == {512, 2048, 4096, 8192}
    assert all(call.kwargs["multibuffer"] is False for call in kernel.warmup.call_args_list)
    with (
        patch.object(gate_ops, "_FLASH_ATTENTION_GATE_KERNEL", owner),
        patch.object(gate_ops, "init_device_properties_triton"),
        patch.object(gate_ops, "get_vectorcore_num", return_value=24),
    ):
        assert gate_ops.flash_attention_gate(a, g) is a
    tasks = tokens * ((hidden + block - 1) // block) if strided else (tokens * hidden + block - 1) // block
    kernel.__getitem__.assert_called_once_with((min(tasks, 24),))
    args = launch.call_args.kwargs
    assert args["projected"] is a and args["gate"] is g
    assert args["numel"] == tokens * hidden
    assert args["HIDDEN"] == hidden and args["BLOCK"] == block
    assert args["PROJECTED_STRIDE"] == a.stride(0) and args["GATE_STRIDE"] == g.stride(0)
    assert args["multibuffer"] is False


@pytest.mark.parametrize("backend", ["mla", "sfa"])
@pytest.mark.parametrize("standalone,fused", [(False, False), (True, False), (False, True), (True, True)])
def test_attention_owner_registers_only_its_gate_layouts(backend, standalone, fused):
    # Execute the production constructor's registration branches without
    # constructing unrelated attention backends or distributed process groups.
    root = Path(__file__).resolve().parents[4]
    tree = ast.parse((root / f"vllm_ascend/attention/{backend}_v1.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == f"Ascend{backend.upper()}Impl")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    branches: list[ast.stmt] = [
        n
        for n in init.body
        if isinstance(n, ast.If)
        and any(
            isinstance(c, ast.Call)
            and isinstance(c.func, ast.Name)
            and c.func.id == "register_flash_attention_gate_warmup"
            for c in ast.walk(n)
        )
    ]
    assert len(branches) == (2 if backend == "mla" else 1)
    impl = SimpleNamespace(
        g_proj=object() if standalone else None,
        qkv_a_gate_fused=fused,
        num_heads=12,
        local_num_heads=12,
        v_head_dim=128,
        q_lora_rank=256,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16)),
    )
    registry = make_registry()
    with registry.activate():
        exec(
            compile(ast.Module(body=branches, type_ignores=[]), "<owner-registration>", "exec"),
            {
                "self": impl,
                "register_flash_attention_gate_warmup": gate_ops.register_flash_attention_gate_warmup,
            },
        )
    registrations = registry._registrations.get(gate_ops._FLASH_ATTENTION_GATE_KERNEL, [])
    expected = ({1536} if standalone else set()) | ({2368} if backend == "mla" and fused else set())
    assert {kwargs["gate_stride"] for _, kwargs in registrations} == expected


def test_disabled_registry_does_not_compile():
    worker = make_mock_worker(enable_jit_warmup=False)
    registry = worker.model_runner.jit_warmup_registry
    with registry.activate():
        gate_ops.register_flash_attention_gate_warmup(torch.bfloat16, 1536)
    warmup = importlib.import_module("vllm_ascend.model_executor.warmup.kernel_warmup")
    with patch.object(gate_ops._FLASH_ATTENTION_GATE_KERNEL, "compile") as compile_kernel:
        warmup.kernel_warmup(worker)
    compile_kernel.assert_not_called()
