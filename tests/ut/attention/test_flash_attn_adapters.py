# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests; native kernels are exercised by the A5 operator tests."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def load(relative):
    spec = importlib.util.spec_from_file_location("adapter", ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def ops(monkeypatch):
    package = ModuleType("cann_ops_transformer")
    module = ModuleType("cann_ops_transformer.ops")
    module.flash_attn_metadata = Mock(return_value="schedule")
    module.flash_attn = Mock(return_value=("output", "lse"))
    monkeypatch.setitem(sys.modules, "cann_ops_transformer", package)
    monkeypatch.setitem(sys.modules, "cann_ops_transformer.ops", module)
    return module


@pytest.mark.parametrize("causal", [False, True])
def test_bf16_prefill_schedule_matches_call(ops, causal):
    module = load("vllm_ascend/attention/flash_attn.py")
    q = torch.zeros(5, 4, 64, dtype=torch.bfloat16)
    k = torch.zeros(5, 2, 64, dtype=torch.bfloat16)
    cu = torch.tensor([0, 2, 5], dtype=torch.int32)
    mask = torch.zeros(2048, 2048, dtype=torch.int8) if causal else None
    assert module.flash_attn_prefill(q, k, k, cu, [2, 3], 0.125, mask) == "output"
    args, metadata = ops.flash_attn_metadata.call_args
    assert args == (4, 2, 64)
    assert metadata["head_dim_v"] == 64
    assert metadata["batch_size"] == 2
    call = ops.flash_attn.call_args.kwargs
    for name in ("cu_seqlens_q", "cu_seqlens_kv"):
        assert call[name] is cu and metadata[name] is cu
    assert call["max_seqlen_q"] == call["max_seqlen_kv"] == 3
    assert call["mask_mode"] == (3 if causal else 0)
    assert call["attn_mask"] is mask
    assert call["metadata"] == "schedule"


def test_expanded_bf16_shapes(ops):
    module = load("vllm_ascend/attention/flash_attn.py")
    schedule, attention = module.native_flash_adapters(2, 0.2, None)
    call = SimpleNamespace(
        query_cu=None,
        kv_cu=None,
        query_used=None,
        kv_used=None,
        query_lengths=[2],
        kv_lengths=[3],
        mask_mode=0,
        schedule="schedule",
    )
    schedule(call)
    q = torch.zeros(2, 2, 192, dtype=torch.bfloat16)
    k = torch.zeros(3, 2, 128, dtype=torch.bfloat16)
    v = torch.ones_like(k)
    rope = torch.ones(3, 1, 64, dtype=torch.bfloat16)
    attention(q, k, v, rope, call)
    args = ops.flash_attn.call_args.args
    assert args[1].shape == (3, 2, 192)
    torch.testing.assert_close(args[1][..., 128:], rope.expand(3, 2, 64))
    assert ops.flash_attn_metadata.call_args.kwargs["head_dim_v"] == 128


def test_c8_tile_bound_does_not_change_runtime_lengths(ops, monkeypatch):
    module = load("vllm_ascend/attention/flash_attn.py")
    quant = ModuleType("vllm_ascend.ops.flash_attn_c8_quant")
    quant.quantize_flash_attn_c8 = Mock(return_value=tuple(range(8)))
    quant.fake_quant_flash_attn_c8 = Mock()
    monkeypatch.setitem(sys.modules, quant.__name__, quant)
    native = Mock(return_value=("output", "lse"))
    monkeypatch.setattr(torch.ops._C_ascend, "flash_attn_c8", native, raising=False)
    schedule, attention = module.native_flash_adapters(2, 0.2, "mask", c8=True)
    call = SimpleNamespace(
        query_cu="qcu",
        kv_cu="kcu",
        query_used="qused",
        kv_used=None,
        query_lengths=[1, 3],
        kv_lengths=[4, 5],
        mask_mode=3,
        schedule="schedule",
    )
    schedule(call)
    assert ops.flash_attn_metadata.call_args.kwargs["max_seqlen_q"] == 65
    assert attention("q", "k", "v", "rope", call) == ("output", "lse")
    assert native.call_args.args == (*range(8), "qcu", "kcu", "schedule")
    assert native.call_args.kwargs["max_seqlen_q"] == 3
    assert native.call_args.kwargs["max_seqlen_kv"] == 5
    assert native.call_args.kwargs["seqused_q"] == "qused"
    with pytest.raises(ValueError, match="select either"):
        module.native_flash_adapters(2, 0.2, None, c8=True, fake_quant=True)


@pytest.mark.parametrize("causal", [False, True])
def test_backend_dispatch_uses_live_tokens(ops, causal):
    import ast

    source = ast.parse((ROOT / "vllm_ascend/attention/attention_v1.py").read_text(encoding="utf-8"))
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "AscendAttentionBackendImpl")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward_fused_infer_attention")
    fn.returns = None
    for arg in fn.args.args:
        arg.annotation = None
    call = Mock(return_value=torch.ones(5, 4, 64, dtype=torch.bfloat16))
    namespace = dict(
        torch=torch,
        _EXTRA_CTX=SimpleNamespace(capturing=False),
        get_current_hardware_profile=lambda: SimpleNamespace(supports=lambda _: True),
        HardwareCapability=SimpleNamespace(FLASH_ATTN_TND_PREFILL=1),
        AscendAttentionState=SimpleNamespace(PrefillNoCache=0),
        AttentionType=SimpleNamespace(DECODER=0),
        enable_dcp=lambda: False,
        envs_vllm=SimpleNamespace(VLLM_BATCH_INVARIANT=False),
        flash_attn_prefill=call,
    )
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "backend", "exec"), namespace)
    backend = SimpleNamespace(
        attn_type=0, head_size=64, sinks=None, sliding_window=None, pcp_enabled=False, scale=0.125
    )
    cu = torch.tensor([0, 2, 5], dtype=torch.int32)
    metadata = SimpleNamespace(
        attn_state=0,
        actual_seq_lengths_q=[2, 5],
        query_start_loc=cu,
        causal=causal,
        attn_mask=torch.zeros(2048, 2048, dtype=torch.bool),
    )
    q = torch.zeros(8, 4, 64, dtype=torch.bfloat16)
    k = torch.zeros(8, 2, 64, dtype=torch.bfloat16)
    output = torch.zeros_like(q)
    result = namespace[fn.name](backend, q, k, k, metadata, output)
    assert result is output
    assert torch.all(output[:5] == 1) and torch.all(output[5:] == 0)
    args = call.call_args.args
    assert args[0].shape == (5, 4, 64) and args[1].shape == (5, 2, 64)
    assert args[4] == [2, 3]
    if causal:
        assert args[6].dtype == torch.int8
    else:
        assert args[6] is None
