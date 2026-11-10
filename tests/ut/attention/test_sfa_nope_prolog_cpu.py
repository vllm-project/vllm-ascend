# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of SFA routing and composition; CANN execution is emulated."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def _load(path, name, scope, owner=None):
    nodes = ast.parse((ROOT / path).read_text(encoding="utf-8")).body
    for node in nodes:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "SFA_NOPE_PROLOG_SUPPORTED_DIMS" for target in node.targets
        ):
            scope["SFA_NOPE_PROLOG_SUPPORTED_DIMS"] = ast.literal_eval(node.value)
    if owner:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == owner).body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[])), path, "exec"), scope)
    return scope[name]


@pytest.fixture
def state(monkeypatch):
    dynamic = type("AscendW8A8DynamicLinearMethod", (), {})
    mx = type("AscendW8A8MXFP8DynamicLinearMethod", (), {})
    profile = SimpleNamespace(supports=lambda capability: True)
    types = SimpleNamespace(NATIVE="native", PROLOG_V3="prolog_v3", MLAPO="mlapo")
    op = Mock()
    monkeypatch.setattr(
        torch.ops._C_ascend,
        "npu_mla_prolog_v3_k3",
        Mock(side_effect=AssertionError("NoPE must use torch_npu.npu_mla_prolog_v3")),
        raising=False,
    )
    scope = {
        "torch": torch,
        "torch_npu": SimpleNamespace(
            npu_dynamic_quant=Mock(return_value=(torch.zeros(4, 8, dtype=torch.int8), torch.ones(4))),
            npu_mla_prolog_v3=op,
        ),
        "PreprocessType": types,
        "HardwareCapability": SimpleNamespace(SFA_NOPE_PROLOG_V3=object()),
        "AscendW8A8DynamicLinearMethod": dynamic,
        "AscendW8A8MXFP8DynamicLinearMethod": mx,
        "AscendW8A8LinearMethod": type("AscendW8A8LinearMethod", (), {}),
        "logger": Mock(),
        "get_current_hardware_profile": lambda: profile,
    }
    page_view = _load("vllm_ascend/attention/sfa_v1.py", "_view_cache_as_operator_pages", {"torch": torch})
    scope["_view_cache_as_operator_pages"] = page_view
    impl = SimpleNamespace(
        qk_rope_head_dim=0,
        qk_nope_head_dim=256,
        q_lora_rank=1536,
        kv_lora_rank=512,
        local_num_heads=2,
        fused_qkv_a_proj=SimpleNamespace(quant_method=SimpleNamespace(quant_method=dynamic())),
        q_proj=SimpleNamespace(_chunk_size=0),
        enable_sparse_sfa_c8=True,
        enable_sparse_sfa_turboquant=False,
        c8_cache_dtype=torch.int8,
        has_indexer=True,
        indexer=SimpleNamespace(wq_b=SimpleNamespace(weight_scale=torch.ones(1))),
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(prefill_context_parallel_size=1, decode_context_parallel_size=1)
        ),
        q_a_layernorm=SimpleNamespace(weight=torch.ones(1536), variance_epsilon=1e-5),
        kv_a_layernorm=SimpleNamespace(weight=torch.ones(512), variance_epsilon=1e-6),
        is_kv_consumer=False,
        enable_mlapo=False,
        sfa_qsfa_tile_size=128,
        sfa_qsfa_k_nope_clip_alpha=torch.ones(1),
        sfa_qsfa_kr_cache_dummy=torch.empty(0, dtype=torch.bfloat16),
        _process_weights_for_fused_prolog_v3=Mock(),
        _fused_preprocess_type=Mock(return_value=None),
    )
    for name in (
        "_resolve_preprocess_type",
        "_get_fused_type_unsupported_reasons",
        "_get_nope_prolog_unsupported_reasons",
        "_sfa_preprocess_prolog_v3",
        "_fused_preprocess_type",
        "_try_enable_type",
        "forward",
    ):
        fn = _load("vllm_ascend/attention/sfa_v1.py", name, scope, "AscendSFAImpl")
        setattr(impl, name, fn.__get__(impl))
    impl._fused_preprocess_type = Mock(wraps=impl._fused_preprocess_type)
    impl._get_layer_quant_method = lambda layer: getattr(getattr(layer, "quant_method", None), "quant_method", None)
    for name in (
        "weight_dq",
        "weight_uq_qr",
        "weight_dkv_kr",
        "W_UK_T",
        "dequant_scale_w_dq",
        "dequant_scale_w_uq_qr",
        "dequant_scale_w_dkv_kr",
    ):
        setattr(impl, name, torch.empty(1))
    return SimpleNamespace(impl=impl, scope=scope, profile=profile, op=op, types=types, mx=mx)


def test_nope_automatically_resolves_and_prepares_weights(state):
    assert state.impl._resolve_preprocess_type(torch.bfloat16) == state.types.PROLOG_V3
    state.impl._process_weights_for_fused_prolog_v3.assert_called_once()
    state.impl._fused_preprocess_type.assert_called_once()


@pytest.mark.parametrize("is_kv_consumer", [False, True])
def test_nope_bf16_projections_preserve_native_weights(state, is_kv_consumer):
    state.impl.fused_qkv_a_proj.quant_method.quant_method = None
    state.impl.is_kv_consumer = is_kv_consumer
    assert state.impl._resolve_preprocess_type(torch.bfloat16) == state.types.NATIVE
    state.impl._process_weights_for_fused_prolog_v3.assert_not_called()


def test_nope_does_not_require_custom_operator_registration(state, monkeypatch):
    monkeypatch.delattr(torch.ops._C_ascend, "npu_mla_prolog_v3_k3")
    assert state.impl._resolve_preprocess_type(torch.bfloat16) == state.types.PROLOG_V3
    state.impl._process_weights_for_fused_prolog_v3.assert_called_once()


@pytest.mark.parametrize(
    "unsupported",
    [
        "hardware",
        "mxfp8",
        "c16",
        "fp8_cache",
        "D",
        "Hcq",
        "Hckv",
        "pcp",
        "dcp",
        "indexer",
        "fp16",
        "turboquant",
        "chunked",
    ],
)
def test_nope_unsupported_contracts_fall_back_before_weight_changes(state, unsupported):
    impl = state.impl
    dtype = torch.bfloat16
    if unsupported == "hardware":
        state.profile.supports = lambda capability: False
    elif unsupported == "mxfp8":
        impl.fused_qkv_a_proj.quant_method.quant_method = state.mx()
    elif unsupported == "c16":
        impl.enable_sparse_sfa_c8 = False
    elif unsupported == "fp8_cache":
        impl.c8_cache_dtype = torch.float8_e4m3fn
    elif unsupported in ("D", "Hcq", "Hckv"):
        setattr(impl, {"D": "qk_nope_head_dim", "Hcq": "q_lora_rank", "Hckv": "kv_lora_rank"}[unsupported], 128)
    elif unsupported in ("pcp", "dcp"):
        name = "prefill_context_parallel_size" if unsupported == "pcp" else "decode_context_parallel_size"
        setattr(impl.vllm_config.parallel_config, name, 2)
    elif unsupported == "indexer":
        impl.indexer.wq_b = None
    elif unsupported == "turboquant":
        impl.enable_sparse_sfa_turboquant = True
    elif unsupported == "chunked":
        impl.q_proj._chunk_size = 1
    else:
        dtype = torch.float16
    assert impl._resolve_preprocess_type(dtype) == state.types.NATIVE
    impl._process_weights_for_fused_prolog_v3.assert_not_called()


@pytest.mark.parametrize("storage_block_size", [128, 640, 2304])
@pytest.mark.parametrize("rope_dim", [0, 64])
def test_prolog_preserves_cache_storage_slots_and_indexer_scales(state, storage_block_size, rope_dim):
    impl = state.impl
    impl._quant_type = state.scope["AscendW8A8DynamicLinearMethod"]
    impl.qk_rope_head_dim = rope_dim
    backing = torch.full((4, storage_block_size, 1, 528 + rope_dim * 2), -9, dtype=torch.int8)
    cache = backing[1:3]
    before = backing.clone()
    slots = torch.tensor([0, storage_block_size - 1, storage_block_size, -1], dtype=torch.int64)
    hidden = torch.randn(4, 8, dtype=torch.bfloat16)
    call_slots = torch.cat((slots, torch.tensor([7, 8]))) if rope_dim == 0 else slots
    q = torch.randn(4, 2, 512)
    qr = torch.empty(4, 2, rope_dim)
    qc = torch.zeros(4, 1536, dtype=torch.int8)
    scale = torch.arange(1, 5, dtype=torch.float32).view(4, 1)

    def prolog(**kwargs):
        written_cache = kwargs["kv_cache"]
        assert written_cache.data_ptr() == cache.data_ptr()
        assert kwargs["cache_index"].data_ptr() == call_slots.data_ptr()
        assert kwargs["kr_cache"] is impl.sfa_qsfa_kr_cache_dummy
        assert kwargs["weight_quant_mode"] == 2
        assert kwargs["kv_cache_quant_mode"] == 3
        assert kwargs["ckvkr_repo_mode"] == kwargs["quant_scale_repo_mode"] == 1
        assert kwargs["tile_size"] == 128
        if rope_dim == 0:
            assert kwargs["rope_sin"] is kwargs["rope_cos"] is impl.sfa_qsfa_kr_cache_dummy
            for row, slot in enumerate(kwargs["cache_index"].tolist()):
                if slot >= 0:
                    written_cache.view(-1, written_cache.shape[-1])[slot].fill_(row)
        return q, qr, torch.empty(0), qc, scale

    op = state.scope["torch_npu"].npu_mla_prolog_v3
    op.side_effect = prolog
    result = impl._sfa_preprocess_prolog_v3(
        hidden,
        (cache,),
        None if rope_dim == 0 else torch.ones(4, 1, 1, 64),
        None if rope_dim == 0 else torch.zeros(4, 1, 1, 64),
        call_slots,
        attn_metadata=SimpleNamespace(block_size=128 if storage_block_size > 1024 else storage_block_size),
    )
    # The NoPE caller trims slots after the per-step int64 conversion.
    if rope_dim == 0:
        assert op.call_args.kwargs["cache_index"].numel() == hidden.shape[0]
        for row, slot in enumerate(slots.tolist()):
            if slot >= 0:
                before[1 + slot // storage_block_size, slot % storage_block_size].fill_(row)
    torch.testing.assert_close(backing, before)
    assert result[0] is hidden
    torch.testing.assert_close(result[1], q)
    assert result[2].shape == (4, 2, rope_dim)
    assert result[3][0].data_ptr() == qc.data_ptr()
    torch.testing.assert_close(result[3][1], scale.view(-1))
    op.assert_called_once()


def test_nope_prolog_rejects_short_slot_mapping(state):
    with pytest.raises(AssertionError, match="one cache index per input token"):
        state.impl._sfa_preprocess_prolog_v3(torch.ones(4, 8), (), None, None, torch.arange(3))


def test_nope_preserves_bf16_indexer_query(state):
    impl = state.impl
    impl.indexer.wq_b = SimpleNamespace()
    assert impl._resolve_preprocess_type(torch.bfloat16) == state.types.PROLOG_V3
    assert impl._nope_native_indexer_query
    projected = torch.randn(4, 2048, dtype=torch.bfloat16)
    expected = torch.randn(4, 1536, dtype=torch.bfloat16)
    impl.fused_qkv_a_proj = Mock(return_value=(projected, None))
    impl.q_a_layernorm = Mock(return_value=expected)
    impl.q_a_layernorm.variance_epsilon = 1e-5
    state.op.return_value = (torch.ones(4, 2, 512), torch.empty(4, 2, 0), None, None, None)
    result = impl._sfa_preprocess_prolog_v3(
        torch.ones(4, 8, dtype=torch.bfloat16),
        (torch.empty(2, 128, 1, 528, dtype=torch.int8),),
        None,
        None,
        torch.arange(4),
        attn_metadata=SimpleNamespace(block_size=128),
    )
    assert not state.op.call_args.kwargs["query_norm_flag"]
    torch.testing.assert_close(impl.q_a_layernorm.call_args.args[0], projected[:, :1536])
    assert result[3] is expected


def test_nope_empty_eager_batch_does_not_call_prolog(state):
    state.scope["CUDAGraphMode"] = SimpleNamespace(FULL="full")
    state.scope["get_forward_context"] = lambda: SimpleNamespace(cudagraph_runtime_mode="none")
    output = torch.ones(4, 8)
    metadata = SimpleNamespace(slot_mapping=torch.arange(4), num_actual_tokens=0)
    result = state.impl.forward("sfa", torch.ones(4, 8), (), metadata, output)
    assert result is output
    assert output.count_nonzero() == 0
    state.op.assert_not_called()
