"""Post-load scale refresh and observable Direct/Native initialization routing."""

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch_npu  # noqa: F401
import vllm.logger as vllm_logger

# Match worker startup: register ops before importing the attention backend.
import vllm_ascend.ops  # noqa: F401
from vllm_ascend.attention import sfa_kv_offload as sfa
from vllm_ascend.attention import sfa_v1 as sfa_base
from vllm_ascend.attention.attention_v1 import AscendAttentionState


@pytest.fixture
def initialization(monkeypatch):
    config = SimpleNamespace(
        sparse_kv_offload_config=SimpleNamespace(keep_device_kv_cache=False, use_fused_overlap=True)
    )
    monkeypatch.setattr(sfa, "get_ascend_config", lambda: config)
    monkeypatch.setenv("VLLM_PATCH_TRACE", "1")
    # The test has no distributed group; exercise the real logger's once cache
    # on the rank that normally emits initialization messages.
    monkeypatch.setattr(vllm_logger, "_should_log_with_scope", lambda scope: True)
    vllm_logger._print_info_once.cache_clear()
    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            if "Sparse KV offload decode preprocessing selected:" in record.getMessage():
                records.append(record)

    handler = Capture()
    level = sfa.logger.level
    sfa.logger.setLevel(logging.INFO)
    sfa.logger.addHandler(handler)
    try:
        yield config, records
    finally:
        sfa.logger.removeHandler(handler)
        sfa.logger.setLevel(level)
        vllm_logger._print_info_once.cache_clear()


def make_impl(heads=16):
    # Use the actual class/methods without loading a model or touching an NPU.
    impl = sfa.AscendSFAKVOffloadImpl.__new__(sfa.AscendSFAKVOffloadImpl)
    scheme = sfa.AscendW8A8LinearMethod.__new__(sfa.AscendW8A8LinearMethod)
    impl.fused_qkv_a_proj = SimpleNamespace(
        weight=SimpleNamespace(shape=(6144, 2624)), quant_method=SimpleNamespace(quant_method=scheme)
    )
    impl.q_proj = SimpleNamespace(
        weight=SimpleNamespace(shape=(2048, heads * 256)),
        quant_method=SimpleNamespace(quant_method=scheme),
    )
    impl.W_UK_T = SimpleNamespace(shape=(heads, 192, 512))
    impl.use_fused_overlap = True
    impl.enable_mlapo = True
    impl.enable_dsa_cp = impl.enable_sparse_sfa_c8 = impl.enable_sparse_sfa_turboquant = False
    impl.num_heads, impl.num_kv_heads = heads, 1
    impl.q_lora_rank, impl.kv_lora_rank = 2048, 512
    impl.qk_nope_head_dim, impl.qk_rope_head_dim = 192, 64
    impl.q_a_layernorm = impl.kv_a_layernorm = object()
    impl.is_kv_consumer = True
    impl.gamma2 = torch.ones(512, dtype=torch.bfloat16)
    impl.quant_scale0 = torch.tensor([0.5], dtype=torch.bfloat16)
    impl.quant_scale1 = torch.tensor([0.25], dtype=torch.bfloat16)
    # Only weight reformatting is replaced; the actual precondition checks,
    # selection, scale refresh, allocation and INFO-once behavior still execute.
    impl._process_weights_for_fused_mlapo = Mock()
    return impl


@pytest.mark.parametrize("heads", [16, 32])
def test_reload_refreshes_fixed_scale_backing_and_logs_once(initialization, heads):
    _, records = initialization
    impl = make_impl(heads)
    assert impl._resolve_preprocess_type(torch.bfloat16) == sfa.PreprocessType.MLAPO
    storage = [getattr(impl, f"_mlapo_quant_scale_storage{i}") for i in (0, 1)]
    views = [getattr(impl, f"_mlapo_quant_scale{i}") for i in (0, 1)]
    pointers = [value.data_ptr() for value in storage + views]
    output_pointers = (impl._mlapo_current_k.data_ptr(), impl._mlapo_current_rope.data_ptr())
    impl.quant_scale0 = torch.tensor([0.75], dtype=torch.bfloat16)
    impl.quant_scale1 = torch.tensor([0.125], dtype=torch.bfloat16)
    assert impl._resolve_preprocess_type(torch.bfloat16) == sfa.PreprocessType.MLAPO
    assert pointers == [getattr(impl, f"_mlapo_quant_scale_storage{i}").data_ptr() for i in (0, 1)] + [
        getattr(impl, f"_mlapo_quant_scale{i}").data_ptr() for i in (0, 1)
    ]
    assert output_pointers == (impl._mlapo_current_k.data_ptr(), impl._mlapo_current_rope.data_ptr())
    for i, value in enumerate((0.75, 0.125)):
        assert torch.equal(storage[i], torch.full((16,), value, dtype=torch.bfloat16))
        assert views[i].item() == value
    assert len(records) == 1
    assert records[0].levelno == logging.INFO
    assert f"path=Direct heads={heads}" in records[0].getMessage()


@pytest.mark.parametrize(
    "condition, reason",
    [
        ("keep_cache", "keep_device_kv_cache=True"),
        ("copy_sfa", "use_fused_overlap=False"),
        ("disabled", "enable_mlapo=False"),
        ("activation", "activation_dtype=torch.float32"),
        ("down_quant", "fused_qkv_a_proj_quant="),
        ("up_quant", "q_proj_quant="),
        ("num_kv_heads", "num_kv_heads=2"),
        ("kv_lora_rank", "kv_lora_rank=256"),
        ("qk_rope_head_dim", "qk_rope_head_dim=32"),
        ("q_lora_rank", "q_lora_rank=1024"),
        ("qk_nope_head_dim", "qk_nope_head_dim=128"),
        ("hidden_size", "fused_qkv_a_proj_weight_shape=(4096, 2624)"),
        ("head_shape", "dimensions do not match the local head count"),
        ("q_norm", "Fused preprocessing requires q_a_layernorm and kv_a_layernorm."),
        ("kv_norm", "Fused preprocessing requires q_a_layernorm and kv_a_layernorm."),
        ("sparse_c8", "MLAPO does not support sparse C8; use PROLOG_V3 instead."),
        ("turboquant", "MLAPO does not support the TurboQuant 4-bit latent cache."),
        ("unsupported", "fused MLAPO unsupported"),
    ],
)
def test_native_fallback_records_real_reason(initialization, condition, reason):
    config, records = initialization
    impl = make_impl()
    dtype = torch.bfloat16
    if condition == "keep_cache":
        config.sparse_kv_offload_config.keep_device_kv_cache = True
    elif condition == "copy_sfa":
        impl.use_fused_overlap = False
    elif condition == "disabled":
        impl.enable_mlapo = False
    elif condition == "activation":
        dtype = torch.float32
    elif condition == "down_quant":
        impl.fused_qkv_a_proj.quant_method.quant_method = object()
    elif condition == "up_quant":
        impl.q_proj.quant_method.quant_method = object()
    elif condition in ("num_kv_heads", "kv_lora_rank", "qk_rope_head_dim", "q_lora_rank", "qk_nope_head_dim"):
        setattr(
            impl,
            condition,
            {
                "num_kv_heads": 2,
                "kv_lora_rank": 256,
                "qk_rope_head_dim": 32,
                "q_lora_rank": 1024,
                "qk_nope_head_dim": 128,
            }[condition],
        )
    elif condition == "hidden_size":
        impl.fused_qkv_a_proj.weight.shape = (4096, 2624)
    elif condition == "head_shape":
        impl.q_proj.weight.shape = (2048, 8 * 256)
    elif condition == "q_norm":
        impl.q_a_layernorm = None
    elif condition == "kv_norm":
        impl.kv_a_layernorm = None
    elif condition == "sparse_c8":
        impl.enable_sparse_sfa_c8 = True
    elif condition == "turboquant":
        impl.enable_sparse_sfa_turboquant = True
    elif condition == "unsupported":
        # Exercise the defensive branch when a future try-enable rejects a
        # configuration without one of today's documented rejection reasons.
        impl._try_enable_type = Mock(return_value=False)
    assert impl._resolve_preprocess_type(dtype) == sfa.PreprocessType.NATIVE
    assert impl._resolve_preprocess_type(dtype) == sfa.PreprocessType.NATIVE
    assert not hasattr(impl, "_mlapo_current_k")
    assert len(records) == 1
    assert records[0].levelno == logging.INFO
    assert "path=Native" in records[0].getMessage() and reason in records[0].getMessage()


def test_different_head_configurations_each_emit_initialization_info(initialization):
    _, records = initialization
    for heads in (16, 16, 32, 32):
        assert make_impl(heads)._resolve_preprocess_type(torch.bfloat16) == sfa.PreprocessType.MLAPO
    assert len(records) == 2
    assert [f"path=Direct heads={heads}" in record.getMessage() for heads, record in zip((16, 32), records)] == [
        True,
        True,
    ]


@pytest.mark.parametrize("direct", [False, True])
def test_patch_trace_disabled_keeps_routing_and_emits_no_diagnostic(initialization, monkeypatch, direct):
    _, records = initialization
    monkeypatch.setenv("VLLM_PATCH_TRACE", "0")
    impl = make_impl()
    impl.enable_mlapo = direct
    expected = sfa.PreprocessType.MLAPO if direct else sfa.PreprocessType.NATIVE
    assert impl._resolve_preprocess_type(torch.bfloat16) == expected
    assert records == []


@pytest.mark.parametrize("invalid", [torch.ones(1, dtype=torch.float32), torch.ones(2, dtype=torch.bfloat16)])
def test_scale_refresh_rejects_invalid_source_before_changing_either_backing(initialization, invalid):
    impl = make_impl()
    assert impl._resolve_preprocess_type(torch.bfloat16) == sfa.PreprocessType.MLAPO
    before = [getattr(impl, f"_mlapo_quant_scale_storage{i}").clone() for i in (0, 1)]
    impl.quant_scale0 = torch.tensor([0.75], dtype=torch.bfloat16)
    impl.quant_scale1 = invalid
    with pytest.raises(ValueError, match="scalar BF16 scales"):
        impl._refresh_mlapo_quant_scales()
    for i, value in enumerate(before):
        assert torch.equal(getattr(impl, f"_mlapo_quant_scale_storage{i}"), value)


@pytest.mark.parametrize(
    "state, prefills, decodes, tokens, expected",
    [
        ("ChunkedPrefill", 0, 1, 6, "MLAPO"),
        ("ChunkedPrefill", 1, 1, 6, "Native"),
        ("DecodeOnly", 1, 1, 6, "Native"),
        ("DecodeOnly", 0, 0, 6, "Native"),
        ("DecodeOnly", None, None, 6, "MLAPO"),
        ("SpecDecoding", None, None, 6, "MLAPO"),
        ("ChunkedPrefill", None, None, 6, "Native"),
        ("DecodeOnly", 1, None, 6, "Native"),
        ("SpecDecoding", 0, None, 6, "MLAPO"),
        ("ChunkedPrefill", 0, None, 6, "Native"),
        ("DecodeOnly", None, 1, 6, "MLAPO"),
        ("ChunkedPrefill", 0, 1, 1025, "Native"),
    ],
)
def test_offload_mlapo_forward_routes_v1_and_v2_metadata(monkeypatch, state, prefills, decodes, tokens, expected):
    # Execute the real forward until it selects Native or the fused preprocess.
    # Stop at that boundary; no projection, NPU op, FSA or model output is mocked
    # into a numerical consistency claim.
    class SelectedPreprocess(Exception):
        pass

    metadata = SimpleNamespace(
        cos=None, sin=None, num_input_tokens=tokens, attn_state=getattr(AscendAttentionState, state)
    )
    if prefills is not None:
        metadata.num_prefills = prefills
    if decodes is not None:
        metadata.num_decodes = decodes
    impl = SimpleNamespace(
        enable_sparse_sfa_turboquant=False,
        qk_rope_head_dim=64,
        g_proj=None,
        preprocess_type=sfa.PreprocessType.MLAPO,
        runtime_has_indexer=False,
        layerwise_kv_cache_hook=None,
        _compose_sfa_kv_cache=lambda value: value,
        _get_sfa_kv_slot_mapping=lambda _metadata: torch.zeros(tokens, dtype=torch.int32),
        _get_indexer_attn_metadata=lambda: None,
        _get_parallel_forward_context=lambda *_args: SimpleNamespace(
            actual_seq_lengths_query=None, actual_seq_lengths_key=None
        ),
        _prepare_native_hidden_states=Mock(side_effect=SelectedPreprocess("Native")),
        _sfa_preprocess_mlapo=Mock(side_effect=SelectedPreprocess("MLAPO")),
    )
    monkeypatch.setattr(
        sfa_base, "get_ascend_config", lambda: SimpleNamespace(sparse_kv_offload_config=SimpleNamespace(enabled=True))
    )
    monkeypatch.setattr(sfa_base, "wait_for_kv_layer_from_connector", Mock())
    hidden_states = torch.empty((tokens, 4))
    with pytest.raises(SelectedPreprocess, match=expected):
        sfa_base.AscendSFAImpl.forward(impl, "layer.0", hidden_states, (object(),), metadata, output=hidden_states)


@pytest.mark.parametrize(
    "tp_rank, fused, mode",
    [(0, True, "eager"), (1, True, "capture"), (0, False, "capture"), (0, True, "runtime_full")],
)
def test_direct_submission_uses_manager_rank_and_preserves_writeback_contract(
    initialization, monkeypatch, tp_rank, fused, mode
):
    impl = make_impl()
    assert impl._resolve_preprocess_type(torch.bfloat16) == sfa.PreprocessType.MLAPO
    # Main's attention implementation has no tp_rank field. TRACE must use the
    # manager's actual rank, and the mock device boundary never runs arithmetic.
    assert not hasattr(impl, "tp_rank")
    impl.layer_name, impl._current_layer_name = None, "layer.0"
    impl._in_graph_runtime = Mock(return_value=mode != "eager")
    for name in (
        "wd_qkv",
        "deq_scale_qkv",
        "gamma1",
        "beta1",
        "wu_q",
        "qb_deq_scl",
        "quant_offset0",
        "quant_bias_qkv",
        "quant_offset1",
        "qb_qt_bias",
        "ctkv_scale",
        "q_nope_scale",
    ):
        setattr(impl, name, object())
    manager = SimpleNamespace(
        tp_rank=tp_rank,
        use_fused_overlap=True,
        _get_offload_layer_id=lambda _layer: 0,
        k_caches_cpu=[torch.empty((8, 512), dtype=torch.bfloat16)],
        v_caches_cpu=[torch.empty((8, 64), dtype=torch.bfloat16)],
        offload_new_kv=Mock(),
        wait_for_current_kv_writeback=Mock(),
    )
    monkeypatch.setattr(sfa, "get_sparse_kv_offload_manager", lambda: manager)
    monkeypatch.setattr(sfa, "_EXTRA_CTX", SimpleNamespace(capturing=mode == "capture"))
    monkeypatch.setenv("VLLM_ASCEND_MLAPO_FUSED_CPU_WRITEBACK", str(int(fused)))
    operation = Mock()
    monkeypatch.setattr(torch.ops._C_ascend, "mla_preprocess_without_kv_cache_write", operation, raising=False)
    hidden = torch.empty((6, 6144), dtype=torch.bfloat16)
    rope = torch.empty((6, 1, 1, 64), dtype=torch.bfloat16)
    slots = torch.arange(6, dtype=torch.int32)
    returned = impl._sfa_preprocess_mlapo(hidden, (), rope, rope, slots, num_input_tokens=6)
    assert returned[0] is hidden
    assert [tuple(value.shape) for value in returned[1:]] == [(6, 16, 512), (6, 16, 64), (6, 2048)]
    submitted = operation.call_args.kwargs
    assert submitted["current_k"].data_ptr() == impl._mlapo_current_k.data_ptr()
    assert submitted["current_rope"].data_ptr() == impl._mlapo_current_rope.data_ptr()
    assert ("writeback_slots" in submitted) is (fused and tp_rank == 0)
    if fused and tp_rank == 0:
        assert submitted["writeback_slots"] is slots
        assert submitted["writeback_k_base"] == manager.k_caches_cpu[0].data_ptr()
        assert submitted["writeback_rope_base"] == manager.v_caches_cpu[0].data_ptr()
        assert submitted["writeback_capacity"] == 8
    published = manager.offload_new_kv.call_args.kwargs
    assert published["direct_kv_output"] is True
    assert published["writeback_in_producer"] is fused
    assert published["capturing"] is (mode != "eager")
    assert manager.wait_for_current_kv_writeback.call_count == int(mode == "runtime_full")
