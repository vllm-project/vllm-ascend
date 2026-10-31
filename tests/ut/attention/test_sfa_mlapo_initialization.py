"""Original scale reuse, Direct/Native routing and fused CPU writeback."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

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
    # Parent eligibility checks retain their rank-scoped warnings. This CPU
    # test has no distributed group.
    monkeypatch.setattr(vllm_logger, "_should_log_with_scope", lambda scope: True)
    yield config


def make_impl(heads=16, hidden=6144, q_rank=2048, nope=192, kv_rank=512, rope=64, dtype=torch.bfloat16):
    # Use the actual class/methods without loading a model or touching an NPU.
    impl = sfa.AscendSFAKVOffloadImpl.__new__(sfa.AscendSFAKVOffloadImpl)
    scheme = sfa.AscendW8A8LinearMethod.__new__(sfa.AscendW8A8LinearMethod)
    impl.fused_qkv_a_proj = SimpleNamespace(
        weight=SimpleNamespace(shape=(hidden, q_rank + kv_rank + rope)),
        quant_method=SimpleNamespace(quant_method=scheme),
    )
    impl.q_proj = SimpleNamespace(
        weight=SimpleNamespace(shape=(q_rank, heads * (nope + rope))),
        quant_method=SimpleNamespace(quant_method=scheme),
    )
    impl.W_UK_T = SimpleNamespace(shape=(heads, nope, kv_rank))
    impl.use_fused_overlap = True
    impl.enable_mlapo = True
    impl.enable_dsa_cp = impl.enable_sparse_sfa_c8 = impl.enable_sparse_sfa_turboquant = False
    impl.num_heads, impl.num_kv_heads = heads, 1
    impl.q_lora_rank, impl.kv_lora_rank = q_rank, kv_rank
    impl.qk_nope_head_dim, impl.qk_rope_head_dim = nope, rope
    impl.q_a_layernorm = impl.kv_a_layernorm = object()
    impl.is_kv_consumer = True
    impl.gamma2 = torch.ones(kv_rank, dtype=dtype)
    impl.quant_scale0 = torch.tensor([0.5], dtype=dtype)
    impl.quant_scale1 = torch.tensor([0.25], dtype=dtype)
    # Only weight reformatting is replaced; the actual precondition checks,
    # selection and output allocation still execute.
    impl._process_weights_for_fused_mlapo = Mock()
    return impl


def finish_weight_loading(impl, dtype):
    impl.preprocess_type = impl._resolve_preprocess_type(dtype)
    # The fixture already supplies the parent's normalized weights. Execute
    # the actual subclass post-load allocation after preprocessing selection.
    with patch.object(sfa_base.AscendSFAImpl, "process_weights_after_loading"):
        impl.process_weights_after_loading(dtype)
    return impl.preprocess_type


@pytest.mark.parametrize(
    "consumer, token_limit, offload_enabled, released",
    [
        (True, 32, True, True),
        (True, 32, False, True),
        (True, 1024, True, True),
        (True, 1025, True, False),
        (False, 32, True, False),
        (None, 32, True, False),
    ],
)
def test_mlapo_weight_preparation_preserves_original_release_conditions(
    initialization, monkeypatch, consumer, token_limit, offload_enabled, released
):
    initialization.sparse_kv_offload_config.enabled = offload_enabled
    hidden, q_rank, kv_rank, rope, heads, nope = 64, 32, 32, 32, 2, 32
    down_width, up_width = q_rank + kv_rank + rope, heads * (nope + rope)

    def projection(input_width, output_width):
        return SimpleNamespace(
            weight=torch.ones((input_width, output_width), dtype=torch.int8),
            deq_scale=torch.ones(output_width, dtype=torch.float32),
            quant_bias=torch.zeros(output_width, dtype=torch.int32),
            input_scale=torch.tensor([0.5], dtype=torch.bfloat16),
            input_offset=torch.zeros(1, dtype=torch.int8),
        )

    impl = SimpleNamespace(
        kv_a_proj_with_mqa=None,
        fused_qkv_a_proj=projection(hidden, down_width),
        q_proj=projection(q_rank, up_width),
        q_lora_rank=q_rank,
        kv_lora_rank=kv_rank,
        qk_rope_head_dim=rope,
        qk_nope_head_dim=nope,
        num_heads=heads,
        q_a_layernorm=SimpleNamespace(
            weight=torch.ones(q_rank, dtype=torch.bfloat16),
            bias=torch.zeros(q_rank, dtype=torch.bfloat16),
        ),
        kv_a_layernorm=SimpleNamespace(weight=torch.ones(kv_rank, dtype=torch.bfloat16)),
        vllm_config=SimpleNamespace(
            kv_transfer_config=None if consumer is None else SimpleNamespace(is_kv_consumer=consumer),
            scheduler_config=SimpleNamespace(max_num_batched_tokens=token_limit),
        ),
    )
    sources = [
        (layer, name, getattr(layer, name))
        for layer in (impl.fused_qkv_a_proj, impl.q_proj)
        for name in ("weight", "deq_scale", "quant_bias")
    ]
    # Execute the production transforms on small CPU tensors. Only NPU format
    # conversion and allocator cache release are mocked; this is not an NPU test.
    format_cast = Mock(side_effect=lambda value, _format: value)
    empty_cache = Mock()
    monkeypatch.setattr(sfa_base.torch_npu, "npu_format_cast", format_cast)
    monkeypatch.setattr(torch.npu, "empty_cache", empty_cache)
    sfa_base.AscendSFAImpl._process_weights_for_fused_mlapo(impl, torch.bfloat16)
    for layer, name, original in sources:
        if released:
            assert getattr(layer, name) is None
        else:
            assert getattr(layer, name) is original
    for name, count in (
        ("wd_qkv", hidden * down_width),
        ("wu_q", q_rank * up_width),
        ("deq_scale_qkv", down_width),
        ("quant_bias_qkv", down_width),
        ("qb_deq_scl", up_width),
        ("qb_qt_bias", up_width),
    ):
        assert getattr(impl, name).numel() == count
    assert format_cast.call_count == 2
    assert empty_cache.call_count == int(released)


@pytest.mark.parametrize("heads", [2, 16, 32])
@pytest.mark.parametrize("hidden,q_rank,nope,kv_rank,rope", [(6144, 2048, 192, 512, 64), (7168, 1536, 128, 256, 128)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_initialization_preserves_original_scales(initialization, heads, hidden, q_rank, nope, kv_rank, rope, dtype):
    impl = make_impl(heads, hidden, q_rank, nope, kv_rank, rope, dtype)
    scales = (impl.quant_scale0, impl.quant_scale1)
    assert finish_weight_loading(impl, dtype) == sfa.PreprocessType.MLAPO
    assert impl.quant_scale0 is scales[0] and impl.quant_scale1 is scales[1]
    assert impl._mlapo_current_k.shape == (sfa.MLAPO_MAX_SUPPORTED_TOKENS, kv_rank)
    assert impl._mlapo_current_rope.shape == (sfa.MLAPO_MAX_SUPPORTED_TOKENS, rope)
    assert impl._mlapo_current_k.dtype == impl._mlapo_current_rope.dtype == dtype
    output_pointers = (impl._mlapo_current_k.data_ptr(), impl._mlapo_current_rope.data_ptr())
    impl.quant_scale0 = torch.tensor([0.75], dtype=dtype)
    impl.quant_scale1 = torch.tensor([0.125], dtype=dtype)
    scales = (impl.quant_scale0, impl.quant_scale1)
    assert finish_weight_loading(impl, dtype) == sfa.PreprocessType.MLAPO
    assert impl.quant_scale0 is scales[0] and impl.quant_scale1 is scales[1]
    assert output_pointers == (impl._mlapo_current_k.data_ptr(), impl._mlapo_current_rope.data_ptr())


@pytest.mark.parametrize(
    "condition",
    [
        "keep_cache",
        "copy_sfa",
        "disabled",
        "missing_down_proj",
        "down_quant",
        "q_norm",
        "kv_norm",
        "sparse_c8",
        "turboquant",
        "unsupported",
    ],
)
def test_unsupported_configuration_falls_back_to_native(initialization, condition):
    config = initialization
    impl = make_impl()
    dtype = torch.bfloat16
    if condition == "keep_cache":
        config.sparse_kv_offload_config.keep_device_kv_cache = True
    elif condition == "copy_sfa":
        impl.use_fused_overlap = False
    elif condition == "disabled":
        impl.enable_mlapo = False
    elif condition == "missing_down_proj":
        impl.fused_qkv_a_proj = None
    elif condition == "down_quant":
        impl.fused_qkv_a_proj.quant_method.quant_method = object()
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


@pytest.mark.parametrize("supported", [False, True])
def test_initialization_uses_original_eligibility_and_weight_preparation(initialization, supported):
    impl = make_impl()
    if not supported:
        impl.q_a_layernorm = None
    impl._try_enable_type = Mock(wraps=impl._try_enable_type)
    impl._get_fused_type_unsupported_reasons = Mock(wraps=impl._get_fused_type_unsupported_reasons)

    expected = sfa.PreprocessType.MLAPO if supported else sfa.PreprocessType.NATIVE
    assert impl._resolve_preprocess_type(torch.bfloat16) == expected
    impl._try_enable_type.assert_called_once_with(sfa.PreprocessType.MLAPO, torch.bfloat16)
    impl._get_fused_type_unsupported_reasons.assert_called_once_with(sfa.PreprocessType.MLAPO)
    if supported:
        impl._process_weights_for_fused_mlapo.assert_called_once_with(torch.bfloat16)
    else:
        impl._process_weights_for_fused_mlapo.assert_not_called()


@pytest.mark.parametrize("direct", [False, True])
def test_native_warning_explains_fallback_and_is_absent_for_direct(initialization, monkeypatch, direct):
    impl = make_impl()
    impl.enable_mlapo = direct
    warning = Mock()
    monkeypatch.setattr(sfa.logger, "warning_once", warning)
    expected = sfa.PreprocessType.MLAPO if direct else sfa.PreprocessType.NATIVE
    assert impl._resolve_preprocess_type(torch.bfloat16) == expected
    if direct:
        warning.assert_not_called()
    else:
        warning.assert_called_once_with(
            "Direct MLAPO was not selected for the current sparse KV offload configuration; "
            "using Native SFA preprocessing. Reasons: %s",
            "enable_mlapo=False",
        )


@pytest.mark.parametrize(
    "state, prefills, decodes, tokens, expected",
    [
        ("ChunkedPrefill", 0, 1, 6, "MLAPO"),
        ("ChunkedPrefill", 1, 1, 6, "MLAPO"),
        ("DecodeOnly", 1, 1, 6, "MLAPO"),
        ("DecodeOnly", 0, 0, 6, "MLAPO"),
        ("DecodeOnly", None, None, 6, "MLAPO"),
        ("SpecDecoding", None, None, 6, "MLAPO"),
        ("ChunkedPrefill", None, None, 6, "MLAPO"),
        ("DecodeOnly", 1, None, 6, "MLAPO"),
        ("SpecDecoding", 0, None, 6, "MLAPO"),
        ("ChunkedPrefill", 0, None, 6, "MLAPO"),
        ("DecodeOnly", None, 1, 6, "MLAPO"),
        ("ChunkedPrefill", 0, 1, 1024, "MLAPO"),
        ("ChunkedPrefill", 0, 1, 1025, "Native"),
    ],
)
def test_mlapo_forward_preserves_original_token_limit(monkeypatch, state, prefills, decodes, tokens, expected):
    # Execute the real forward until it selects Native or the fused preprocess.
    # Stop at that boundary; no projection, NPU op, FSA or model output is mocked
    # into a numerical consistency claim. This does not validate prefill offload.
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
        fused_qkv_a_proj=Mock(side_effect=SelectedPreprocess("Native")),
        _sfa_preprocess_mlapo=Mock(side_effect=SelectedPreprocess("MLAPO")),
    )
    monkeypatch.setattr(sfa_base, "wait_for_kv_layer_from_connector", Mock())
    hidden_states = torch.empty((tokens, 4))
    with pytest.raises(SelectedPreprocess, match=expected):
        sfa_base.AscendSFAImpl.forward(impl, "layer.0", hidden_states, (object(),), metadata, output=hidden_states)


@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("mode", ["eager", "capture", "runtime_full"])
@pytest.mark.parametrize("pool_error", [None, "missing", "dtype", "width"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_direct_submission_uses_manager_rank_and_preserves_writeback_contract(
    initialization, monkeypatch, tp_rank, mode, pool_error, dtype
):
    impl = make_impl(kv_rank=256, rope=128, dtype=dtype)
    assert finish_weight_loading(impl, dtype) == sfa.PreprocessType.MLAPO
    # Main's attention implementation has no tp_rank field. Writeback must use
    # the manager's actual rank; the mocked device boundary runs no arithmetic.
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
        k_caches_cpu=[torch.empty((8, impl.kv_lora_rank), dtype=dtype)],
        v_caches_cpu=[torch.empty((8, impl.qk_rope_head_dim), dtype=dtype)],
        offload_new_kv=Mock(),
        wait_for_current_kv_writeback=Mock(),
    )
    if pool_error == "missing":
        manager.k_caches_cpu[0] = None
    elif pool_error == "dtype":
        manager.k_caches_cpu[0] = torch.empty((8, impl.kv_lora_rank), dtype=torch.float32)
    elif pool_error == "width":
        manager.v_caches_cpu[0] = torch.empty((8, impl.qk_rope_head_dim * 2), dtype=dtype)
    monkeypatch.setattr(sfa, "get_sparse_kv_offload_manager", lambda: manager)
    monkeypatch.setattr(sfa, "_EXTRA_CTX", SimpleNamespace(capturing=mode == "capture"))
    operation = Mock()
    monkeypatch.setattr(torch.ops._C_ascend, "mla_preprocess_with_kv_offload", operation, raising=False)
    hidden = torch.empty((6, 6144), dtype=dtype)
    rope = torch.empty((6, 1, 1, impl.qk_rope_head_dim), dtype=dtype)
    slots = torch.arange(6, dtype=torch.int32)
    if pool_error is not None and tp_rank == 0:
        with pytest.raises(ValueError, match="MLAPO fused writeback"):
            impl._sfa_preprocess_mlapo(hidden, (), rope, rope, slots, num_input_tokens=6)
        operation.assert_not_called()
        manager.offload_new_kv.assert_not_called()
        manager.wait_for_current_kv_writeback.assert_not_called()
        return
    returned = impl._sfa_preprocess_mlapo(hidden, (), rope, rope, slots, num_input_tokens=6)
    assert returned[0] is hidden
    assert [tuple(value.shape) for value in returned[1:]] == [
        (6, 16, impl.kv_lora_rank),
        (6, 16, impl.qk_rope_head_dim),
        (6, impl.q_lora_rank),
    ]
    assert all(value.dtype == dtype for value in returned)
    assert operation.call_args.args[8] is rope
    assert operation.call_args.args[9] is rope
    submitted = operation.call_args.kwargs
    assert submitted["quant_scale0"] is impl.quant_scale0
    assert submitted["quant_scale1"] is impl.quant_scale1
    assert submitted["current_k"].data_ptr() == impl._mlapo_current_k.data_ptr()
    assert submitted["current_rope"].data_ptr() == impl._mlapo_current_rope.data_ptr()
    assert submitted["writeback_slots"] is slots
    if tp_rank == 0:
        assert submitted["writeback_k_base"] == manager.k_caches_cpu[0].data_ptr()
        assert submitted["writeback_rope_base"] == manager.v_caches_cpu[0].data_ptr()
        assert submitted["writeback_capacity"] == 8
    else:
        assert submitted.get("writeback_k_base", 0) == 0
        assert submitted.get("writeback_rope_base", 0) == 0
        assert submitted.get("writeback_capacity", 0) == 0
    published = manager.offload_new_kv.call_args.kwargs
    assert published["k"] is submitted["current_k"]
    assert published["v"] is submitted["current_rope"]
    assert published["k"].shape == (6, impl.kv_lora_rank)
    assert published["v"].shape == (6, impl.qk_rope_head_dim)
    assert published["writeback_in_producer"] is True
    assert published["capturing"] is (mode != "eager")
    assert manager.wait_for_current_kv_writeback.call_count == int(mode == "runtime_full")
