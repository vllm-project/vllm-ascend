# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU composition checks; the emulated provider does not validate NPU kernels."""

import ast
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
MLA = "vllm_ascend/attention/mla_v1.py"


class NativeLinear:
    pass


class MXFP8Linear:
    pass


class OtherQuantLinear:
    pass


def load(name, scope):
    nodes = ast.parse((ROOT / MLA).read_text(encoding="utf-8")).body
    constants = [
        node
        for node in nodes
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "PROLOG_V3_SUPPORTED_INPUT_SIZES" for target in node.targets
        )
    ]
    owner = next(node for node in nodes if isinstance(node, ast.ClassDef) and node.name == "AscendMLAImpl")
    node = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == name)
    node.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, *constants, node], type_ignores=[])), MLA, "exec"),
        scope,
    )
    return scope[name]


def bind(impl, scope, *names):
    for name in names:
        setattr(impl, name, MethodType(load(name, scope), impl))


def scope():
    return {
        "torch": torch,
        "MLAPO_MAX_SUPPORTED_TOKENS": 1024,
        "MLAPO_CACHE_BLOCK_ALIGNMENT": 16,
        "MLAPO_MAX_CACHE_BLOCK_SIZE": 1024,
        "FLASHMLA_BLOCK_SIZE": 128,
        "UnquantizedLinearMethod": NativeLinear,
        "AscendW8A8MXFP8DynamicLinearMethod": MXFP8Linear,
        "cdiv": lambda x, y: (x + y - 1) // y,
        "DecodeMLAPreprocessResult": lambda q, pe, k, kr, **kw: SimpleNamespace(ql_nope=q, q_pe=pe, k=k, kr=kr, **kw),
    }


def scatter(*, key, value, key_cache, value_cache, slot_mapping):
    block_size = key_cache.shape[1]
    for row, slot in enumerate(slot_mapping.tolist()):
        if slot >= 0:
            key_cache[slot // block_size, slot % block_size].copy_(key[row])
            value_cache[slot // block_size, slot % block_size].copy_(value[row])


def fused_cache(dtype=torch.bfloat16):
    page_stride = 128 * 576 + 32
    raw = torch.full((3 * page_stride + 64,), -17.0, dtype=dtype)
    return raw.as_strided((3, 128, 1, 576), (page_stride, 576, 576, 1), 32), raw


def workspace(capacity=128, block_size=128):
    blocks = (capacity + block_size - 1) // block_size
    return (
        torch.empty(blocks, block_size, 1, 512, dtype=torch.bfloat16),
        torch.empty(blocks, block_size, 1, 64, dtype=torch.bfloat16),
        torch.arange(capacity, dtype=torch.int64),
    )


def make_impl():
    qnorm = SimpleNamespace(weight=torch.ones(1536, dtype=torch.bfloat16), variance_epsilon=1e-5)
    knorm = SimpleNamespace(weight=torch.ones(512, dtype=torch.bfloat16), variance_epsilon=1e-6)
    return SimpleNamespace(
        layer_name="mla",
        use_flashmla=True,
        enable_mlapo=True,
        use_mla_rope=False,
        is_draft_model=False,
        dtype=torch.bfloat16,
        support_fp8_attention=True,
        fa_quant_layer=False,
        enable_kv_nz=False,
        pcp_enabled=False,
        is_pcp_decode_sharded=False,
        dcp_enabled=False,
        num_heads=12,
        num_heads_padded=16,
        mlapo_num_heads=16,
        num_kv_heads=1,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        qk_head_dim=192,
        q_lora_rank=1536,
        q_a_layernorm=qnorm,
        kv_a_layernorm=knorm,
        fused_qkv_a_proj=SimpleNamespace(input_size=7168, quant_method=NativeLinear()),
        q_proj=SimpleNamespace(input_size=1536, _chunk_size=0, quant_method=NativeLinear()),
        mlapo_weight_quant_mode=0,
        _mlapo_uses_native_weights=True,
        _flashmla_writer_buffers=workspace(),
        _dcp_current_kv_buffers=workspace(),
        weight_dq=object(),
        weight_uq_qr=object(),
        mlapo_W_UK_T=object(),
        weight_dkv_kr=object(),
        _decode_requires_current_kv=lambda _: False,
    )


def eligibility_impl():
    impl = make_impl()
    namespace = scope()
    impl._get_context_prolog_quant_method = load("_get_context_prolog_quant_method", namespace)
    bind(
        impl,
        namespace,
        "supports_prolog_v3_dtype",
        "supports_prolog_v3_projection",
        "_supports_flashmla_mlapo",
        "_can_use_flashmla_mlapo",
    )
    return impl


@pytest.mark.parametrize("native", [True, False])
def test_flashmla_no_rope_accepts_native_and_exact_mxfp8(native):
    impl = eligibility_impl()
    if not native:
        impl._mlapo_uses_native_weights = False
        impl.mlapo_weight_quant_mode = 3
        impl.fused_qkv_a_proj.quant_method = SimpleNamespace(quant_method=MXFP8Linear())
        impl.q_proj.quant_method = SimpleNamespace(quant_method=MXFP8Linear())
    # This path uses the unified provider; availability of the RoPE-only
    # torch_npu PROLOG_V3 writer must not control no-RoPE eligibility.
    assert impl._supports_flashmla_mlapo()


@pytest.mark.parametrize(
    "field,value",
    [
        ("use_flashmla", False),
        ("enable_mlapo", False),
        ("is_draft_model", True),
        ("use_mla_rope", True),
        ("support_fp8_attention", False),
        ("dtype", torch.float16),
        ("fa_quant_layer", True),
        ("enable_kv_nz", True),
        ("pcp_enabled", True),
        ("is_pcp_decode_sharded", True),
        ("fused_qkv_a_proj", None),
        ("q_proj", None),
        ("q_lora_rank", None),
        ("q_a_layernorm", None),
        ("kv_a_layernorm", None),
        ("mlapo_weight_quant_mode", 2),
    ],
)
def test_flashmla_mlapo_static_rejections(field, value):
    impl = eligibility_impl()
    setattr(impl, field, value)
    assert not impl._supports_flashmla_mlapo()


@pytest.mark.parametrize("unsupported_projection", ["wide_input", "chunked_q"])
def test_flashmla_mlapo_rejects_unsupported_projection(unsupported_projection):
    impl = eligibility_impl()
    if unsupported_projection == "wide_input":
        impl.fused_qkv_a_proj.input_size = 14336
    else:
        impl.q_proj._chunk_size = 1
    assert not impl._supports_flashmla_mlapo()


@pytest.mark.parametrize("kind", ["mixed", "subclass", "other", "native_mode3", "mxfp8_mode0"])
def test_flashmla_mlapo_rejects_unsupported_or_mixed_weight_contracts(kind):
    class MXFP8Subclass(MXFP8Linear):
        pass

    impl = eligibility_impl()
    impl._mlapo_uses_native_weights = False
    impl.mlapo_weight_quant_mode = 3
    impl.fused_qkv_a_proj.quant_method = SimpleNamespace(quant_method=MXFP8Linear())
    impl.q_proj.quant_method = SimpleNamespace(quant_method=MXFP8Linear())
    if kind == "mixed":
        impl.q_proj.quant_method = NativeLinear()
    elif kind == "subclass":
        impl.q_proj.quant_method.quant_method = MXFP8Subclass()
    elif kind == "other":
        impl.q_proj.quant_method.quant_method = OtherQuantLinear()
    elif kind == "native_mode3":
        impl._mlapo_uses_native_weights = True
        impl.fused_qkv_a_proj.quant_method = NativeLinear()
        impl.q_proj.quant_method = NativeLinear()
    else:
        impl.mlapo_weight_quant_mode = 0
    assert not impl._supports_flashmla_mlapo()


@pytest.mark.parametrize(
    "tokens,expected", [(0, False), (1, True), (9, True), (129, True), (1024, True), (1025, False)]
)
@pytest.mark.parametrize("dcp", [False, True])
def test_flashmla_mlapo_runtime_decode_limits(tokens, expected, dcp):
    impl = eligibility_impl()
    impl.dcp_enabled = dcp
    impl._flashmla_writer_buffers = workspace(1025)
    impl._dcp_current_kv_buffers = workspace(1025)
    history, _ = fused_cache()
    metadata = SimpleNamespace(num_decode_tokens=tokens, num_prefills=0, external_flashmla=object())
    assert impl._can_use_flashmla_mlapo((history[..., :512], history[..., 512:]), metadata) is expected


@pytest.mark.parametrize("dcp", [False, True])
@pytest.mark.parametrize("reason", ["prefill", "external", "disabled", "missing_workspace", "workspace_capacity"])
def test_flashmla_mlapo_runtime_requires_pure_decode_and_preallocated_capacity(dcp, reason):
    impl = eligibility_impl()
    impl.dcp_enabled = dcp
    history, _ = fused_cache()
    metadata = SimpleNamespace(num_decode_tokens=4, num_prefills=0, external_flashmla=object())
    if reason == "prefill":
        metadata.num_prefills = 1
    elif reason == "external":
        metadata.external_flashmla = None
    elif reason == "disabled":
        impl.enable_mlapo = False
    else:
        attr = "_dcp_current_kv_buffers" if dcp else "_flashmla_writer_buffers"
        setattr(impl, attr, None if reason == "missing_workspace" else workspace(3))
    assert not impl._can_use_flashmla_mlapo((history[..., :512], history[..., 512:]), metadata)


@pytest.mark.parametrize("reason", ["dtype", "rank", "block_size", "heads", "nope_width", "raw_width"])
def test_flashmla_mlapo_runtime_checks_cache_contract(reason):
    impl = eligibility_impl()
    k = torch.empty(3, 128, 1, 512, dtype=torch.bfloat16)
    kr = torch.empty(3, 128, 1, 64, dtype=torch.bfloat16)
    if reason == "dtype":
        kr = kr.float()
    elif reason == "rank":
        kr = kr.flatten(1, 2)
    elif reason == "block_size":
        k = k[:, :64]
    elif reason == "heads":
        kr = kr.expand(3, 128, 2, 64)
    elif reason == "nope_width":
        k = k[..., :511]
    else:
        kr = kr[..., :63]
    metadata = SimpleNamespace(num_decode_tokens=4, num_prefills=0, external_flashmla=object())
    assert not impl._can_use_flashmla_mlapo((k, kr), metadata)


@pytest.mark.parametrize("dcp", [False, True])
@pytest.mark.parametrize(
    "block_size,expected",
    [
        (0, False),
        (15, False),
        (16, True),
        (128, True),
        (384, True),
        (768, True),
        (1024, True),
        (1032, False),
        (1536, False),
    ],
)
def test_flashmla_mlapo_runtime_checks_provider_workspace_block_contract(dcp, block_size, expected):
    impl = eligibility_impl()
    impl.dcp_enabled = dcp
    if block_size:
        buffers = workspace(4, block_size=block_size)
    else:
        buffers = (torch.empty(1, 0, 1, 512), torch.empty(1, 0, 1, 64), torch.arange(4))
    setattr(impl, "_dcp_current_kv_buffers" if dcp else "_flashmla_writer_buffers", buffers)
    history, _ = fused_cache()
    metadata = SimpleNamespace(num_decode_tokens=4, num_prefills=0, external_flashmla=object())
    assert impl._can_use_flashmla_mlapo((history[..., :512], history[..., 512:]), metadata) is expected


def test_no_rope_workspace_caps_capacity_uses_kernel_blocks_and_keeps_addresses():
    impl = make_impl()
    impl._flashmla_writer_buffers = None
    impl.W_UV = torch.empty(1)
    impl.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4096),
        cache_config=SimpleNamespace(block_size=384),
    )
    bind(impl, scope(), "_prepare_flashmla_writer_buffers", "_prolog_v3_writer_workspace")
    impl._prepare_flashmla_writer_buffers()
    buffers = impl._flashmla_writer_buffers
    assert buffers[0].shape == (8, 128, 1, 512)
    assert buffers[1].shape == (8, 128, 1, 64)
    torch.testing.assert_close(buffers[2], torch.arange(1024))
    addresses = tuple(t.data_ptr() for t in buffers)
    impl._prepare_flashmla_writer_buffers()
    assert impl._flashmla_writer_buffers is buffers
    history, _ = fused_cache()
    views = history[..., :512], history[..., 512:]
    for tokens in (9, 129, 1024):
        writer, indices, staged = impl._prolog_v3_writer_workspace(tokens, views, torch.arange(tokens))
        assert staged and writer[0].shape[0] == (tokens + 127) // 128
        assert (writer[0].data_ptr(), writer[1].data_ptr(), indices.data_ptr()) == addresses
    with pytest.raises(RuntimeError, match="scheduled token rows"):
        impl._prolog_v3_writer_workspace(1025, views, torch.arange(1025))


@pytest.mark.parametrize(
    "case,expected_mlapo",
    [
        ("native", True),
        ("mxfp8", True),
        ("dcp", True),
        ("dcp_current", True),
        ("129_tokens", True),
        ("too_many_tokens", False),
        ("mixed_prefill", False),
        ("draft", False),
        ("fp16_cache", False),
        ("no_workspace", False),
        ("disabled", False),
    ],
)
def test_forward_routes_token_fused_decode_with_real_eligibility_helper(case, expected_mlapo):
    impl = eligibility_impl()
    tokens = 1025 if case == "too_many_tokens" else 129 if case == "129_tokens" else 9
    impl._flashmla_writer_buffers = workspace(1024)
    impl._dcp_current_kv_buffers = workspace(1024)
    if case == "mxfp8":
        impl._mlapo_uses_native_weights = False
        impl.mlapo_weight_quant_mode = 3
        impl.fused_qkv_a_proj.quant_method = SimpleNamespace(quant_method=MXFP8Linear())
        impl.q_proj.quant_method = SimpleNamespace(quant_method=MXFP8Linear())
    elif case in ("dcp", "dcp_current"):
        impl.dcp_enabled = True
        impl._decode_requires_current_kv = lambda _: case == "dcp_current"
    elif case == "draft":
        impl.is_draft_model = True
    elif case == "disabled":
        impl.enable_mlapo = False
    elif case == "no_workspace":
        impl._flashmla_writer_buffers = None
    history, _ = fused_cache(torch.float16 if case == "fp16_cache" else torch.bfloat16)
    metadata = SimpleNamespace(
        num_actual_tokens=tokens,
        num_decodes=1,
        num_decode_tokens=tokens,
        num_prefills=int(case == "mixed_prefill"),
        external_flashmla=object(),
    )
    result = object()
    impl.v_head_dim = 128
    impl.use_output_gate = False
    impl.layerwise_kv_cache_hook = None
    impl.get_num_actual_tokens = lambda m: m.num_actual_tokens
    impl._prefill_requires_current_kv = lambda _: False
    impl._cache_requires_prolog_v3_writer = lambda _: False
    impl._uses_strided_kv_adapter = lambda: False
    impl.mla_preprocess_only_decode = Mock(return_value=(result, None))
    impl._mla_preprocess = Mock(return_value=(result, None))
    impl._forward_external_flashmla = Mock(return_value=torch.ones(tokens, 12 * 128))
    impl._forward_decode = Mock(side_effect=AssertionError("FlashMLA must retain its external attention backend"))
    impl.o_proj = lambda x, **_: (x,)
    namespace = scope() | {
        "_EXTRA_CTX": SimpleNamespace(num_tokens=tokens),
        "maybe_save_kv_layer_to_connector": Mock(),
    }
    bind(impl, namespace, "forward")
    output = torch.empty(tokens, 12 * 128)
    assert impl.forward("mla", torch.zeros(tokens, 16), history, metadata, output) is output
    torch.testing.assert_close(output, torch.ones_like(output))
    selected = impl.mla_preprocess_only_decode if expected_mlapo else impl._mla_preprocess
    fallback = impl._mla_preprocess if expected_mlapo else impl.mla_preprocess_only_decode
    selected.assert_called_once()
    fallback.assert_not_called()
    # forward supplies zero-copy split views while retaining the original
    # single Tensor for FlashMLA's attention call.
    views = selected.call_args.args[-2]
    assert len(views) == 2
    assert all(v.untyped_storage().data_ptr() == history.untyped_storage().data_ptr() for v in views)
    assert impl._forward_external_flashmla.call_args.args == (result, history, metadata)


@pytest.mark.parametrize("dcp,current", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("tokens", [4, 129])
def test_flashmla_no_rope_mlapo_stages_rows_and_preserves_formal_cache(monkeypatch, dcp, current, tokens):
    impl = make_impl()
    impl.dcp_enabled = dcp
    impl._decode_requires_current_kv = lambda _: current
    impl._flashmla_writer_buffers = workspace(256)
    # The existing DCP buffers use manager blocks; formal FlashMLA uses
    # kernel blocks of 128. At T=129 the correct slice is one block of 384.
    impl._dcp_current_kv_buffers = workspace(768, block_size=384)
    history, raw = fused_cache()
    before = raw.clone()
    slots = torch.full((tokens,), -1, dtype=torch.int32)
    slots[:2] = torch.tensor([0, 128], dtype=torch.int32)
    metadata = SimpleNamespace(
        num_decode_tokens=tokens,
        num_prefills=0,
        # Deliberately conflicts with the FlashMLA live/padded slots.
        slot_mapping=torch.arange(tokens) + 1,
        external_flashmla=SimpleNamespace(slots=slots, current=object() if current else None),
        decode=SimpleNamespace(cos=None, sin=None),
    )
    captured = {}

    def prolog(**kwargs):
        captured.update(kwargs)
        assert kwargs["rope_sin"] is None and kwargs["rope_cos"] is None
        assert kwargs["rmsnorm_epsilon_cq"] == impl.q_a_layernorm.variance_epsilon
        assert kwargs["rmsnorm_epsilon_ckv"] == impl.kv_a_layernorm.variance_epsilon
        assert kwargs["cache_mode"] == "PA_BSND"
        assert kwargs["weight_quant_mode"] == 0
        assert kwargs["query_quant_mode"] == kwargs["kv_cache_quant_mode"] == 0
        torch.testing.assert_close(kwargs["cache_index"], torch.arange(tokens))
        for cache, width in ((kwargs["kv_cache"], 512), (kwargs["kr_cache"], 64)):
            assert cache.stride(1) == width and cache.is_contiguous()
            assert cache.shape[:2] == ((1, 384) if dcp else ((tokens + 127) // 128, 128))
            assert cache.untyped_storage().data_ptr() != raw.untyped_storage().data_ptr()
        keys = torch.arange(tokens, dtype=torch.bfloat16)[:, None, None].expand(tokens, 1, 512) + 2
        scatter(
            key=keys,
            value=keys[..., :64] + 10,
            key_cache=kwargs["kv_cache"],
            value_cache=kwargs["kr_cache"],
            slot_mapping=kwargs["cache_index"],
        )
        q, pe = torch.ones(tokens, 16, 512), torch.full((tokens, 16, 64), 2.0)
        q[:, 12:] = torch.nan
        pe[:, 12:] = torch.inf
        return q, pe, None, None, None

    provider = Mock(side_effect=prolog)
    monkeypatch.setitem(sys.modules, "cann_ops_transformer", SimpleNamespace(mla_prolog=provider))
    scatter_op = Mock(side_effect=scatter)
    namespace = scope() | {"DeviceOperator": SimpleNamespace(reshape_and_cache=scatter_op)}
    bind(impl, namespace, "_prolog_v3_writer_workspace", "mla_preprocess_only_decode")

    def reorg(q, pe):
        # Padded heads must be removed before DCP can gather local heads.
        assert q.shape == (tokens, 12, 512) and pe.shape == (tokens, 12, 64)
        return (q.repeat(1, 2, 1), pe.repeat(1, 2, 1)) if dcp else (q, pe)

    impl.reorg_decode_q = Mock(side_effect=reorg)
    buffers = impl._dcp_current_kv_buffers if dcp else impl._flashmla_writer_buffers
    addresses = tuple(t.data_ptr() for t in buffers)
    hidden = torch.zeros(tokens + 2, 16, dtype=torch.bfloat16)
    views = history[..., :512], history[..., 512:]
    result, prefill = impl.mla_preprocess_only_decode(hidden, views, metadata)
    assert prefill is None
    provider.assert_called_once()
    assert captured["token_x"].shape == (tokens, 16)
    assert captured["kv_cache"].data_ptr() == addresses[0]
    assert captured["kr_cache"].data_ptr() == addresses[1]
    assert captured["cache_index"].data_ptr() == addresses[2]
    assert tuple(t.data_ptr() for t in buffers) == addresses
    scatter_op.assert_called_once()
    assert scatter_op.call_args.kwargs["slot_mapping"].data_ptr() == slots.data_ptr()
    for result_cache in (result.k, result.kr):
        assert result_cache.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
    torch.testing.assert_close(result.ql_nope, torch.ones(tokens, 24 if dcp else 12, 512))
    torch.testing.assert_close(result.q_pe, torch.full((tokens, 24 if dcp else 12, 64), 2.0))
    if current:
        torch.testing.assert_close(result.current_k_nope[:, 0, 0], torch.arange(tokens, dtype=torch.bfloat16) + 2)
        torch.testing.assert_close(result.current_k_pe[:, 0, 0], torch.arange(tokens, dtype=torch.bfloat16) + 12)
    else:
        assert result.current_k_nope is None and result.current_k_pe is None
    torch.testing.assert_close(history[:2, 0, 0, 0], torch.tensor([2.0, 3.0], dtype=torch.bfloat16))
    torch.testing.assert_close(history[:2, 0, 0, 512], torch.tensor([12.0, 13.0], dtype=torch.bfloat16))
    changed = torch.zeros_like(raw, dtype=torch.bool)
    for block in (0, 1):
        offset = history.storage_offset() + block * history.stride(0)
        changed[offset : offset + 576] = True
    torch.testing.assert_close(raw[~changed], before[~changed])


def test_legacy_component_major_mlapo_keeps_direct_history_write(monkeypatch):
    impl = make_impl()
    impl.use_flashmla = False
    impl.mlapo_num_heads = impl.num_heads
    raw = torch.full((2 * 128 * 576,), -17.0, dtype=torch.bfloat16)
    k = raw.as_strided((2, 128, 1, 512), (128 * 576, 512, 512, 1))
    kr = raw.as_strided((2, 128, 1, 64), (128 * 576, 64, 64, 1), 128 * 512)
    slots = torch.tensor([0, 128], dtype=torch.int32)
    metadata = SimpleNamespace(num_decode_tokens=2, slot_mapping=slots, external_flashmla=None)

    def prolog(**kwargs):
        assert kwargs["kv_cache"] is k and kwargs["kr_cache"] is kr
        torch.testing.assert_close(kwargs["cache_index"], slots.to(torch.int64))
        return torch.ones(2, 12, 512), torch.ones(2, 12, 64), None, None, None

    provider = Mock(side_effect=prolog)
    monkeypatch.setitem(sys.modules, "cann_ops_transformer", SimpleNamespace(mla_prolog=provider))
    scatter_op = Mock(side_effect=AssertionError("legacy direct writes do not stage current rows"))
    namespace = scope() | {"DeviceOperator": SimpleNamespace(reshape_and_cache=scatter_op)}
    bind(impl, namespace, "mla_preprocess_only_decode")
    impl.reorg_decode_q = lambda q, pe: (q, pe)
    result, prefill = impl.mla_preprocess_only_decode(torch.zeros(2, 16), (k, kr), metadata)
    assert result.k is k and result.kr is kr and prefill is None
    provider.assert_called_once()
    scatter_op.assert_not_called()
