from types import SimpleNamespace as Ns
from unittest.mock import MagicMock, sentinel

import pytest
import torch

from tests.ut.attention.test_kimi_c8_runtime import runtime
from vllm_ascend.attention import kimi_c8_decode as decode
from vllm_ascend.attention import kimi_c8_prolog as prolog
from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import AscendW8A8MXFP8DynamicLinearMethod as MXFP8


@pytest.fixture(autouse=True)
def hardware_cast(monkeypatch):
    monkeypatch.setattr(prolog, "_to_nz", lambda value: value, raising=False)


def weights(heads, device="meta"):
    result = {"quant_type": MXFP8, "weight_uk": torch.empty(heads, 128, 512, dtype=torch.bfloat16, device=device)}
    for name, output, hidden in (("dq", 1536, 7168), ("uq_qr", heads * 192, 1536), ("dkv_kr", 576, 7168)):
        result["weight_" + name] = torch.empty(hidden, output, dtype=torch.float8_e4m3fn, device=device)
        result["dequant_scale_w_" + name] = torch.full((output, hidden // 32), 127, dtype=torch.uint8, device=device)
    return result


@pytest.mark.parametrize("heads", [12, 96])
def test_native_nz_weights_reuse_storage_and_scale_exponent_bytes(heads):
    source = weights(heads, "cpu")
    source = {
        key: torch.nn.Parameter(value, requires_grad=False) if isinstance(value, torch.Tensor) else value
        for key, value in source.items()
    }
    source["dequant_scale_w_dq"][33, 17] = 129
    prepared = prolog.prepare_quant_mla_prolog_weights(source)
    assert all(type(value) is torch.Tensor for value in prepared.values())
    assert prepared["wqa"].shape == (7168, 1536) and prepared["wqb"].shape == (1536, heads * 192)
    assert prepared["wqa"].data_ptr() == source["weight_dq"].data_ptr()
    assert prepared["descale_wqa"].shape == (112, 1536, 2)
    assert prepared["descale_wqa"].view(torch.uint8)[8, 33, 1] == 129
    assert prepared["wkb"].data_ptr() == source["weight_uk"].data_ptr()
    source["dequant_scale_w_dq"] = torch.empty(1536, 224, dtype=torch.float32)
    with pytest.raises(ValueError, match="E8M0 bytes"):
        prolog.prepare_quant_mla_prolog_weights(source)


def api(monkeypatch, op):
    modes = dict(
        w_qr_quant_mode=sentinel.mx,
        qkv_quant_mode=sentinel.c8,
        cache_mode=sentinel.nz,
        rope_mode=sentinel.no_rope,
        return_qr=False,
    )
    check = MagicMock()
    module = Ns(
        quant_mla_prolog=op,
        quant_mla_prolog_precheck=check,
        WQrQuantMode=Ns(A8W8_MXFP8=sentinel.mx),
        QkvQuantMode=Ns(A8C8_Q_FP8_E4M3_PER_TOKEN_HEAD_KV_FP8_E4M3_PER_TENSOR=sentinel.c8),
        CacheMode=Ns(PA_NZ=sentinel.nz),
        RopeMode=Ns(NO_ROPE=sentinel.no_rope),
    )
    monkeypatch.setattr(
        prolog, "import_module", lambda name: module if name == "ops.quant_mla_prolog" else pytest.fail(name)
    )
    return modes, check


@pytest.mark.parametrize("heads,graph", [(12, False), (12, True), (96, False), (96, True)])
def test_new_api_reciprocal_scale_preflight_cache_alias_and_reload(monkeypatch, heads, graph):
    import torch_npu

    scheme = Ns(quant_method=object.__new__(MXFP8))
    impl, _, _ = runtime(graph, 8 if heads == 12 else 1)
    impl.num_heads, impl.num_kv_heads, impl.q_lora_rank, impl.kv_lora_rank = heads, 1, 1536, 512
    impl.qk_nope_head_dim, impl.qk_rope_head_dim = 128, 64
    impl.fused_qkv_a_proj, impl.q_proj = Ns(quant_method=scheme, input_size=7168), Ns(quant_method=scheme)
    impl.q_a_layernorm = Ns(weight=torch.ones(1536), variance_epsilon=1e-6)
    impl.kv_a_layernorm = Ns(weight=torch.ones(512), variance_epsilon=1e-6)
    impl._prepare_prolog_v3_weights = lambda: weights(heads)
    quantized, scale = torch.empty(2, 7168, dtype=torch.float8_e4m3fn), torch.full((2, 224), 127, dtype=torch.uint8)
    monkeypatch.setattr(torch_npu, "npu_dynamic_mx_quant", lambda *a, **kw: (quantized, scale))
    cache = torch.full((2, 1, 18, 1664, 32), 17, dtype=torch.uint8)
    op = MagicMock(
        return_value=(
            torch.empty(2, heads, 576, dtype=torch.float8_e4m3fn),
            torch.ones(2, heads, 1),
            torch.empty(0),
            torch.empty(0),
        )
    )
    from vllm_ascend.device import kimi_c8_cache_write as writer

    def prolog_write(**kwargs):
        workspace = kwargs["kv_cache"].view(torch.uint8).view(-1, 1, 18, 128, 32)
        workspace[0, 0, :, 0].fill_(7)
        workspace[0, 0, :, 1].fill_(11)
        return op.return_value

    def scatter(source, target, slots):
        target[1, 0, :, 0].copy_(source[0, 0, :, 1])

    op.side_effect = prolog_write
    monkeypatch.setattr(writer, "copy_prolog_kv_to_cache", scatter)
    modes, check = api(monkeypatch, op)
    assert decode.prepare_kimi_c8_prolog_runtime(impl) and impl._prolog_v3_weights is None
    norm_ptr = impl._kimi_c8_prolog_norm_weights[0].data_ptr()

    def forbidden(*args, **kwargs):
        raise AssertionError("Graph cannot synchronize tensors to host")

    with monkeypatch.context() as guard:
        if graph:
            guard.setattr(torch.Tensor, "item", forbidden)
            guard.setattr(torch.Tensor, "tolist", forbidden)
        q, descale = decode.project_and_write_kimi_c8_prolog(
            impl, torch.ones(2, 7168), cache, torch.tensor([-1, 1664]), num_decode_tokens=2, is_graph=graph
        )
    assert q.shape == (2, heads, 576) and descale.shape == (2, heads)
    args = op.call_args.kwargs
    assert args["x"] is quantized and args["descale_x"].shape == (2, 224)
    assert args["kv_cache"].shape[1] == 128 and args["kv_cache"].data_ptr() != cache.data_ptr()
    assert cache.shape[3] == 1664 and cache[0].eq(17).all() and cache[1, 0, :, 0].eq(11).all()
    assert args["wqb"].shape == (1536, heads * 192)
    assert {key: args[key] for key in modes} == modes
    assert args["quant_scale_kv"].item() == 4.0 and check.call_count == 1
    pointer = args["quant_scale_kv"].data_ptr()
    impl.fak_descale_float.fill_(0.5)
    impl.q_a_layernorm.weight.fill_(2)
    assert decode.prepare_kimi_c8_prolog_runtime(impl)
    assert impl._kimi_c8_prolog_norm_weights[0].data_ptr() == norm_ptr
    assert impl._kimi_c8_prolog_norm_weights[0].eq(2).all()
    assert impl._kimi_c8_prolog_weights["quant_scale_kv"].data_ptr() == pointer
    assert check.call_args.kwargs["quant_scale_kv"].item() == 2.0 and check.call_count == 2
