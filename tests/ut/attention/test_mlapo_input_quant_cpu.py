# SPDX-License-Identifier: Apache-2.0
"""The BF16 MLAPO path must be selected only for the validated OCP cases."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize(
    "tokens,heads,dcp,scale_alg,dtype,rope,width,expected",
    [
        (32, 12, 1, 0, torch.bfloat16, False, 7168, True),
        (64, 96, 8, 0, torch.bfloat16, False, 7168, True),
        (32, 12, 1, 1, torch.bfloat16, False, 7168, False),
        (64, 96, 8, 1, torch.bfloat16, False, 7168, False),
        (16, 12, 1, 0, torch.bfloat16, False, 7168, False),
        (32, 12, 1, 0, torch.float32, False, 7168, False),
        (32, 12, 1, 0, torch.bfloat16, True, 7168, False),
        (32, 12, 1, 0, torch.bfloat16, False, 4096, False),
    ],
)
def test_prolog_input_quantization_dispatch(tokens, heads, dcp, scale_alg, dtype, rope, width, expected):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/mla_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendMLAImpl")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "mla_preprocess_only_decode")
    calls = []

    class Captured(Exception):
        pass

    def quantize(x, **kwargs):
        calls.append(kwargs["scale_alg"])
        return x.to(torch.float8_e4m3fn), torch.empty(tokens, width // 32, dtype=torch.uint8)

    def prolog(*, fused_c8=False, **kwargs):
        assert kwargs["token_x"].dtype == (torch.bfloat16 if expected else torch.float8_e4m3fn)
        assert (kwargs["dequant_scale_x"] is None) == expected
        assert fused_c8 == (expected and dcp == 8)
        assert ("kv_descale" in kwargs) == fused_c8
        if fused_c8:
            assert kwargs["kv_descale"] is impl.fak_descale_float
        raise Captured

    scope = dict(
        torch=torch,
        torch_npu=SimpleNamespace(npu_dynamic_mx_quant=quantize, npu_mla_prolog_v3=prolog),
        get_dynamic_mx_quant_scale_alg=lambda _: scale_alg,
        envs=SimpleNamespace(VLLM_ASCEND_ENABLE_FLASH_MLA=True),
        _npu_mla_prolog_v3_no_rope=prolog,
        _npu_mla_prolog_dcp_c8=lambda **kwargs: prolog(fused_c8=True, **kwargs),
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    norm = SimpleNamespace(weight=SimpleNamespace(data=None), variance_epsilon=1e-6)
    impl = SimpleNamespace(
        kv_lora_rank=512,
        support_fp8_attention=True,
        mlapo_weight_quant_mode=3,
        flash_fused_dcp_prolog=True,
        vllm_config=None,
        dequant_scale_w_dq=torch.empty(1, dtype=torch.uint8),
        dequant_scale_w_uq_qr=torch.empty(1, dtype=torch.uint8),
        dequant_scale_w_dkv_kr=torch.empty(1, dtype=torch.uint8),
        fak_descale_reciprocal=torch.tensor([32.0]),
        fak_descale_float=torch.tensor([1 / 32.0]),
        use_mla_rope=rope,
        _mlapo_empty_rope=torch.empty(0, 64),
        fa_quant_layer=True,
        weight_dq=None,
        weight_uq_qr=None,
        mlapo_W_UK_T=None,
        weight_dkv_kr=None,
        q_a_layernorm=norm,
        kv_a_layernorm=norm,
        mlapo_num_heads=heads,
    )
    cache = (torch.empty(4, 128, 1, 512, dtype=torch.float8_e4m3fn), torch.empty(4, 128, 1, 64, dtype=torch.bfloat16))
    metadata = SimpleNamespace(
        flash=SimpleNamespace(
            dcp_size=dcp,
            query=torch.empty(tokens, heads, 576),
            slots=torch.arange(tokens),
            current_slots=torch.arange(tokens),
            current_cache=torch.empty(4, 1, 128, 576, dtype=torch.bfloat16),
        ),
        decode=SimpleNamespace(cos=torch.empty(tokens, 64), sin=torch.empty(tokens, 64)),
    )
    with pytest.raises(Captured):
        scope[method.name](impl, torch.empty(tokens, width, dtype=dtype), cache, metadata)
    assert calls == ([] if expected else [scale_alg])
