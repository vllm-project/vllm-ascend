# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.attention.mla_static import prepare_static_mlapo_weights


def _parameters():
    dims = dict(q_lora_rank=32, kv_lora_rank=16, num_heads=2, qk_nope_head_dim=16, qk_rope_head_dim=16)
    down_channels = dims["q_lora_rank"] + dims["kv_lora_rank"] + dims["qk_rope_head_dim"]
    up_channels = dims["num_heads"] * (dims["qk_nope_head_dim"] + dims["qk_rope_head_dim"])
    parameters = dict(
        qkv_weight=(torch.arange(64 * down_channels).reshape(64, down_channels) % 127).to(torch.int8),
        qkv_deq_scale=torch.arange(down_channels, dtype=torch.float32) / 4 + 0.125,
        qkv_quant_bias=torch.arange(down_channels, dtype=torch.int32) * 100_003 + 123,
        q_weight=(torch.arange(dims["q_lora_rank"] * up_channels).reshape(-1, up_channels) % 127).to(torch.int8),
        q_deq_scale=torch.arange(up_channels, dtype=torch.float32) / 8 + 0.0625,
        q_quant_bias=torch.arange(up_channels, dtype=torch.int32) * 200_003 + 321,
    )
    return parameters, dims


def _unpack(weight, rows, columns):
    # Invert the physical [N/32, M, 32] representation independently of transdata.
    blocks = weight.squeeze(0)
    padded_rows = blocks.shape[1]
    padded_columns = blocks.shape[0] * 32
    result = blocks.reshape(-1, padded_rows // 16, 16, 32)
    result = result.permute(1, 2, 0, 3).reshape(padded_rows, padded_columns)
    assert torch.count_nonzero(result[rows:]) == 0
    assert torch.count_nonzero(result[:rows, columns:]) == 0
    return result[:rows, :columns]


def test_static_mlapo_preserves_affine_parameters_and_native_sources():
    parameters, dims = _parameters()
    originals = {name: tensor.clone() for name, tensor in parameters.items()}
    result = prepare_static_mlapo_weights(**parameters, **dims)

    # Q_A precedes KV_A in the checkpoint; the kernel expects KV_A then Q_A.
    down_order = torch.tensor([*range(32, 48), *range(48, 64, 2), *range(49, 64, 2), *range(32)])
    up_order = torch.tensor(
        [*range(16), *range(16, 32, 2), *range(17, 32, 2), *range(32, 48), *range(48, 64, 2), *range(49, 64, 2)]
    )
    down_weight = _unpack(result.wd_qkv, len(down_order), parameters["qkv_weight"].shape[0])
    up_weight = _unpack(result.wu_q, len(up_order), dims["q_lora_rank"])
    torch.testing.assert_close(down_weight, parameters["qkv_weight"][:, down_order].t())
    torch.testing.assert_close(result.deq_scale_qkv, parameters["qkv_deq_scale"][down_order])
    torch.testing.assert_close(result.quant_bias_qkv, parameters["qkv_quant_bias"][down_order])
    torch.testing.assert_close(up_weight, parameters["q_weight"][:, up_order].t())
    torch.testing.assert_close(result.qb_deq_scl, parameters["q_deq_scale"][up_order])
    torch.testing.assert_close(result.qb_qt_bias, parameters["q_quant_bias"][up_order])

    inputs = torch.arange(parameters["qkv_weight"].shape[0], dtype=torch.int32).unsqueeze(0) - 7
    down_native = (inputs @ parameters["qkv_weight"].int() + parameters["qkv_quant_bias"]).float()
    down_native *= parameters["qkv_deq_scale"]
    down_fused = (inputs @ down_weight.int().t() + result.quant_bias_qkv).float() * result.deq_scale_qkv
    torch.testing.assert_close(down_fused, down_native[:, down_order], rtol=0, atol=0)
    normalized_q = torch.arange(dims["q_lora_rank"], dtype=torch.int32).unsqueeze(0) - 13
    up_native = (normalized_q @ parameters["q_weight"].int() + parameters["q_quant_bias"]).float()
    up_native *= parameters["q_deq_scale"]
    up_fused = (normalized_q @ up_weight.int().t() + result.qb_qt_bias).float() * result.qb_deq_scl
    torch.testing.assert_close(up_fused, up_native[:, up_order], rtol=0, atol=0)

    # Undoing both permutations recovers every quantized value exactly.
    torch.testing.assert_close(down_weight[down_order.argsort()].t(), originals["qkv_weight"])
    torch.testing.assert_close(up_weight[up_order.argsort()].t(), originals["q_weight"])
    for name, tensor in parameters.items():
        torch.testing.assert_close(tensor, originals[name], rtol=0, atol=0)
    assert result.wd_qkv.dtype == result.wu_q.dtype == torch.int8
    assert result.deq_scale_qkv.dtype == result.qb_deq_scl.dtype == torch.float32
    assert result.quant_bias_qkv.dtype == result.qb_qt_bias.dtype == torch.int32
    assert all(tensor.is_contiguous() for tensor in result)


@pytest.mark.parametrize(
    "name",
    ["qkv_weight", "qkv_deq_scale", "qkv_quant_bias", "q_weight", "q_deq_scale", "q_quant_bias"],
)
def test_static_mlapo_rejects_incompatible_parameter_shapes(name):
    parameters, dims = _parameters()
    parameters[name] = parameters[name][:-1] if name != "qkv_weight" else parameters[name][:, :-1]
    with pytest.raises(ValueError, match=name):
        prepare_static_mlapo_weights(**parameters, **dims)


@pytest.mark.parametrize(
    "name,dtype",
    [
        ("qkv_weight", torch.int32),
        ("q_weight", torch.float32),
        ("qkv_deq_scale", torch.float16),
        ("q_deq_scale", torch.int64),
        ("qkv_quant_bias", torch.float32),
        ("q_quant_bias", torch.float16),
    ],
)
def test_static_mlapo_rejects_incompatible_parameter_dtypes(name, dtype):
    parameters, dims = _parameters()
    parameters[name] = parameters[name].to(dtype)
    with pytest.raises(ValueError, match="dtype"):
        prepare_static_mlapo_weights(**parameters, **dims)


@pytest.mark.parametrize(
    "name,value",
    [("q_lora_rank", 0), ("kv_lora_rank", -1), ("num_heads", 0), ("qk_nope_head_dim", -1), ("qk_rope_head_dim", 3)],
)
def test_static_mlapo_rejects_invalid_dimensions(name, value):
    parameters, dims = _parameters()
    dims[name] = value
    with pytest.raises(ValueError, match="Static MLAPO"):
        prepare_static_mlapo_weights(**parameters, **dims)


def test_static_mlapo_rejects_unaligned_weights():
    parameters, dims = _parameters()
    parameters["qkv_weight"] = parameters["qkv_weight"][:-1]
    with pytest.raises(ValueError, match="divisible"):
        prepare_static_mlapo_weights(**parameters, **dims)
