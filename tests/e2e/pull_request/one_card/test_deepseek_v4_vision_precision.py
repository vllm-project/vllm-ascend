import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
from safetensors.torch import load_file
from vllm.config import VllmConfig, set_current_vllm_config

from vllm_ascend.models.deepseek_v4.vision import (
    DeepseekV4VisionAttention,
    DeepseekV4VisionBlock,
    DeepseekV4ViT,
    get_vision_cos_sin,
)
from vllm_ascend.utils import register_ascend_customop

VISION_DIM = 1024
VISION_HEADS = 16
VISION_INTER_DIM = 2816
RMS_NORM_EPS = 1e-6


@pytest.fixture(scope="module", autouse=True)
def register_ascend_vision_ops():
    register_ascend_customop()
    with set_current_vllm_config(VllmConfig()):
        yield


@pytest.fixture
def vision_config():
    return SimpleNamespace(
        vision_dim=VISION_DIM,
        vision_n_heads=VISION_HEADS,
        vision_inter_dim=VISION_INTER_DIM,
    )


def _new_bf16_module(module_cls, config):
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        module = module_cls(config)
    finally:
        torch.set_default_dtype(original_dtype)
    return module.eval().npu()


def _reference_rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x_fp32 = x.float()
    normalized = x_fp32 * torch.rsqrt(x_fp32.square().mean(-1, keepdim=True) + RMS_NORM_EPS)
    return (weight * normalized).to(dtype)


def _reference_apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).to(dtype)


def _reference_attention(
    attention: DeepseekV4VisionAttention,
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    num_tokens = x.size(0)
    q, k, v = (
        tensor.view(num_tokens, attention.n_heads, attention.head_dim) for tensor in attention.wqkv(x).chunk(3, dim=-1)
    )
    q = _reference_apply_rotary(q, cos, sin)
    k = _reference_apply_rotary(k, cos, sin)
    output = F.scaled_dot_product_attention(
        q.transpose(0, 1),
        k.transpose(0, 1),
        v.transpose(0, 1),
    )
    return attention.wo(output.transpose(0, 1).reshape(num_tokens, -1))


def _reference_mlp(block: DeepseekV4VisionBlock, x: torch.Tensor) -> torch.Tensor:
    gate, up = block.mlp.w1(x).chunk(2, dim=-1)
    return block.mlp.w2(F.silu(gate) * up)


def _reference_block(
    block: DeepseekV4VisionBlock,
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    attention_input = _reference_rms_norm(x, block.norm1.weight)
    residual = x + _reference_attention(block.attn, attention_input, cos, sin)
    mlp_input = _reference_rms_norm(residual, block.norm2.weight)
    return residual + _reference_mlp(block, mlp_input)


def _assert_vision_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    max_abs_error: float,
    mean_abs_error: float,
    min_cosine_similarity: float,
) -> None:
    actual_fp32 = actual.float().reshape(-1)
    expected_fp32 = expected.float().reshape(-1)
    error = (actual_fp32 - expected_fp32).abs()
    cosine_similarity = F.cosine_similarity(actual_fp32, expected_fp32, dim=0)

    max_error = error.max().item()
    mean_error = error.mean().item()
    cosine = cosine_similarity.item()
    metrics = f"max_abs_error={max_error}, mean_abs_error={mean_error}, cosine_similarity={cosine}"
    print(metrics)

    assert max_error <= max_abs_error, metrics
    assert mean_error <= mean_abs_error, metrics
    assert cosine >= min_cosine_similarity, metrics


@pytest.mark.parametrize("grid", [(4, 4), (16, 24), (48, 72)])
def test_fused_attention_matches_original_formula(vision_config, grid):
    torch.manual_seed(7)
    attention = _new_bf16_module(DeepseekV4VisionAttention, vision_config)
    num_tokens = grid[0] * grid[1]
    x = torch.randn(num_tokens, VISION_DIM, dtype=torch.bfloat16, device="npu")
    cos, sin = get_vision_cos_sin(grid[0], grid[1], attention.head_dim // 2, 10000.0)
    cos = cos.npu()
    sin = sin.npu()

    with torch.inference_mode():
        expected = _reference_attention(attention, x, cos, sin)
        actual = attention(x, cos, sin)
    torch.npu.synchronize()

    _assert_vision_close(
        actual,
        expected,
        max_abs_error=0.03125,
        mean_abs_error=0.002,
        min_cosine_similarity=0.9999,
    )


@pytest.mark.parametrize("grid", [(4, 4), (16, 24), (48, 72)])
def test_fused_vision_block_matches_original_formula(vision_config, grid):
    torch.manual_seed(17)
    block = _new_bf16_module(DeepseekV4VisionBlock, vision_config)
    num_tokens = grid[0] * grid[1]
    x = torch.randn(num_tokens, VISION_DIM, dtype=torch.bfloat16, device="npu")
    cos, sin = get_vision_cos_sin(grid[0], grid[1], VISION_DIM // VISION_HEADS // 2, 10000.0)
    cos = cos.npu()
    sin = sin.npu()

    with torch.inference_mode():
        expected = _reference_block(block, x, cos, sin)
        actual = block(x.clone(), cos, sin)
    torch.npu.synchronize()

    _assert_vision_close(
        actual,
        expected,
        max_abs_error=0.0625,
        mean_abs_error=0.004,
        min_cosine_similarity=0.9999,
    )


def test_real_weights_full_vit_matches_original_formula():
    model_path = os.getenv("DEEPSEEK_V4_VISION_MODEL_PATH")
    if not model_path:
        pytest.skip("DEEPSEEK_V4_VISION_MODEL_PATH is not set")
    assert model_path is not None

    model_root = Path(model_path)
    config = SimpleNamespace(**json.loads((model_root / "config.json").read_text()))
    checkpoint = load_file(model_root / "quant_model_weights-00078-of-00078.safetensors")
    vision_weights = {
        name.removeprefix("vision."): tensor for name, tensor in checkpoint.items() if name.startswith("vision.")
    }
    model = _new_bf16_module(DeepseekV4ViT, config)
    model.load_state_dict(vision_weights, strict=True)

    torch.manual_seed(29)
    grid = (16, 24)
    patches = torch.randn(
        grid[0] * grid[1],
        3,
        config.vision_patch_size,
        config.vision_patch_size,
        dtype=torch.bfloat16,
        device="npu",
    )

    with torch.inference_mode():
        expected = model.patch_embed(patches)
        cos, sin = get_vision_cos_sin(grid[0], grid[1], model.rope_dim, model.rope_theta)
        cos = cos.npu()
        sin = sin.npu()
        for block in model.blocks:
            expected = _reference_block(block, expected, cos, sin)
        expected = _reference_rms_norm(expected, model.norm.weight)
        actual = model(patches, *grid)
    torch.npu.synchronize()

    _assert_vision_close(
        actual,
        expected,
        max_abs_error=0.125,
        mean_abs_error=0.005,
        min_cosine_similarity=0.9999,
    )
