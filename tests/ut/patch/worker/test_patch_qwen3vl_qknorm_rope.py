import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

PATCH_PATH = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/worker/patch_qwen3vl.py"


def _load_qwen3_forward(helper):
    source = ast.parse(PATCH_PATH.read_text(encoding="utf-8"))
    forward = next(
        node
        for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward_with_split_qkv_rmsnorm_mrope"
    )
    namespace = {
        "torch": torch,
        "AscendMRotaryEmbedding": type("MRoPE", (), {}),
        "maybe_fused_qknorm_rope": helper,
    }
    exec(compile(ast.Module(body=[forward], type_ignores=[]), str(PATCH_PATH), "exec"), namespace)
    return namespace[forward.name]


@pytest.mark.parametrize("use_fusion", [True, False])
def test_qwen3_attention_explicit_qknorm_rope_and_fallback(use_fusion):
    qkv = torch.arange(32, dtype=torch.bfloat16).reshape(2, 16)
    q, k, v = qkv.split((8, 4, 4), dim=-1)
    positions = torch.arange(2)
    q_norm = Mock(side_effect=lambda x: x)
    k_norm = Mock(side_effect=lambda x: x)
    rotary = Mock(side_effect=lambda positions, query, key: (query, key))
    helper = Mock(return_value=(q, k, v) if use_fusion else None)
    attention = SimpleNamespace(
        qkv_proj=Mock(return_value=(qkv, None)),
        q_norm=q_norm,
        k_norm=k_norm,
        rotary_emb=rotary,
        q_size=8,
        kv_size=4,
        head_dim=4,
        attn=Mock(side_effect=lambda query, key, value: query),
        o_proj=Mock(side_effect=lambda output: (output, None)),
    )

    output = _load_qwen3_forward(helper)(attention, positions, torch.empty(2, 8))

    torch.testing.assert_close(output, q)
    helper.assert_called_once_with(qkv, positions, q_norm, k_norm, rotary, 8, 4, 4)
    if use_fusion:
        q_norm.assert_not_called()
        k_norm.assert_not_called()
        rotary.assert_not_called()
    else:
        q_norm.assert_called_once()
        k_norm.assert_called_once()
        rotary.assert_called_once()
