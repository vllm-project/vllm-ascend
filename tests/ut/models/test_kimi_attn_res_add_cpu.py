# SPDX-License-Identifier: Apache-2.0
"""Test residual-add ownership without importing the NPU runtime."""

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch


def load_functions():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/kimi_k3.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    apply = next(node for node in tree.body if getattr(node, "name", None) == "_apply_ascend_attn_res")
    layer = next(node for node in tree.body if getattr(node, "name", None) == "AscendKimiDecoderLayer")
    forward = next(node for node in layer.body if getattr(node, "name", None) == "forward_attn_residual")
    scope = {"torch": torch}
    prepare = next(node for node in layer.body if getattr(node, "name", None) == "prepare_attn_residual")
    module = ast.Module(body=[apply, prepare, forward], type_ignores=[])
    exec(compile("from __future__ import annotations\n" + ast.unparse(module), str(path), "exec"), scope)
    return scope


def native_reference(prefix, blocks, projection, gamma, eps):
    values = torch.cat((blocks, prefix.unsqueeze(1)), dim=1).float()
    normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
    scores = (normalized * (gamma.float() * projection.float())).sum(-1)
    return (scores.softmax(-1).unsqueeze(-1) * values).sum(1).to(prefix.dtype)


def fused_reference(
    prefix,
    addend,
    blocks,
    projection,
    gamma,
    eps,
    valid,
    output_norm_weight=None,
    output_norm_eps=1e-5,
    block_write_idx=-1,
    return_materialized=False,
    mix=True,
):
    raw_prefix = prefix if addend is None else prefix + addend
    value = native_reference(raw_prefix, blocks[:, :valid], projection, gamma, eps) if mix and valid else raw_prefix
    if block_write_idx >= 0:
        blocks[:, block_write_idx].copy_(raw_prefix)
    if output_norm_weight is not None:
        normalized = value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + output_norm_eps)
        output = (normalized * output_norm_weight.float()).to(value.dtype)
    else:
        output = value
    return output, raw_prefix, value


@pytest.fixture(autouse=True)
def mock_native_ops(monkeypatch):
    monkeypatch.setattr(native_reference, "fused", fused_reference, raising=False)
    monkeypatch.setattr(native_reference, "fused_prefill", fused_reference, raising=False)
    monkeypatch.setattr(torch.ops._C_ascend, "attn_res_fwd", native_reference, raising=False)

    def with_add(prefix, addend, blocks, projection, gamma, eps):
        new_prefix = prefix + addend
        return native_reference(new_prefix, blocks, projection, gamma, eps), new_prefix

    monkeypatch.setattr(torch.ops._C_ascend, "attn_res_fwd_with_add", with_add, raising=False)


@pytest.mark.parametrize("tokens", [0, 4])
@pytest.mark.parametrize("blocks", [0, 1, 4])
def test_native_adapter_selects_only_initialized_blocks(tokens, blocks):
    apply = load_functions()["_apply_ascend_attn_res"]
    torch.manual_seed(42)
    prefix = torch.randn(tokens, 32).bfloat16()[:, ::2]
    residual = torch.randn(tokens, 8, 16).bfloat16()
    residual[:, blocks:] = float("nan")
    projection = SimpleNamespace(weight=torch.randn(1, 16).bfloat16())
    norm = SimpleNamespace(weight=torch.randn(16).bfloat16(), variance_epsilon=1e-5)
    actual = apply(prefix, residual, projection, norm, blocks)
    expected = native_reference(prefix, residual[:, :blocks], projection.weight, norm.weight, norm.variance_epsilon)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if blocks == 0:
        assert actual is prefix


@pytest.mark.parametrize("optimize_prefill", [False, True])
def test_93_standalone_layers_use_native_residual_points_and_preserve_saved_dspark_prefixes(
    monkeypatch, optimize_prefill
):
    scope = load_functions()
    apply = scope["_apply_ascend_attn_res"]
    forward = scope["forward_attn_residual"]
    fused_calls = []
    fused = torch.ops._C_ascend.attn_res_fwd.fused

    def recording_fused(*args, **kwargs):
        fused_calls.append(args[0].clone())
        return fused(*args, **kwargs)

    selected_op = "fused_prefill" if optimize_prefill else "fused"
    other_op = "fused" if optimize_prefill else "fused_prefill"
    monkeypatch.setattr(torch.ops._C_ascend.attn_res_fwd, selected_op, recording_fused)

    def forbidden(*args, **kwargs):
        raise AssertionError("residual point reached the other phase's native operator")

    monkeypatch.setattr(torch.ops._C_ascend.attn_res_fwd, other_op, forbidden)
    torch.manual_seed(17)
    hidden = torch.randn(4, 16).to(torch.bfloat16)
    residual = torch.empty(4, 8, 16, dtype=hidden.dtype)
    projection = SimpleNamespace(weight=torch.randn(1, 16))
    norm = SimpleNamespace(weight=torch.randn(16), variance_epsilon=1e-5)
    for idx in range(93):
        prev_blocks = (idx + 11) // 12
        write = idx % 12 == 0
        layer = SimpleNamespace(
            use_sequence_parallel=False,
            prev_valid_blocks=prev_blocks,
            is_block_write_layer=write,
            block_write_idx=idx // 12,
            self_attention_res_proj=projection,
            self_attention_res_norm=norm,
            mlp_res_proj=projection,
            mlp_res_norm=norm,
            input_layernorm=SimpleNamespace(weight=None, variance_epsilon=1e-5),
            post_attention_layernorm=SimpleNamespace(weight=None, variance_epsilon=1e-5),
            self_attn=lambda *, hidden_states, positions: hidden_states * 0.25,
            mlp=lambda x: x * 0.125,
            _run_mlp=lambda x, _num_tokens: x * 0.125,
        )
        layer.prepare_attn_residual = MethodType(scope["prepare_attn_residual"], layer)
        old_alias, old_copy = hidden, hidden.clone()
        materialized = apply(hidden, residual, projection, norm, prev_blocks)
        if write:
            residual[:, idx // 12].copy_(hidden)
        attn_out = materialized * 0.25
        expected_prefix = attn_out if write else hidden + attn_out
        expected = (
            expected_prefix + apply(expected_prefix, residual, projection, norm, prev_blocks + int(write)) * 0.125
        )

        hidden, returned_residual = forward(layer, torch.arange(4), hidden, residual, optimize_prefill=optimize_prefill)

        torch.testing.assert_close(hidden, expected, rtol=0, atol=0)
        torch.testing.assert_close(old_alias, old_copy, rtol=0, atol=0)
        assert returned_residual is residual
    # Standalone layer calls materialize their tail; the full model absorbs
    # this third call into the next layer's first call instead.
    assert len(fused_calls) == 93 * 3


@pytest.mark.parametrize(
    "metadata,expected",
    [
        (None, False),
        ({}, False),
        ([], False),
        ({"mla": SimpleNamespace(num_prefills=0, num_decodes=1)}, False),
        ({"mla": SimpleNamespace(num_prefills=0, num_decodes=4096)}, False),
        ({"mla": SimpleNamespace(num_prefills=1, num_decodes=1)}, False),
        ({"mla": SimpleNamespace(num_prefills=0, num_decodes=0)}, False),
        ({"mla": SimpleNamespace(num_prefills=1)}, False),
        ({"mla": SimpleNamespace(num_prefills=torch.tensor(1), num_decodes=0)}, False),
        ({"kda": SimpleNamespace(num_prefills=1, num_decodes=0, spec_sequence_masks=torch.tensor([True]))}, False),
        ({"mla": SimpleNamespace(num_prefills=1, num_decodes=0)}, True),
        (
            {
                "mla": SimpleNamespace(num_prefills=1, num_decodes=0),
                "kda": SimpleNamespace(num_prefills=1, num_decodes=0),
            },
            True,
        ),
        (
            {
                "mla": SimpleNamespace(num_prefills=1, num_decodes=0),
                "kda": SimpleNamespace(num_prefills=0, num_decodes=1),
            },
            False,
        ),
    ],
)
@pytest.mark.parametrize(
    "context_available,is_950,has_prefill_op",
    [(True, True, True), (False, True, True), (True, False, True), (True, True, False)],
)
def test_prefill_kernel_requires_explicit_pure_prefill_metadata(
    monkeypatch, metadata, expected, context_available, is_950, has_prefill_op
):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/kimi_k3.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    gate = next(node for node in tree.body if getattr(node, "name", None) == "_use_attn_res_prefill_kernel")
    if not has_prefill_op:
        monkeypatch.delattr(native_reference, "fused_prefill")

    def get_context():
        assert context_available
        return SimpleNamespace(attn_metadata=metadata)

    scope = {
        "torch": torch,
        "is_forward_context_available": lambda: context_available,
        "get_forward_context": get_context,
        "is_950": lambda: is_950,
    }
    exec(compile(ast.Module(body=[gate], type_ignores=[]), str(path), "exec"), scope)
    assert scope["_use_attn_res_prefill_kernel"]() is (expected and context_available and is_950 and has_prefill_op)
