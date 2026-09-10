# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""CPU storage-flow tests for DeepSeek-V4 mHC residual aliasing.

Extract the real methods to avoid importing vLLM/NPU runtime dependencies.
Can run independently with pytest --noconftest on CPU-only installations.
"""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

MODEL_PATH = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v4/model.py"


def _decoder_methods(clone_residual: bool = False) -> ast.Module:
    tree = ast.parse(MODEL_PATH.read_text())
    layer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DeepseekV4DecoderLayer")
    methods = copy.deepcopy(
        [n for n in layer.body if isinstance(n, ast.FunctionDef) and n.name in {"forward", "hc_pre", "hc_post"}]
    )
    forward = next(n for n in methods if n.name == "forward")
    snapshots = [n for n in forward.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "residual"]
    assert len(snapshots) == 2
    for snapshot in snapshots:
        assert ast.unparse(snapshot.value) == "hidden_states"
        if clone_residual:
            snapshot.value = ast.parse("hidden_states.clone()", mode="eval").body
    return ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[]))


def _storage(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


def _run(
    *,
    clone_residual: bool = False,
    unsafe_half: int | None = None,
    input_ids: torch.Tensor | None = None,
    sequence_parallel: bool = False,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[str],
]:
    pre_inputs: list[torch.Tensor] = []
    snapshots: list[torch.Tensor] = []
    post_outputs: list[torch.Tensor] = []
    residuals: list[torch.Tensor] = []
    events: list[str] = []
    positions = torch.arange(2)
    scaling = torch.ones(2)

    def hc_pre(x, fn, scale, base, *args):
        half = len(pre_inputs)
        assert (fn, scale, base) == (half, half, half)
        pre_inputs.append(x)
        snapshots.append(x.clone())
        events.append("pre")
        branch = x[:, 0, :] if unsafe_half == half else x.sum(dim=1)
        return branch, torch.ones(2, 4), torch.eye(4).expand(2, 4, 4)

    def hc_post(x, residual, post, comb):
        events.append("post")
        assert (x.shape, residual.shape, post.shape, comb.shape) == (
            (1, 2, 3),
            (1, 2, 4, 3),
            (1, 2, 4),
            (1, 2, 4, 4),
        )
        residuals.append(residual.squeeze(0))
        result = residual + x.unsqueeze(2)
        post_outputs.append(result.squeeze(0))
        return result

    def norm(x):
        events.append("norm")
        return x.add_(1)

    def attention(*, positions: object, hidden_states, llama_4_scaling):
        assert positions is expected_positions
        assert llama_4_scaling is scaling
        events.append("attention")
        return hidden_states.mul_(2)

    def rms_norm_cast(x):
        events.append("rms_norm_cast")
        return x.add_(1), x.float().clone()

    def mlp(x, *, input_ids: object, hidden_states_fp32):
        assert input_ids is expected_ids
        assert torch.equal(x.float(), hidden_states_fp32)
        events.append("mlp")
        return x.mul_(3)

    def sp_all_gather(x):
        events.append("sp_all_gather")
        padding = torch.full_like(x[:1], -99)
        return torch.cat((x, padding), dim=0)

    def sp_reduce_scatter(x):
        events.append("sp_reduce_scatter")
        return x.clone()

    expected_positions, expected_ids = positions, input_ids
    ops = SimpleNamespace(npu_hc_pre_v2=hc_pre, npu_hc_post=hc_post)
    namespace = {
        "sp_all_gather": sp_all_gather,
        "sp_reduce_scatter": sp_reduce_scatter,
        "torch": SimpleNamespace(Tensor=torch.Tensor, ops=SimpleNamespace(_C_ascend=ops)),
    }
    exec(compile(_decoder_methods(clone_residual), str(MODEL_PATH), "exec"), namespace)
    layer_type = type(
        "DecoderMethods",
        (),
        {name: namespace[name] for name in ("forward", "hc_pre", "hc_post")},
    )
    layer = layer_type()
    for half, prefix in enumerate(("attn", "ffn")):
        for suffix in ("fn", "scale", "base"):
            setattr(layer, f"hc_{prefix}_{suffix}", half)
    layer.hc_mult, layer.hc_sinkhorn_iters, layer.norm_eps, layer.hc_eps = 4, 20, 1e-6, 1e-6
    layer.input_layernorm, layer.self_attn = norm, attention
    layer.rms_norm_cast, layer.mlp = rms_norm_cast, mlp
    layer.use_sequence_parallel_moe = sequence_parallel
    layer.enable_dsa_cp = False
    original = torch.arange(24, dtype=torch.bfloat16).reshape(2, 4, 3)
    incoming_residual = torch.full_like(original, -1)
    result = layer.forward(positions, original, incoming_residual, scaling, input_ids)
    assert torch.equal(incoming_residual, torch.full_like(original, -1))
    return result, pre_inputs, snapshots, post_outputs, residuals, events


@pytest.mark.parametrize("input_ids", [None, torch.tensor([3, 7])])
@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_both_mhc_halves_preserve_residual_storage_and_values(input_ids, sequence_parallel):
    result, inputs, snapshots, outputs, residuals, events = _run(
        input_ids=input_ids,
        sequence_parallel=sequence_parallel,
    )
    baseline, _, _, _, baseline_residuals, baseline_events = _run(
        clone_residual=True,
        input_ids=input_ids,
        sequence_parallel=sequence_parallel,
    )
    expected_events = ["pre", "norm"]
    if sequence_parallel:
        expected_events.extend(["sp_all_gather", "attention", "sp_reduce_scatter"])
    else:
        expected_events.append("attention")
    expected_events.extend(["post", "pre", "rms_norm_cast", "mlp", "post"])
    assert events == expected_events
    assert baseline_events == expected_events
    for actual, expected in zip(result, baseline):
        assert torch.equal(actual, expected)
    for half in range(2):
        assert _storage(residuals[half]) == _storage(inputs[half])
        assert torch.equal(inputs[half], snapshots[half])
        assert torch.equal(residuals[half], baseline_residuals[half])
        assert _storage(outputs[half]) != _storage(inputs[half])
    assert _storage(inputs[1]) == _storage(outputs[0])
    assert result[1] is inputs[1]
    assert _storage(result[0]) != _storage(result[1])
    saved_residual = result[1].clone()
    result[0].zero_()
    assert torch.equal(result[1], saved_residual)


@pytest.mark.parametrize("unsafe_half", [0, 1], ids=["attention", "ffn"])
def test_alias_is_unsafe_when_hc_pre_returns_mutating_view(unsafe_half):
    result, inputs, snapshots, _, residuals, _ = _run(unsafe_half=unsafe_half)
    baseline, _, baseline_snapshots, _, baseline_residuals, _ = _run(
        clone_residual=True,
        unsafe_half=unsafe_half,
    )
    assert not torch.equal(inputs[unsafe_half], snapshots[unsafe_half])
    assert not torch.equal(residuals[unsafe_half], snapshots[unsafe_half])
    assert torch.equal(baseline_residuals[unsafe_half], baseline_snapshots[unsafe_half])
    assert not torch.equal(result[0], baseline[0])
