# SPDX-License-Identifier: Apache-2.0
"""Real NPU QSA projection/normalization and MTP graph regression.

Select the exact source tree through PYTHONPATH. No norm/rotary/kernel mocks.
Thresholds are fixed before execution: BF16 rtol=.02, atol=.04; graph/eager
uses rtol=.016, atol=.008. Full model forward is a separate service test.
"""

import argparse
import hashlib
import json
import subprocess
import traceback
from functools import partial
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.rotary_embedding import get_rope

from vllm_ascend.models.qwen4_exp import qsa
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

P = argparse.ArgumentParser()
P.add_argument("--output", required=True)
P.add_argument("--graph", action="store_true")
P.add_argument("--indexer-only", action="store_true")
A = P.parse_args()
torch.npu.set_device(0)
init_device_properties_triton()
torch.manual_seed(21979)


def git_commit(path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.CalledProcessError:
        return None


RESULT = {
    "source": qsa.__file__,
    "ascend_commit": git_commit(Path(qsa.__file__).parents[3]),
    "qsa_sha256": hashlib.sha256(Path(qsa.__file__).read_bytes()).hexdigest(),
    "torch_version": torch.__version__,
    "threshold": {"rtol": 0.02, "atol": 0.04},
    "cases": [],
}
COUNTS = {"main": 0, "indexer": 0}
for name, key in [("triton_split_qkv_rmsnorm_mrope", "main"), ("triton_qsa_rmsnorm_mrope", "indexer")]:
    if not hasattr(torch.ops.vllm, name):
        continue
    original = getattr(torch.ops.vllm, name)

    def wrapper(*args, _original=original, _key=key, **kwargs):
        COUNTS[_key] += 1
        return _original(*args, **kwargs)

    setattr(torch.ops.vllm, name, wrapper)


def save():
    p = Path(A.output)
    p.with_suffix(".tmp").write_text(json.dumps(RESULT, indent=2))
    p.with_suffix(".tmp").replace(p)


def check(actual, expected, *, exact=False):
    actual, expected = actual.detach().float().cpu(), expected.detach().float().cpu()
    assert torch.isfinite(actual).all(), "nonfinite output"
    delta = actual - expected
    metrics = {
        "max_abs": delta.abs().max().item(),
        "l2_relative": (delta.norm() / expected.norm().clamp_min(1e-12)).item(),
    }
    torch.testing.assert_close(actual, expected, rtol=0 if exact else 0.02, atol=0 if exact else 0.04)
    return metrics


def rotary(mrope):
    parameters = {"rope_type": "default", "rope_theta": 10000000.0, "partial_rotary_factor": 0.25}
    if mrope:
        parameters.update(mrope_section=[11, 11, 10], mrope_interleaved=True)
    return get_rope(head_size=256, max_position=512, rope_parameters=parameters).to(device="npu", dtype=torch.bfloat16)


def norm(width):
    module = GemmaRMSNorm(width, eps=1e-6).to(device="npu", dtype=torch.bfloat16)
    with torch.no_grad():
        module.weight.copy_(torch.linspace(-0.2, 0.4, width, device="npu", dtype=torch.bfloat16))
    return module


def independent_reference(x, module, rot, positions):
    # Gemma norm has beta=1. Round the normalization boundary to BF16 as the
    # decomposed path does, then rotate with independently selected cache rows.
    x = x.detach().float().cpu()
    weights = module.weight.detach().float().cpu()
    x = (
        (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + module.variance_epsilon) * (1 + weights))
        .bfloat16()
        .float()
    )
    positions = positions.cpu()
    cache = rot.cos_sin_cache.detach().float().cpu()
    if positions.ndim == 2 and not getattr(rot, "mrope_section", None):
        positions = positions[0]
    cos, sin = cache[positions].chunk(2, -1)
    if positions.ndim == 2:
        # Interleaved three-axis definition: channel j chooses axis j % 3,
        # up to the configured count on that axis.
        section = rot.mrope_section
        axis = torch.zeros(32, dtype=torch.long)
        for j in range(32):
            candidate = j % 3
            if j < section[candidate] * 3:
                axis[j] = candidate
        channels = torch.arange(32)
        cos, sin = cos[axis, :, channels].T, sin[axis, :, channels].T
    cos, sin = cos[:, None], sin[:, None]
    first, second = x[..., :32], x[..., 32:64]
    return torch.cat([first * cos - second * sin, second * cos + first * sin, x[..., 64:]], -1).bfloat16()


def make_layer(mrope, gate, role):
    layer = qsa.AscendQwen4ExpQSAAttention.__new__(qsa.AscendQwen4ExpQSAAttention)
    torch.nn.Module.__init__(layer)
    layer.layer_name = "mtp.layers.48.self_attn" if role == "mtp" else "model.layers.3.self_attn"
    layer.head_dim, layer.num_heads, layer.num_kv_heads = 256, 2, 1
    layer.q_size, layer.kv_size, layer.attn_output_gate = 512, 256, gate
    layer.use_fused_qk_norm_rope_gate = False  # Actual upstream NPU eligibility.
    layer.q_norm, layer.k_norm, layer.rotary_emb = norm(256), norm(256), rotary(mrope)
    return layer


def inputs(tokens, gate, dims):
    q = torch.randn(tokens, 2, 256, device="npu", dtype=torch.bfloat16)
    k = torch.randn(tokens, 1, 256, device="npu", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    g = torch.randn_like(q) if gate else None
    packed_q = torch.cat([q, g], -1) if gate else q
    packed = torch.cat([packed_q.flatten(1), k.flatten(1), v.flatten(1)], -1).contiguous()
    positions = torch.arange(7, 7 + tokens, device="npu", dtype=torch.int64)
    if dims == 2:
        positions = torch.stack([positions, positions + 19, positions + 37])
    return packed, positions, q, k, v, g


def projection_case(tokens, mrope, gate, role, dims):
    layer = make_layer(mrope, gate, role)
    packed, pos, q, k, v, gate_value = inputs(tokens, gate, dims)
    before = dict(COUNTS)
    output = layer._project_qkv_gate(packed, pos)
    torch.npu.synchronize()
    result = {
        "q": check(output[0], independent_reference(q, layer.q_norm, layer.rotary_emb, pos).flatten(1)),
        "k": check(output[1], independent_reference(k, layer.k_norm, layer.rotary_emb, pos).flatten(1)),
        "v": check(output[2], v.flatten(1), exact=True),
    }
    if gate:
        result["gate"] = check(output[3], gate_value.flatten(1), exact=True)
    else:
        assert output[3] is None
    fused_expected = role == "target" and mrope and gate and dims == 2
    result["fused_calls"] = {key: COUNTS[key] - before[key] for key in COUNTS}
    assert result["fused_calls"]["main"] == int(fused_expected), result
    if role == "mtp":
        try:
            upstream = qsa.upstream_qsa.Qwen4ExpQSAAttention._project_qkv_gate(layer, packed, pos)
            torch.npu.synchronize()
        except RuntimeError as error:
            # Plain RoPE with three-axis positions is unsupported upstream.
            # Numerical assertion failures and every other exception must fail.
            if mrope or dims != 2 or "shape" not in str(error):
                raise
            result["upstream_exception"] = type(error).__name__ + ": " + str(error)
        else:
            comparisons = []
            for actual, reference in zip(output, upstream):
                if actual is None or reference is None:
                    assert actual is reference
                    comparisons.append({"both_none": True})
                else:
                    comparisons.append(check(actual, reference))
            result["upstream"] = comparisons
    return result


def indexer_case(tokens, mrope, dims):
    obj = qsa.AscendQSAIndexer.__new__(qsa.AscendQSAIndexer)
    torch.nn.Module.__init__(obj)
    obj.index_head_dim, obj.index_n_heads, obj.index_kv_heads = 128, 2, 1
    obj.rotary_emb, obj.q_layernorm, obj.k_layernorm = rotary(mrope), norm(128), norm(128)
    # A real linear projection; the comparison uses its actual output.
    obj.index_qk_proj = torch.nn.Linear(128, 384, bias=False, dtype=torch.bfloat16, device="npu")

    class LinearPair(torch.nn.Module):
        def __init__(self, linear):
            super().__init__()
            self.linear = linear

        def forward(self, x):
            return self.linear(x), None

    obj.index_qk_proj = LinearPair(obj.index_qk_proj)
    hidden = torch.randn(tokens, 128, device="npu", dtype=torch.bfloat16)
    projected = obj.index_qk_proj(hidden)[0]
    q, raw = projected[:, :256].reshape(tokens, 2, 128), projected[:, 256:].reshape(tokens, 1, 128)
    pos = torch.arange(9, 9 + tokens, device="npu", dtype=torch.int64)
    if dims == 2:
        pos = torch.stack([pos, pos + 11, pos + 23])
    before = dict(COUNTS)
    out_q, out_raw = obj.project_qk(hidden, pos)
    first = pos.T if dims == 2 else pos[:, None].expand(-1, 3)
    normalized = obj.normalize_compressed_keys(raw, first)
    torch.npu.synchronize()
    result = {
        "q": check(out_q, independent_reference(q, obj.q_layernorm, obj.rotary_emb, pos)),
        "raw": check(out_raw, raw, exact=True),
        "compressed": check(normalized, independent_reference(raw, obj.k_layernorm, obj.rotary_emb, pos)),
    }
    result["fused_calls"] = {key: COUNTS[key] - before[key] for key in COUNTS}
    assert result["fused_calls"]["indexer"] == (2 if mrope else 0), result
    return result


def graph_case(tokens, mrope, upstream=False):
    layer = make_layer(mrope, True, "mtp")
    project = (
        (lambda qkv, positions: qsa.upstream_qsa.Qwen4ExpQSAAttention._project_qkv_gate(layer, qkv, positions))
        if upstream
        else layer._project_qkv_gate
    )
    packed, pos, *_ = inputs(tokens, True, 2 if mrope else 1)
    for _ in range(3):
        project(packed, pos)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = project(packed, pos)
    errors = []
    for index in range(10):
        fresh, new_pos, *_ = inputs(tokens, True, 2 if mrope else 1)
        new_pos.add_(index)
        packed.copy_(fresh)
        pos.copy_(new_pos)
        graph.replay()
        torch.npu.synchronize()
        # clone before eager may reuse temporary allocations.
        actual = [x.clone() if x is not None else None for x in output]
        eager = project(packed, pos)
        torch.npu.synchronize()
        for a, b in zip(actual, eager):
            torch.testing.assert_close(a, b, rtol=0.016, atol=0.008)
        errors.append([check(a, b) for a, b in zip(actual, eager)])
    return {
        "capture": 1,
        "replays": 10,
        "errors": errors,
        "implementation": "upstream" if upstream else "PR projection",
        "scope": "real MTP projection, not complete drafter forward",
    }


def run(name, function):
    item = {"name": name}
    try:
        item.update(status="PASS", result=function())
    except Exception as error:
        item.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
    RESULT["cases"].append(item)
    save()
    print(name, item["status"], flush=True)


with set_current_vllm_config(VllmConfig()), torch.inference_mode():
    for tokens in [1, 8]:
        for mrope in [False, True]:
            for dims in [1, 2]:
                if not A.indexer_only:
                    for gate in [False, True]:
                        for role in ["target", "mtp"]:
                            run(
                                f"projection-T{tokens}-mrope{mrope}-dim{dims}-gate{gate}-{role}",
                                partial(projection_case, tokens, mrope, gate, role, dims),
                            )
                run(f"indexer-T{tokens}-mrope{mrope}-dim{dims}", partial(indexer_case, tokens, mrope, dims))
    if A.graph:
        for tokens in [1, 8]:
            for mrope in [False, True]:
                run(f"mtp-projection-graph-T{tokens}-mrope{mrope}", partial(graph_case, tokens, mrope))
                run(f"upstream-mtp-projection-graph-T{tokens}-mrope{mrope}", partial(graph_case, tokens, mrope, True))
RESULT["counts"] = COUNTS
save()
raise SystemExit(int(any(case["status"] == "FAIL" for case in RESULT["cases"])))
