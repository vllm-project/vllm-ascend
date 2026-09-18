# SPDX-License-Identifier: Apache-2.0
"""Exercise KDA output ownership without importing the NPU/vLLM runtime."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from einops import rearrange
from torch.utils._python_dispatch import TorchDispatchMode


def load_forward(metadata):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/kimi_kda.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    prepare = next(node for node in tree.body if getattr(node, "name", None) == "_prepare_beta")
    cls = next(node for node in tree.body if getattr(node, "name", None) == "AscendKimiK3DeltaAttention")
    forward = next(node for node in cls.body if getattr(node, "name", None) == "_forward")
    forward.decorator_list = []
    scope = dict(
        torch=torch,
        rearrange=rearrange,
        get_forward_context=lambda: SimpleNamespace(attn_metadata={"kda": metadata}),
        GDNAttentionMetadata=SimpleNamespace,
        envs=SimpleNamespace(VLLM_ASCEND_ENABLE_FLASH_MLA=True),
        _PACKED_CONV_WEIGHT_NAME="ascend_conv1d_weight",
    )
    exec(compile(ast.Module(body=[prepare, forward], type_ignores=[]), str(path), "exec"), scope)
    return scope["_forward"]


class _RecordOps(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func))
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("mode", ["spec", "decode", "prefill", "mixed", "mixed_holes", "idle"])
@pytest.mark.parametrize("padding", [0, 2])
def test_forward_output_skips_redundant_initialization(mode, padding):
    n, h, d = 6, 2, 4
    starts = torch.tensor([0, n], dtype=torch.int32)
    slots = torch.tensor([0], dtype=torch.int32)
    conv_meta = SimpleNamespace(
        query_start_loc=starts, cache_indices=slots, num_accepted_tokens=None, initial_state_mode=None
    )
    mixed = mode in ("mixed", "mixed_holes")
    spec_indices = torch.tensor([0, 2] if mode == "mixed_holes" else [0, 2, 4])
    non_spec_indices = torch.tensor([1, 3] if mode == "mixed_holes" else [1, 3, 5])
    metadata = SimpleNamespace(
        num_actual_tokens=n,
        num_prefills=int(mode == "prefill"),
        num_decodes=int(mode in ("decode", "mixed", "mixed_holes")),
        num_decode_tokens=n,
        spec_sequence_masks=torch.tensor([True]) if mode == "spec" or mixed else None,
        spec_token_indx=spec_indices,
        non_spec_token_indx=non_spec_indices,
        spec_decode_metadata=SimpleNamespace(spec_causal_conv1d=conv_meta),
        non_spec_decode_metadata=SimpleNamespace(causal_conv1d=conv_meta),
        non_spec_prefill_metadata=SimpleNamespace(causal_conv1d=conv_meta, chunk=None),
        spec_query_start_loc=starts,
        non_spec_query_start_loc=starts,
        spec_state_indices_tensor=slots,
        non_spec_state_indices_tensor=slots,
        prefill_state_indices=slots,
        prefill_has_initial_state=torch.tensor([False]),
    )
    values = torch.arange(1, (n + padding) * h * d * 3 + 1, dtype=torch.float32).reshape(n + padding, -1)
    output = torch.full((1, n + padding, h, d), torch.nan)
    recurrent_outputs = []
    norm_inputs = []
    before_norm = []
    trace = _RecordOps()

    def recurrent(q, k, v, *args, **kwargs):
        result = v.contiguous()
        recurrent_outputs.append(result)
        return result

    def norm(x, g, *, out):
        before_norm.extend(trace.ops)
        norm_inputs.append(x)
        out.copy_(x * torch.sigmoid(g))
        return out

    attention = SimpleNamespace(
        prefix="kda",
        head_dim=d,
        _conv_max_query_len=4,
        kv_cache=(torch.zeros(1), torch.zeros(1)),
        get_buffer=lambda name: torch.empty(1),
        _run_causal_conv1d=lambda x, *args, **kwargs: x,
        _run_recurrent=recurrent,
        _run_prefill=recurrent,
        o_norm=norm,
    )
    forward = load_forward(metadata)
    with trace:
        forward(
            attention,
            values,
            torch.zeros(1, n + padding, h, d),
            torch.zeros(n + padding, h, d),
            torch.zeros(1, n + padding, h),
            output,
        )
    expected = values[:n, 2 * h * d :].reshape(1, n, h, d) * 0.5
    if mode == "mixed_holes":
        # Unscheduled rows have no output contract. Poisoning them verifies
        # they do not leak into any live token through norm/scatter.
        expected[:, 4:] = torch.nan
    elif mode == "idle":
        expected.zero_()
    torch.testing.assert_close(output[:, :n], expected, equal_nan=True)
    if mode == "idle":
        torch.testing.assert_close(output[:, n:], torch.zeros_like(output[:, n:]))
    else:
        assert torch.isnan(output[:, n:]).all()
        assert "aten.zero_.default" not in trace.ops
    if mode in ("spec", "decode", "prefill"):
        assert norm_inputs[0] is recurrent_outputs[0]
        assert "aten.zero_.default" not in before_norm
    elif mixed:
        assert "aten.zero_.default" not in before_norm
    else:
        assert not norm_inputs

    if mode in ("spec", "decode", "prefill"):
        # Reuse the graph output with fewer live tokens. Previously live rows
        # become padding and need neither a clear nor a copy on this replay.
        previous = output.clone()
        metadata.num_actual_tokens = 2
        metadata.num_decode_tokens = 2
        starts[-1] = 2
        values.add_(7)
        trace.ops.clear()
        with trace:
            forward(
                attention,
                values,
                torch.zeros(1, n + padding, h, d),
                torch.zeros(n + padding, h, d),
                torch.zeros(1, n + padding, h),
                output,
            )
        expected = values[:2, 2 * h * d :].reshape(1, 2, h, d) * 0.5
        torch.testing.assert_close(output[:, :2], expected)
        torch.testing.assert_close(output[:, 2:], previous[:, 2:], equal_nan=True)
        assert "aten.zero_.default" not in trace.ops
