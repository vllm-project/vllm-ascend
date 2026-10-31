# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU output-assembly checks; device KDA and gated norm are emulated."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from einops import rearrange


def _load_forward_on_cpu(metadata):
    # Load the actual dispatch without importing vLLM or initializing an NPU
    # backend. Device kernels and the norm are supplied by each test.
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/kimi_kda.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    prepare_beta = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_prepare_beta")
    attention = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendKimiK3DeltaAttention")
    forward = next(n for n in attention.body if isinstance(n, ast.FunctionDef) and n.name == "_forward")
    forward.decorator_list = []
    save = MagicMock()
    scope = {
        "torch": torch,
        "rearrange": rearrange,
        "get_forward_context": lambda: SimpleNamespace(attn_metadata=metadata),
        "GDNAttentionMetadata": SimpleNamespace,
        "_PACKED_CONV_WEIGHT_NAME": "ascend_conv1d_weight",
        "wait_for_kv_layer_from_connector": MagicMock(),
        "record_attention_compute_start": MagicMock(),
        "maybe_save_kv_layer_to_connector": save,
    }
    exec(compile(ast.Module(body=[prepare_beta, forward], type_ignores=[]), str(source), "exec"), scope)
    return scope["_forward"], save


def _run_kda_output_case(mode, gate, output, norm, *, core_spec=None, core_non_spec=None, indices=None):
    """Exercise the real forward dispatch with CPU stand-ins for device ops."""
    tokens, heads, dim = gate.shape
    conv_meta = SimpleNamespace(
        query_start_loc=torch.tensor([0, tokens]),
        cache_indices=torch.tensor([0]),
        num_accepted_tokens=None,
        initial_state_mode=torch.tensor([False]),
    )
    metadata = SimpleNamespace(
        num_actual_tokens=tokens,
        num_prefills=int(mode == "prefill"),
        num_decodes=int(mode in ("decode", "mixed")),
        num_decode_tokens=tokens,
        spec_sequence_masks=torch.tensor([True]) if mode in ("spec", "mixed") else None,
        spec_token_indx=indices[0] if indices is not None else None,
        non_spec_token_indx=indices[1] if indices is not None else None,
        spec_decode_metadata=SimpleNamespace(spec_causal_conv1d=conv_meta),
        non_spec_decode_metadata=SimpleNamespace(causal_conv1d=conv_meta),
        non_spec_prefill_metadata=SimpleNamespace(causal_conv1d=conv_meta, chunk=object()),
        spec_query_start_loc=conv_meta.query_start_loc,
        non_spec_query_start_loc=conv_meta.query_start_loc,
        spec_state_indices_tensor=torch.tensor([0]),
        non_spec_state_indices_tensor=torch.tensor([0]),
        prefill_state_indices=torch.tensor([0]),
        prefill_has_initial_state=torch.tensor([False]),
    )
    recurrent_outputs = [core_spec] if mode == "spec" else [core_non_spec]
    if mode == "mixed":
        recurrent_outputs = [core_spec, core_non_spec]
    attention = SimpleNamespace(
        prefix="kda",
        head_dim=dim,
        num_spec=3,
        kv_cache=(torch.zeros(1), torch.zeros(1)),
        get_parameter=lambda name: torch.empty(1),
        _run_causal_conv1d=lambda x, *args, **kwargs: x,
        _run_recurrent=MagicMock(side_effect=recurrent_outputs),
        _run_prefill=MagicMock(return_value=core_non_spec),
        o_norm=SimpleNamespace(forward_oot=norm),
    )
    forward, save = _load_forward_on_cpu({"kda": metadata})
    forward(
        attention,
        torch.zeros(tokens, 3 * heads * dim, dtype=output.dtype),
        torch.zeros(1, tokens, heads, dim, dtype=output.dtype),
        gate,
        torch.zeros(1, tokens, heads),
        output,
        beta_is_preprocessed=True,
    )
    save.assert_called_once_with("", [])
    assert attention._run_prefill.call_count == int(mode == "prefill")
    assert attention._run_recurrent.call_count == {"spec": 1, "decode": 1, "prefill": 0, "mixed": 2, "idle": 0}[mode]


def _cpu_gated_rms_norm(x, gate, weight, activation):
    value = x.float()
    value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
    value = value * weight.float()
    if activation == "swish":
        value = value * gate.float()
    return (value * gate.float().sigmoid()).to(x.dtype)


@pytest.mark.parametrize(
    "mode,tokens", [("decode", 1), ("decode", 96), ("spec", 96), ("spec", 104), ("prefill", 15959)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("activation", ["sigmoid", "swish"])
def test_kda_single_core_norm_reads_source_without_output_assembly(mode, tokens, dtype, activation):
    torch.manual_seed(23)
    heads, dim = 12, 128
    core = torch.randn(1, tokens, heads, dim, dtype=dtype)
    before = core.clone()
    packed_gate = torch.randn(tokens, heads + 2 * heads * dim, dtype=dtype)
    gate = packed_gate[:, heads + heads * dim :].view(tokens, heads, dim)
    weight = torch.randn(dim, dtype=dtype)
    output = torch.full((1, tokens + 7, heads, dim), torch.nan, dtype=dtype)
    output_ptr = output.data_ptr()
    expected = _cpu_gated_rms_norm(core, gate, weight, activation)

    def norm_into(x, g, *, out):
        assert x is core
        assert out.data_ptr() == output_ptr
        # Detect either an unnecessary clear or a copy before the norm.
        assert torch.isnan(out).all()
        assert g.data_ptr() == gate.data_ptr()
        assert g.stride() == gate.stride()
        out.copy_(_cpu_gated_rms_norm(x, g, weight, activation))

    _run_kda_output_case(
        mode,
        gate,
        output,
        norm_into,
        core_spec=core if mode == "spec" else None,
        core_non_spec=core if mode != "spec" else None,
    )
    torch.testing.assert_close(output[:, :tokens], expected, rtol=0, atol=0)
    torch.testing.assert_close(core, before, rtol=0, atol=0)
    assert output.data_ptr() == output_ptr
    assert torch.count_nonzero(output[:, tokens:]) == 0


@pytest.mark.parametrize("mode", ["spec", "decode", "prefill"])
@pytest.mark.parametrize("layout", ["source_stride", "output_stride", "dtype", "broadcast"])
def test_kda_single_core_preserves_copy_before_norm(mode, layout):
    torch.manual_seed(24)
    tokens, heads, dim = 5, 2, 128
    core = torch.randn(1, tokens, heads, dim, dtype=torch.bfloat16)
    output = torch.full((1, tokens + 3, heads, dim), torch.nan, dtype=torch.bfloat16)
    if layout == "source_stride":
        core = torch.randn(1, tokens, heads, dim * 2, dtype=core.dtype)[..., ::2]
    elif layout == "output_stride":
        output = torch.full((1, tokens + 3, heads, dim * 2), torch.nan, dtype=output.dtype)[..., ::2]
    elif layout == "dtype":
        core = torch.randn(1, tokens, heads, dim, dtype=torch.float32)
    else:
        core = core[:, :1]
    before = core.clone()
    gate = torch.randn(tokens, heads, dim, dtype=output.dtype)
    weight = torch.randn(dim)
    # The old copy casts/broadcasts before FP32 normalization.
    assembled = torch.empty_like(output[:, :tokens]).copy_(core)
    expected = _cpu_gated_rms_norm(assembled, gate, weight, "sigmoid")

    def norm_into(x, g, *, out):
        assert x is out
        assert x.dtype == output.dtype
        torch.testing.assert_close(x, assembled, rtol=0, atol=0)
        out.copy_(_cpu_gated_rms_norm(x, g, weight, "sigmoid"))

    _run_kda_output_case(
        mode,
        gate,
        output,
        norm_into,
        core_spec=core if mode == "spec" else None,
        core_non_spec=core if mode != "spec" else None,
    )
    torch.testing.assert_close(output[:, :tokens], expected, rtol=0, atol=0)
    torch.testing.assert_close(core, before, rtol=0, atol=0)
    assert torch.count_nonzero(output[:, tokens:]) == 0


@pytest.mark.parametrize("sparse", [False, True])
def test_kda_mixed_output_keeps_index_order_and_zero_holes(sparse):
    torch.manual_seed(25)
    tokens, heads, dim = 6, 2, 128
    spec_indices = torch.tensor([4, 0] if sparse else [4, 0, 2])
    non_spec_indices = torch.tensor([3, 1] if sparse else [3, 1, 5])
    core_spec = torch.randn(1, len(spec_indices), heads, dim, dtype=torch.bfloat16)
    core_non_spec = torch.randn(1, len(non_spec_indices), heads, dim, dtype=torch.bfloat16)
    gate = torch.randn(tokens, heads, dim, dtype=torch.bfloat16)
    output = torch.full((1, tokens + 2, heads, dim), torch.nan, dtype=torch.bfloat16)
    weight = torch.randn(dim)
    assembled = torch.zeros_like(output[:, :tokens])
    assembled[:, spec_indices] = core_spec
    assembled[:, non_spec_indices] = core_non_spec
    expected = _cpu_gated_rms_norm(assembled, gate, weight, "sigmoid")

    def norm_into(x, g, *, out):
        assert x is out
        torch.testing.assert_close(x, assembled, rtol=0, atol=0)
        out.copy_(_cpu_gated_rms_norm(x, g, weight, "sigmoid"))

    _run_kda_output_case(
        "mixed",
        gate,
        output,
        norm_into,
        core_spec=core_spec,
        core_non_spec=core_non_spec,
        indices=(spec_indices, non_spec_indices),
    )
    torch.testing.assert_close(output[:, :tokens], expected, rtol=0, atol=0)
    assert torch.count_nonzero(output[:, tokens:]) == 0


def test_kda_output_reuse_clears_shrinking_tail_and_idle_batch():
    torch.manual_seed(26)
    output = torch.full((1, 8, 2, 128), torch.nan, dtype=torch.bfloat16)
    weight = torch.randn(128)
    for tokens in (7, 3):
        core = torch.randn(1, tokens, 2, 128, dtype=output.dtype)
        gate = torch.randn(tokens, 2, 128, dtype=output.dtype)
        expected = _cpu_gated_rms_norm(core, gate, weight, "sigmoid")

        def norm_into(x, g, *, out):
            out.copy_(_cpu_gated_rms_norm(x, g, weight, "sigmoid"))

        _run_kda_output_case("decode", gate, output, norm_into, core_non_spec=core)
        torch.testing.assert_close(output[:, :tokens], expected, rtol=0, atol=0)
        assert torch.count_nonzero(output[:, tokens:]) == 0

    norm = MagicMock(side_effect=AssertionError("Idle batches must not run norm"))
    _run_kda_output_case("idle", torch.full((4, 2, 128), torch.nan), output, norm)
    assert torch.count_nonzero(output) == 0


def test_kda_output_without_metadata_is_fully_zeroed():
    forward, save = _load_forward_on_cpu(None)
    output = torch.full((1, 8, 2, 128), torch.nan)
    forward(SimpleNamespace(), torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0), output)
    assert torch.count_nonzero(output) == 0
    save.assert_not_called()
