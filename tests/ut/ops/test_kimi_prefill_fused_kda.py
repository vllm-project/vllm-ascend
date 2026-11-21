# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for FlashKDA dispatch and V2 page ownership.

Production methods run with external kernels and sealed state-copy launches
replaced by CPU doubles. Numerical equivalence belongs to the NPU gate.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from einops import rearrange

from tests.ut.ops import test_kimi_mega_kda as contracts
from tests.ut.worker.test_mega_kda_v2 import _paged_layer


@pytest.fixture
def runtime():
    ns = contracts.runtime.__wrapped__()
    contracts._definitions(
        "vllm_ascend/ops/kimi_prefill_fused_kda.py",
        {
            "PrefillFusedKDAOperators",
            "PrefillKDAMetadataWorkspace",
            "PrefillKDAMetadataContext",
            "create_prefill_kda_metadata_workspace",
            "load_prefill_fused_kda_operator",
            "run_prefill_fused_kda",
        },
        ns,
    )
    ns.update(
        KimiK3DeltaAttention=ns["AscendKimiK3DeltaAttention"],
        rearrange=rearrange,
        run_chunk_kda=MagicMock(),
        run_recurrent_kda=MagicMock(side_effect=lambda q, k, v, *args, **kwargs: v.clone()),
    )
    contracts._definitions("vllm_ascend/ops/kimi_kda.py", {"_prepare_beta"}, ns)
    contracts._definitions(
        "vllm_ascend/ops/kimi_kda.py",
        {"AscendKimiK3DeltaAttention"},
        ns,
        methods={
            "_run_prefill_fused_kda",
            "_forward",
            "_postprocess_bfg",
            "_run_recurrent",
            "_run_prefill",
            "_forward_default",
        },
        bases="KimiK3DeltaAttention",
    )
    return ns


class _StateCopy:
    def __init__(self):
        self.gather = MagicMock(side_effect=self._gather)
        self.scatter = MagicMock(side_effect=self._scatter)

    @staticmethod
    def _gather(state, indices, flags):
        valid = (indices >= 0) & (indices < state.shape[0]) & flags.bool()
        packed = state[indices.clamp(0, state.shape[0] - 1).long()].clone()
        packed[~valid] = 0
        return packed

    @staticmethod
    def _scatter(state, packed, indices):
        valid = (indices >= 0) & (indices < state.shape[0])
        state[indices[valid].long()] = packed[valid].to(state.dtype)


def _install_prefill(layer, ns):
    layer._ascend_kda_state_copy = _StateCopy()
    layer._kda_state_copy_ready = True
    metadata = MagicMock(return_value=object())
    prefill = MagicMock(side_effect=lambda q, k, v, **kwargs: (v.clone(), kwargs["initial_state"] + 3))
    layer._prefill_fused_kda_operator = ns["PrefillFusedKDAOperators"](prefill, metadata)
    return prefill, metadata


def _inputs(tokens=5):
    packed = torch.arange(tokens * 3 * 8, dtype=torch.float32).reshape(1, tokens, 3, 2, 4).to(torch.bfloat16)
    q, k, v = (packed[:, :, index] for index in range(3))
    gate = torch.randn_like(q)
    beta = torch.linspace(-7, 7, tokens * 2).reshape(1, tokens, 2).to(torch.bfloat16)
    return q, k, v, gate, beta


@pytest.mark.parametrize("missing", ["prefill", "metadata", "attribute", None])
def test_loader_requires_both_reference_exports(runtime, missing):
    prefill, metadata = MagicMock(), MagicMock()

    def import_module(name):
        if missing == "prefill" and name == "ops.flash_kda":
            raise ModuleNotFoundError(name)
        if missing == "metadata" and name == "ops.flash_kda_metadata":
            raise ImportError(name)
        if missing == "attribute":
            return SimpleNamespace()
        return SimpleNamespace(flash_kda=prefill, flash_kda_metadata=metadata)

    runtime["import_module"].side_effect = import_module
    result = runtime["load_prefill_fused_kda_operator"]()
    if missing:
        assert result is None
    else:
        assert result.prefill is prefill and result.metadata is metadata
    prefill.assert_not_called()
    metadata.assert_not_called()


@pytest.mark.parametrize("export", ["flash_kda", "flash_kda_metadata"])
def test_loader_rejects_non_callable_exports(runtime, export):
    exports = {"flash_kda": MagicMock(), "flash_kda_metadata": MagicMock()}
    exports[export] = None
    runtime["import_module"].return_value = SimpleNamespace(**exports)
    with pytest.raises(TypeError, match="must be callable"):
        runtime["load_prefill_fused_kda_operator"]()


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_tnd_prefill_gathers_clears_and_scatters_only_selected_pages(runtime, compact, state_dtype):
    layer, backing, page_bytes = _paged_layer(runtime)
    if state_dtype == torch.bfloat16:
        # A separate dtype contract; production Kimi caches normally use FP32.
        layer.kv_cache = (layer.kv_cache[0], layer.kv_cache[1].to(state_dtype))
    prefill, metadata_op = _install_prefill(layer, runtime)
    state = layer.kv_cache[1]
    state.fill_(9)
    # Asymmetry catches accidental VK/KV transposes even though K == V.
    state[5].copy_(torch.arange(32).reshape(2, 4, 4))
    before = backing.clone()
    original = state.clone()
    layout = state.data_ptr(), state.stride(), state.storage_offset()
    indices = torch.tensor([5, 2, 8], dtype=torch.int64) if compact else torch.tensor([5, 2], dtype=torch.int32)
    flags = torch.tensor([True, True, False]) if compact else torch.tensor([True, False])
    starts = torch.tensor([0, 2, 2, 5] if compact else [0, 2, 5], dtype=torch.int32)
    chunk = SimpleNamespace(keep_meta=torch.tensor([True, False, True]) if compact else None)
    args = _inputs()
    output = layer._run_prefill_fused_kda(*args, state, indices, flags, chunk, query_start_loc=starts)
    torch.testing.assert_close(output, args[2])
    call = prefill.call_args
    assert call.kwargs["layout_qkv"] == "TND"
    assert call.kwargs["lower_bound"] == -5.0
    assert call.kwargs["scale"] == layer.head_dim**-0.5
    for actual, expected in zip((*call.args, call.kwargs["g"], call.kwargs["beta"]), args):
        assert actual.is_contiguous()
        torch.testing.assert_close(actual, expected.squeeze(0))
    initial = call.kwargs["initial_state"]
    assert initial.dtype == torch.float32
    torch.testing.assert_close(initial[0], original[5].float())
    assert torch.count_nonzero(initial[1]) == 0
    offsets = metadata_op.call_args.kwargs["cu_seqlens"]
    assert offsets.dtype == torch.int32 and offsets.tolist() == [0, 2, 5]
    if not compact:
        assert offsets is starts
    selected = [5, 8] if compact else [5, 2]
    expected = original.clone()
    expected[selected[0]] = (original[5].float() + 3).to(state.dtype)
    expected[selected[1]] = 3
    torch.testing.assert_close(state, expected)
    assert (state.data_ptr(), state.stride(), state.storage_offset()) == layout
    if state_dtype == torch.float32:
        expected_backing = before.clone()
        conv_bytes = layer.kv_cache[0][0].numel() * 2
        for slot in selected:
            offset = 32 + slot * page_bytes + conv_bytes
            payload = expected[slot].contiguous().view(torch.int8).reshape(-1)
            expected_backing[offset : offset + payload.numel()] = payload
        torch.testing.assert_close(backing, expected_backing, rtol=0, atol=0)


@pytest.mark.parametrize("failure", ["kernel", "output", "state", "state_dtype"])
def test_failure_never_publishes_bad_ssm_state_or_retries(runtime, failure):
    layer, backing, _ = _paged_layer(runtime)
    prefill, _ = _install_prefill(layer, runtime)
    before = backing.clone()

    def fail(q, k, v, **kwargs):
        kwargs["initial_state"].fill_(123)
        if failure == "kernel":
            raise RuntimeError("kernel failed")
        if failure == "state_dtype":
            return v, kwargs["initial_state"].to(torch.bfloat16)
        return (v[:1], kwargs["initial_state"]) if failure == "output" else (v, kwargs["initial_state"][:0])

    prefill.side_effect = fail
    with pytest.raises(RuntimeError):
        layer._run_prefill_fused_kda(
            *_inputs(),
            layer.kv_cache[1],
            torch.tensor([2, 5]),
            torch.tensor([True, False]),
            SimpleNamespace(keep_meta=None),
            query_start_loc=torch.tensor([0, 2, 5]),
        )
    layer._ascend_kda_state_copy.scatter.assert_not_called()
    runtime["run_chunk_kda"].assert_not_called()
    torch.testing.assert_close(backing, before, rtol=0, atol=0)


@pytest.mark.parametrize("invalid", ["flags", "offsets", "missing_offsets", "unprepared"])
def test_prefill_rejects_invalid_metadata_before_operator(runtime, invalid):
    layer, _, _ = _paged_layer(runtime)
    prefill, _ = _install_prefill(layer, runtime)
    flags = None if invalid == "flags" else torch.ones(2, dtype=torch.bool)
    starts = torch.tensor([0, 5]) if invalid == "offsets" else torch.tensor([0, 2, 5])
    if invalid == "missing_offsets":
        starts = None
    layer._kda_state_copy_ready = invalid != "unprepared"
    with pytest.raises((ValueError, RuntimeError)):
        layer._run_prefill_fused_kda(
            *_inputs(),
            layer.kv_cache[1],
            torch.tensor([2, 5]),
            flags,
            SimpleNamespace(keep_meta=None),
            query_start_loc=starts,
        )
    prefill.assert_not_called()
    layer._ascend_kda_state_copy.gather.assert_not_called()


def test_empty_segments_do_not_launch_or_overwrite_state(runtime):
    layer, backing, _ = _paged_layer(runtime)
    prefill, _ = _install_prefill(layer, runtime)
    before = backing.clone()
    output = layer._run_prefill_fused_kda(
        *_inputs(0),
        layer.kv_cache[1],
        torch.tensor([2, 5]),
        torch.ones(2, dtype=torch.bool),
        SimpleNamespace(keep_meta=torch.zeros(2, dtype=torch.bool)),
        query_start_loc=torch.zeros(3, dtype=torch.int32),
    )
    assert output.shape == (1, 0, 2, 4)
    prefill.assert_not_called()
    layer._ascend_kda_state_copy.scatter.assert_not_called()
    torch.testing.assert_close(backing, before, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["prefill", "decode_prefill", "spec_prefill"])
@pytest.mark.parametrize("path", ["fused", "missing", "preprocessed", "dtype", "unbounded"])
def test_forward_dispatches_raw_prefill_beta_and_preprocessed_decode_beta(runtime, mode, path):
    layer = contracts._attention(runtime)
    _install_prefill(layer, runtime)
    fused = path == "fused"
    if path == "missing":
        layer._prefill_fused_kda_operator = None
    if path == "unbounded":
        layer.gate_lower_bound = None
    tokens = 7
    q, k, v, gate, raw_beta = _inputs(tokens)
    if path == "dtype":
        q, k, v, gate, raw_beta = (tensor.to(torch.float16) for tensor in (q, k, v, gate, raw_beta))
    mixed_qkv = torch.cat((q, k, v), dim=-2).reshape(tokens, -1)
    spec = mode == "spec_prefill"
    decode = mode == "decode_prefill"
    # Spec tokens interleave with prefill to exercise token index selection.
    non_spec = torch.tensor([0, 2, 3, 4, 6]) if spec else None
    spec_indices = torch.tensor([1, 5]) if spec else None
    prefill_starts = torch.tensor([0, 2, 5]) if mode != "prefill" else torch.tensor([0, 3, 7])
    conv = SimpleNamespace(
        query_start_loc=prefill_starts,
        cache_indices=torch.tensor([5, 2]),
        initial_state_mode=torch.tensor([True, False]),
    )
    spec_conv = SimpleNamespace(
        query_start_loc=torch.tensor([0, 2]), cache_indices=torch.tensor([9]), num_accepted_tokens=torch.tensor([1])
    )
    md = SimpleNamespace(
        num_actual_tokens=tokens,
        num_prefills=2,
        num_decodes=int(decode),
        num_decode_tokens=2 if decode else 0,
        num_spec_decodes=int(spec),
        spec_sequence_masks=torch.tensor([True, False, False]) if spec else None,
        spec_token_indx=spec_indices,
        non_spec_token_indx=non_spec,
        spec_decode_metadata=SimpleNamespace(spec_causal_conv1d=spec_conv),
        spec_query_start_loc=spec_conv.query_start_loc,
        spec_state_indices_tensor=torch.tensor([[9, 10]]),
        non_spec_query_start_loc=torch.tensor([0, 2, 4, 7]) if decode else prefill_starts,
        non_spec_state_indices_tensor=torch.tensor([9, 5, 2]) if decode else torch.tensor([5, 2]),
        prefill_query_start_loc=prefill_starts,
        prefill_state_indices=torch.tensor([5, 2]),
        prefill_has_initial_state=torch.tensor([True, False]),
        non_spec_prefill_metadata=SimpleNamespace(
            causal_conv1d=conv,
            chunk=SimpleNamespace(
                keep_meta=None,
                cu_seqlens_host=tuple(prefill_starts.tolist()),
                cu_seqlens_kern=None,
                chunk_indices_chunk64_host=(0, 0, 1, 0),
            ),
        ),
    )
    trace = []
    runtime["wait_for_kv_layer_from_connector"].side_effect = lambda *args: trace.append("wait")
    runtime["maybe_save_kv_layer_to_connector"].side_effect = lambda *args: trace.append("save")
    layer._run_causal_conv1d = MagicMock(side_effect=lambda x, *args, **kwargs: trace.append("conv") or x)
    layer.o_norm.forward_oot = lambda x, g, out: out.copy_(x)
    prefill = layer._prefill_fused_kda_operator
    if prefill is not None:
        prefill.prefill.side_effect = lambda q, k, v, **kwargs: (
            trace.append("prefill") or v,
            kwargs["initial_state"] + 3,
        )
    runtime["run_chunk_kda"].side_effect = lambda q, k, v, *args, **kwargs: (v, torch.full((2, 2, 4, 4), 3.0))
    runtime["get_forward_context"].return_value = SimpleNamespace(attn_metadata={layer.prefix: md})
    out = torch.full((1, tokens + 2, 2, 4), float("nan"), dtype=v.dtype)
    input_beta = raw_beta.float().sigmoid() if path == "preprocessed" else raw_beta
    layer._forward(
        mixed_qkv,
        gate,
        torch.zeros(tokens, 2, 4),
        input_beta,
        out,
        beta_is_preprocessed=path == "preprocessed",
    )
    torch.testing.assert_close(out[:, :tokens], v)
    assert torch.count_nonzero(out[:, tokens:]) == 0
    expected_beta = raw_beta[:, non_spec] if spec else raw_beta[:, 2:] if decode else raw_beta
    if fused:
        torch.testing.assert_close(prefill.prefill.call_args.kwargs["beta"], expected_beta.squeeze(0))
        runtime["run_chunk_kda"].assert_not_called()
        assert trace.index("wait") < trace.index("conv") < trace.index("prefill") < trace.index("save")
    else:
        torch.testing.assert_close(runtime["run_chunk_kda"].call_args.args[4], expected_beta.float().sigmoid())
        if prefill is not None:
            prefill.prefill.assert_not_called()
    if mode != "prefill":
        expected_decode = raw_beta[:, spec_indices] if spec else raw_beta[:, :2]
        torch.testing.assert_close(runtime["run_recurrent_kda"].call_args.args[4], expected_decode.float().sigmoid())
    layer._ascend_kda_state_copy.scatter.assert_called_once()


def test_mixed_projection_preserves_logits_for_fused_prefill(runtime):
    layer = contracts._attention(runtime)
    _install_prefill(layer, runtime)
    layer._fused_bfg_output_sizes = (2, 8, 8)
    projected = torch.randn(5, 18, dtype=torch.bfloat16)
    beta, _, _ = layer._postprocess_bfg(projected)
    torch.testing.assert_close(beta, projected[:, :2].unsqueeze(0))
    layer._prefill_fused_kda_operator = None
    processed, _, _ = layer._postprocess_bfg(projected)
    torch.testing.assert_close(processed, beta.float().sigmoid())


@pytest.mark.parametrize("fused", [False, True])
def test_mixed_projection_passes_beta_activation_contract_to_forward(runtime, fused):
    layer = contracts._attention(runtime)
    _install_prefill(layer, runtime)
    if not fused:
        layer._prefill_fused_kda_operator = None
    layer.uses_mixed_projection = True
    q, k, v, gate, beta = _inputs()
    layer._run_overlapped_qkv_bfg = MagicMock(return_value=(q, beta, gate, gate.squeeze(0)))
    layer._forward = MagicMock()
    layer.o_proj = lambda value: (value, None)
    runtime["AscendKimiK3DeltaAttention"]._forward_default(layer, q.squeeze(0), torch.arange(5))
    assert layer._forward.call_args.kwargs["beta_is_preprocessed"] is (not fused)


def test_prefill_then_mega_verify_reads_published_original_state(runtime):
    layer, _, _ = _paged_layer(runtime)
    _install_prefill(layer, runtime)
    state = layer.kv_cache[1]
    indices = torch.tensor([1, 9, 17, 25], dtype=torch.int32)
    starts = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32)
    state[indices.long()] = 42
    layer._run_prefill_fused_kda(
        *_inputs(4),
        state,
        indices,
        torch.zeros(4, dtype=torch.bool),
        SimpleNamespace(keep_meta=None),
        query_start_loc=starts,
    )
    metadata = contracts._metadata()
    plan, hidden = contracts._plan(runtime, layer, metadata)
    assert plan.conv_num_accepted_tokens.tolist() == [8] * 4

    def mega(*args):
        assert args[10] is state
        torch.testing.assert_close(args[10][args[11].long()], torch.full((4, 2, 4, 4), 3.0))
        return torch.zeros_like(hidden)

    layer._mega_operator.side_effect = mega
    layer._execute_mega_kda(hidden, torch.arange(32), plan)
    layer._mega_operator.assert_called_once()
