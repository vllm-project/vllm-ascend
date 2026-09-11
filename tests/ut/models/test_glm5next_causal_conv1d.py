# SPDX-License-Identifier: Apache-2.0
"""The GLM-5.3-Flash KDA convolution entry points.

The decode / draft-verify update has to advance every request's conv state
without reading a device tensor on the host: a host sync is rejected outright
while an ACL graph is being captured, so one ``.item()`` aborts decode-FULL
capture. Where the fused Ascend operator is unavailable -- A5 withholds
``RUNTIME_CUSTOM_OPS``, so custom ops are off there entirely -- a batched torch
expression stands in for it, and the bulk of this file checks that expression
against the per-request implementation it replaces, since the two have to agree
on where a verify step rewinds to and on which requests are inert.

The rest pins what the fused operator is handed, because its layouts differ
from the fallback's in every argument.
"""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.models.glm5next.ops import causal_conv1d as conv1d_module
from vllm_ascend.models.glm5next.ops.causal_conv1d import (
    _ACTIVATION_SILU,
    _RUN_MODE_UPDATE,
    _RUN_MODE_VARLEN,
    _batched_causal_conv1d_update,
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm_ascend.ops.causal_conv1d import (
    causal_conv1d_update as per_request_causal_conv1d_update,
)

CONV_WIDTH = 4
MERGED_DIM = 24
NUM_SLOTS = 8


@pytest.fixture
def fused_op_calls(monkeypatch) -> list[dict]:
    """Record calls to a stand-in for the fused Ascend operator."""
    calls: list[dict] = []

    def npu_causal_conv1d_custom(output, x, weight, **kwargs):
        calls.append({"output": output, "x": x, "weight": weight, **kwargs})
        return output

    monkeypatch.setattr(
        torch.ops,
        "_C_ascend",
        types.SimpleNamespace(npu_causal_conv1d_custom=npu_causal_conv1d_custom),
        raising=False,
    )
    monkeypatch.setattr(conv1d_module, "has_fused_conv1d", lambda: True)
    return calls


@pytest.fixture
def without_fused_op(monkeypatch) -> None:
    monkeypatch.setattr(conv1d_module, "has_fused_conv1d", lambda: False)


# (query_lens, num_accepted_tokens, num_spec)
UPDATE_CASES = {
    # Plain decode: one token per request.
    "decode": ([1, 1, 1, 1], None, 0),
    # Requests the metadata builder padded away carry zero-length ranges and
    # all point at the same null block, so they must stay inert.
    "decode-padded": ([1, 1, 0, 0], None, 0),
    "verify-all-accepted": ([4, 4, 4], [4, 4, 4], 3),
    # The count is what the step before accepted, so it names the window this
    # step rewinds to rather than how much of this step survives. Rejection
    # sampling always keeps a token, so one is the floor.
    "verify-rejections": ([4, 4, 4], [1, 3, 2], 3),
    "verify-padded": ([4, 4, 0, 0], [2, 4, 1, 1], 3),
}


def _update_inputs(query_lens, num_accepted, num_spec, *, state_dim_first, dtype):
    state_len = CONV_WIDTH - 1
    num_requests = len(query_lens)
    num_tokens = sum(query_lens)
    is_verify = num_accepted is not None

    x = torch.randn(num_tokens, MERGED_DIM, dtype=dtype)
    # The kernel layout is [width, dim]; vLLM keeps the checkpoint copy in fp32.
    weight = torch.randn(CONV_WIDTH, MERGED_DIM, dtype=torch.float32)
    query_start_loc = torch.tensor(
        [0, *torch.cumsum(torch.tensor(query_lens), 0).tolist()],
        dtype=torch.int32,
    )
    state_slots = torch.arange(1, num_requests + 1, dtype=torch.int32) % NUM_SLOTS
    # On the speculative path the state index tensor carries one column per
    # draft slot, and only the first column names the conv state.
    max_query_len = num_spec + 1 if is_verify else 1
    cache_indices = (
        state_slots.unsqueeze(1).expand(num_requests, max_query_len).contiguous() if is_verify else state_slots
    )

    # The cache is allocated state_len + num_spec wide to leave room for the
    # draft slots, and can be oriented either way round.
    allocated = state_len + num_spec
    per_slot = (MERGED_DIM, allocated) if state_dim_first else (allocated, MERGED_DIM)
    conv_state = torch.randn(NUM_SLOTS, *per_slot, dtype=dtype)

    return {
        "x": x,
        "weight": weight,
        "conv_state": conv_state,
        "query_start_loc": query_start_loc,
        "state_slots": state_slots,
        "cache_indices": cache_indices,
        "num_accepted_tokens": (torch.tensor(num_accepted, dtype=torch.int32) if is_verify else None),
    }


@pytest.mark.parametrize("case", UPDATE_CASES.keys())
@pytest.mark.parametrize("state_dim_first", [False, True], ids=["state-major", "dim-major"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_batched_update_matches_the_per_request_implementation(case, state_dim_first, dtype) -> None:
    torch.manual_seed(0)
    inputs = _update_inputs(*UPDATE_CASES[case], state_dim_first=state_dim_first, dtype=dtype)

    expected_state = inputs["conv_state"].clone()
    expected = per_request_causal_conv1d_update(
        inputs["x"].clone(),
        expected_state,
        # The per-request implementation takes the transposed weight, and
        # convolves in the weight dtype.
        inputs["weight"].transpose(0, 1),
        None,
        activation="silu",
        conv_state_indices=inputs["state_slots"],
        num_accepted_tokens=inputs["num_accepted_tokens"],
        query_start_loc=inputs["query_start_loc"],
    )

    actual_state = inputs["conv_state"].clone()
    actual = _batched_causal_conv1d_update(
        inputs["x"].clone(),
        actual_state,
        inputs["weight"],
        None,
        query_start_loc=inputs["query_start_loc"],
        cache_indices=inputs["cache_indices"],
        num_accepted_tokens=inputs["num_accepted_tokens"],
    )

    tolerance = 3e-3 if dtype is torch.bfloat16 else 1e-5
    torch.testing.assert_close(actual, expected, rtol=0, atol=tolerance)
    torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=tolerance)


def test_batched_update_leaves_untouched_slots_alone(without_fused_op) -> None:
    """A slot moves only for a request that brought tokens.

    The columns past state_len are deliberately not pinned: they carry the
    windows a verify step leaves for the step after it, one per count the
    sampler could accept, so a request with tokens has to rewrite them.
    """
    torch.manual_seed(0)
    inputs = _update_inputs(*UPDATE_CASES["verify-padded"], state_dim_first=False, dtype=torch.float32)
    conv_state = inputs["conv_state"].clone()
    before = conv_state.clone()

    causal_conv1d_update(
        inputs["x"],
        conv_state,
        inputs["weight"],
        None,
        query_start_loc=inputs["query_start_loc"],
        cache_indices=inputs["cache_indices"],
        num_accepted_tokens=inputs["num_accepted_tokens"],
    )

    query_lens = (inputs["query_start_loc"][1:] - inputs["query_start_loc"][:-1]).tolist()
    slots = inputs["state_slots"].tolist()
    advanced = {slot for slot, length in zip(slots, query_lens) if length > 0}
    assert any(length == 0 for length in query_lens), "the fixture should pad some requests away"
    # Both the slots no request named and the ones the builder padded away.
    inert = sorted(set(range(NUM_SLOTS)) - advanced)
    torch.testing.assert_close(conv_state[inert], before[inert])


def test_batched_update_is_free_of_host_reads(monkeypatch, without_fused_op) -> None:
    """A single host read here would abort ACL graph capture."""
    torch.manual_seed(0)
    inputs = _update_inputs(*UPDATE_CASES["verify-rejections"], state_dim_first=False, dtype=torch.float32)

    def forbidden(self, *args, **kwargs):
        raise AssertionError("the batched update must not sync the host")

    for method in ("item", "tolist"):
        monkeypatch.setattr(torch.Tensor, method, forbidden, raising=True)

    causal_conv1d_update(
        inputs["x"],
        inputs["conv_state"].clone(),
        inputs["weight"],
        None,
        query_start_loc=inputs["query_start_loc"],
        cache_indices=inputs["cache_indices"],
        num_accepted_tokens=inputs["num_accepted_tokens"],
    )


def _fused_inputs() -> dict:
    num_tokens = 6
    return {
        "x": torch.randn(num_tokens, MERGED_DIM),
        "conv_state": torch.zeros(NUM_SLOTS, CONV_WIDTH - 1, MERGED_DIM),
        "weight": torch.randn(CONV_WIDTH, MERGED_DIM),
        "query_start_loc": torch.tensor([0, 3, num_tokens], dtype=torch.int32),
    }


def test_update_hands_the_fused_operator_its_own_layouts(fused_op_calls) -> None:
    inputs = _fused_inputs()
    cache_indices = torch.arange(2 * 4, dtype=torch.int32).reshape(2, 4)
    num_accepted_tokens = torch.tensor([2, 1], dtype=torch.int32)

    out = causal_conv1d_update(
        inputs["x"],
        inputs["conv_state"],
        inputs["weight"],
        None,
        query_start_loc=inputs["query_start_loc"],
        cache_indices=cache_indices,
        num_accepted_tokens=num_accepted_tokens,
    )

    assert len(fused_op_calls) == 1
    call = fused_op_calls[0]
    # The operator writes through its declared output alias, so the wrapper has
    # to return that tensor rather than an independent allocation.
    assert out is call["output"]
    assert out.shape == inputs["x"].shape
    assert call["weight"].shape == (CONV_WIDTH, MERGED_DIM)
    assert call["conv_state"] is inputs["conv_state"]
    assert call["query_start_loc_opt"] is inputs["query_start_loc"]
    # Narrowing to the first column here would drop the draft slots the
    # operator slides the state across.
    assert torch.equal(call["cache_indices_opt"], cache_indices)
    assert torch.equal(call["num_accepted_tokens_opt"], num_accepted_tokens)
    assert call["initial_state_mode_opt"] is None
    assert call["run_mode"] == _RUN_MODE_UPDATE
    assert call["activation_mode"] == _ACTIVATION_SILU


def test_varlen_hands_the_fused_operator_token_major_input(fused_op_calls) -> None:
    inputs = _fused_inputs()
    initial_state_mode = torch.tensor([False, True])

    out = causal_conv1d_fn(
        inputs["x"],
        inputs["conv_state"],
        inputs["weight"],
        None,
        query_start_loc=inputs["query_start_loc"],
        cache_indices=torch.tensor([0, 1], dtype=torch.int32),
        initial_state_mode=initial_state_mode,
    )

    assert len(fused_op_calls) == 1
    call = fused_op_calls[0]
    assert out is call["output"]
    # Token-major, unlike the fallback, which wants [dim, num_tokens].
    assert call["x"].shape == inputs["x"].shape
    assert call["weight"].shape == (CONV_WIDTH, MERGED_DIM)
    assert call["initial_state_mode_opt"] is initial_state_mode
    assert call["num_accepted_tokens_opt"] is None
    assert call["run_mode"] == _RUN_MODE_VARLEN


def test_varlen_fallback_uses_channel_major_tokens(monkeypatch, without_fused_op) -> None:
    inputs = _fused_inputs()
    captured: dict = {}

    def fake_fn(x, weight, bias, **kwargs):
        captured.update(x=x, weight=weight, **kwargs)
        return torch.zeros(MERGED_DIM, x.shape[-1])

    monkeypatch.setattr(conv1d_module, "_torch_causal_conv1d_fn", fake_fn)

    out = causal_conv1d_fn(
        inputs["x"],
        inputs["conv_state"],
        inputs["weight"],
        None,
        query_start_loc=inputs["query_start_loc"],
        cache_indices=torch.tensor([0, 1], dtype=torch.int32),
        initial_state_mode=torch.tensor([False, True]),
    )

    assert captured["x"].shape == (MERGED_DIM, inputs["x"].shape[0])
    assert captured["weight"].shape == (MERGED_DIM, CONV_WIDTH)
    # The reference convolves in the weight dtype, so it has to stay fp32.
    assert captured["weight"].dtype == torch.float32
    # Callers get token-major output back either way.
    assert out.shape == inputs["x"].shape


@pytest.mark.parametrize(
    ("fused_available", "expected_dtype"),
    # The fused operator wants the activation dtype; the fallback convolves in
    # the weight dtype, so it keeps the checkpoint's fp32.
    [(True, torch.bfloat16), (False, torch.float32)],
    ids=["fused", "fallback"],
)
def test_packed_conv_weight_keeps_qkv_channel_order(monkeypatch, fused_available, expected_dtype) -> None:
    from vllm_ascend.models.glm5next import kda as kda_module

    monkeypatch.setattr(kda_module, "has_fused_conv1d", lambda: fused_available)

    channels = MERGED_DIM // 3
    # vLLM stores each conv weight checkpoint-compatible as fp32 [C, 1, width].
    weights = [torch.randn(channels, 1, CONV_WIDTH) for _ in range(3)]
    layer = object.__new__(kda_module.Glm5NextLinearAttention)
    layer.q_conv1d, layer.k_conv1d, layer.v_conv1d = (types.SimpleNamespace(weight=weight) for weight in weights)
    layer.model_config = types.SimpleNamespace(dtype=torch.bfloat16)
    layer._packed_conv_weight = None

    layer._pack_conv_weight()

    packed = layer._packed_conv_weight
    assert packed is not None
    assert packed.shape == (CONV_WIDTH, MERGED_DIM)
    assert packed.dtype == expected_dtype
    assert packed.is_contiguous()
    expected = torch.cat([weight.view(channels, CONV_WIDTH) for weight in weights], dim=0)
    torch.testing.assert_close(packed, expected.transpose(0, 1).to(expected_dtype))


def _reference_verify_step(inputs, request: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Convolve one request's rows from the window its acceptance count names.

    Written out rather than delegated, so that the implementations are checked
    against the operator's definition and not only against each other.
    """
    state_len = CONV_WIDTH - 1
    slot = int(inputs["state_slots"][request])
    read_at = int(inputs["num_accepted_tokens"][request]) - 1
    start = int(inputs["query_start_loc"][request])
    end = int(inputs["query_start_loc"][request + 1])
    history = torch.cat(
        [inputs["conv_state"][slot, read_at : read_at + state_len], inputs["x"][start:end]],
        dim=0,
    ).float()
    weight = inputs["weight"].float()
    rows = end - start
    return F.silu(sum(history[tap : tap + rows] * weight[tap] for tap in range(CONV_WIDTH))), history


def test_verify_step_convolves_every_row_from_the_accepted_window(without_fused_op) -> None:
    """Every row of a verify step is convolved, and the state stays rewindable.

    ``num_accepted_tokens`` describes the step before, so reading it as a count
    of this step's tokens dropped every row past the accepted prefix -- seven of
    eight on a steady decode -- and left them carrying raw projections, which no
    draft could match.
    """
    torch.manual_seed(0)
    inputs = _update_inputs(*UPDATE_CASES["verify-rejections"], state_dim_first=False, dtype=torch.float32)
    conv_state = inputs["conv_state"].clone()

    actual = causal_conv1d_update(
        inputs["x"].clone(),
        conv_state,
        inputs["weight"],
        None,
        query_start_loc=inputs["query_start_loc"],
        cache_indices=inputs["cache_indices"],
        num_accepted_tokens=inputs["num_accepted_tokens"],
    )

    state_len = CONV_WIDTH - 1
    for request in range(len(inputs["state_slots"])):
        start = int(inputs["query_start_loc"][request])
        end = int(inputs["query_start_loc"][request + 1])
        expected, history = _reference_verify_step(inputs, request)
        torch.testing.assert_close(actual[start:end], expected, rtol=0, atol=1e-5)
        # A row must not come through as the projection it arrived as.
        assert not torch.allclose(actual[start:end], inputs["x"][start:end])
        slot = int(inputs["state_slots"][request])
        for accepted in range(1, end - start + 1):
            torch.testing.assert_close(
                conv_state[slot, accepted - 1 : accepted - 1 + state_len],
                history[accepted : accepted + state_len],
                rtol=0,
                atol=1e-5,
            )
