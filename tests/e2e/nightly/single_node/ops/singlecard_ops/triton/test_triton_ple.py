# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""NPU comparisons against upstream PLE, including state writeback."""

from itertools import accumulate

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.models.qwen4_exp.common.ops.ple import ple_conv as upstream_ple_conv
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

from vllm_ascend.models.qwen4_exp import ple as ple_adapter
from vllm_ascend.models.qwen4_exp.ple import _ple_conv_ascend, patch_upstream_ple_short_conv


def _check_output_accuracy(expected: torch.Tensor, actual: torch.Tensor) -> tuple[float, float]:
    atol = rtol = 2**-9 if expected.dtype == torch.float16 else 2**-6
    max_abs_limit = 0.1 if expected.dtype == torch.float16 else 1.0
    expected_fp32 = expected.float()
    abs_error = (expected_fp32 - actual.float()).abs()
    within_tolerance = abs_error <= atol + rtol * expected_fp32.abs()
    pass_ratio = torch.count_nonzero(within_tolerance).item() / expected.numel()
    max_abs = abs_error.max().item()
    assert pass_ratio >= 0.99, f"only {pass_ratio:.6%} of elements meet the mixed tolerance"
    assert max_abs <= max_abs_limit, f"max absolute error {max_abs} exceeds {max_abs_limit}"
    return pass_ratio, max_abs


@pytest.mark.parametrize("dtype,max_abs_limit", [(torch.float16, 0.1), (torch.bfloat16, 1.0)])
def test_output_accuracy_uses_mixed_tolerance(dtype, max_abs_limit):
    expected = torch.zeros(100, dtype=dtype)
    actual = expected.clone()
    actual[0] = 0.02
    pass_ratio, _ = _check_output_accuracy(expected, actual)
    assert pass_ratio == 0.99

    actual[1] = 0.02
    with pytest.raises(AssertionError, match="mixed tolerance"):
        _check_output_accuracy(expected, actual)

    actual[1] = 0
    actual[0] = max_abs_limit * 2
    with pytest.raises(AssertionError, match="max absolute error"):
        _check_output_accuracy(expected, actual)


def test_model_dispatch_uses_ascend_decode_and_upstream_spec(monkeypatch):
    from vllm.models.qwen4_exp.amd import ple_layer

    patch_upstream_ple_short_conv()
    assert ple_layer.ple_conv is _ple_conv_ascend
    calls = []
    monkeypatch.setattr(ple_adapter, "ple_conv_prefill", lambda *args, **kwargs: calls.append("ascend_prefill"))
    monkeypatch.setattr(ple_adapter, "ple_conv_decode", lambda *args, **kwargs: calls.append("decode"))
    monkeypatch.setattr(ple_adapter, "upstream_ple_conv", lambda *args, **kwargs: calls.append(kwargs["mode"]))
    for mode in ("prefill", "decode", "spec"):
        ple_layer.ple_conv(mode=mode)
    assert calls == ["ascend_prefill", "decode", "spec"]


def test_decode_batch_sizes():
    torch.manual_seed(104)
    channels, dilation = 10240, 3
    for tokens in (1, 10, 50, 100, 104, 200):
        for dtype in (torch.bfloat16, torch.float16):
            inputs = torch.randn(tokens, channels, device="npu", dtype=dtype)
            weights = torch.randn(channels, 4, device="npu", dtype=dtype)
            state = torch.randn(tokens + 1, 12, channels, device="npu", dtype=dtype).transpose(1, 2)
            residual = torch.randn_like(inputs)
            outer_residual = torch.randn_like(inputs)
            kwargs = dict(
                inputs=inputs,
                conv_weights=weights,
                state_indices=torch.arange(1, tokens + 1, device="npu", dtype=torch.int32),
                outer_residual=outer_residual,
                mode="decode",
                dilation=dilation,
                has_initial_states=None if tokens == 1 else torch.ones(tokens, device="npu", dtype=torch.bool),
            )
            expected, actual = residual.clone(), residual.clone()
            expected_state, actual_state = state.clone(), state.clone()
            upstream_ple_conv(**kwargs, residual=expected, conv_state=expected_state)
            _ple_conv_ascend(**kwargs, residual=actual, conv_state=actual_state)
            torch.npu.synchronize()
            assert torch.equal(expected, actual)
            assert torch.equal(expected_state, actual_state)


def test_modes_and_token_mapping():
    torch.manual_seed(29)
    for mode in ("decode", "spec", "prefill"):
        for dtype in (torch.bfloat16, torch.float16):
            for outer_enabled in (False, True):
                requests = 14
                tokens = requests if mode == "decode" else (3 * requests if mode == "spec" else 64)
                channels, dilation, spec_len = 10240, 3, 3
                state_width = 12
                x = torch.randn(tokens, channels, device="npu", dtype=dtype)
                weights = torch.randn(channels, 4, device="npu", dtype=dtype)
                state = torch.randn(requests + 1, channels, state_width, device="npu", dtype=dtype)
                residual = torch.randn_like(x)
                outer = torch.randn_like(x) if outer_enabled else None
                slots = torch.arange(1, requests + 1, device="npu", dtype=torch.int32)
                slots[-1] = NULL_BLOCK_ID
                if mode == "decode":
                    qsl = None
                elif mode == "spec":
                    qsl = torch.arange(0, tokens + 1, 3, device="npu", dtype=torch.int32)
                else:
                    lengths = [64 // requests] * requests
                    for request in range(64 % requests):
                        lengths[request] += 1
                    qsl = torch.tensor([0, *accumulate(lengths)], device="npu", dtype=torch.int32)
                kwargs = dict(
                    inputs=x,
                    conv_weights=weights,
                    state_indices=slots,
                    outer_residual=outer,
                    mode=mode,
                    dilation=dilation,
                    query_start_loc=qsl,
                    num_accepted_tokens=(
                        torch.tensor([r % spec_len + 1 for r in range(requests)], device="npu", dtype=torch.int32)
                        if mode == "spec"
                        else None
                    ),
                    has_initial_states=(
                        torch.tensor([r % 2 == 0 for r in range(requests)], device="npu") if mode != "spec" else None
                    ),
                    spec_query_len=spec_len,
                    token_indices=torch.arange(tokens - 1, -1, -1, device="npu", dtype=torch.int32),
                )
                if mode == "prefill":
                    kwargs.pop("num_accepted_tokens")
                    kwargs.pop("spec_query_len")
                expected, actual = residual.clone(), residual.clone()
                expected_state, actual_state = state.clone(), state.clone()
                upstream_ple_conv(**kwargs, residual=expected, conv_state=expected_state)
                _ple_conv_ascend(**kwargs, residual=actual, conv_state=actual_state)
                torch.npu.synchronize()
                pass_ratio, max_abs = _check_output_accuracy(expected, actual)
                state_exact = torch.equal(expected_state, actual_state)
                print(
                    f"CASE mode={mode} dtype={dtype} outer={outer_enabled} "
                    f"pass_ratio={pass_ratio:.6%} max_abs={max_abs:.6f} state_exact={state_exact}",
                    flush=True,
                )
                assert state_exact


def test_prefill_one_request():
    torch.manual_seed(37)
    for dtype in (torch.bfloat16, torch.float16):
        for tokens in (256, 8192):
            channels = 10240
            x = torch.randn(tokens, channels, device="npu", dtype=dtype)
            weights = torch.randn(channels, 4, device="npu", dtype=dtype)
            # Long prefill uses the model's channel-contiguous state view;
            # the short case also covers a contiguous [slot, channel, window].
            if tokens == 8192:
                original_state = torch.randn(2, 12, channels, device="npu", dtype=dtype).transpose(1, 2)
                assert original_state.stride(1) == 1
            else:
                original_state = torch.randn(2, channels, 12, device="npu", dtype=dtype)
            original_output = torch.randn_like(x)
            for initialized in (True, False):
                for null_slot in (False, True):
                    for outer_enabled in (False, True):
                        # Keep the long shape to the common valid-state path.
                        if tokens == 8192 and (not initialized or null_slot):
                            continue
                        expected_state = original_state.clone()
                        actual_state = original_state.clone()
                        expected_output = original_output.clone()
                        actual_output = original_output.clone()
                        outer = torch.randn_like(x) if outer_enabled else None
                        kwargs = dict(
                            inputs=x,
                            conv_weights=weights,
                            state_indices=torch.tensor(
                                [NULL_BLOCK_ID if null_slot else 1], device="npu", dtype=torch.int32
                            ),
                            outer_residual=outer,
                            mode="prefill",
                            dilation=3,
                            query_start_loc=torch.tensor([0, tokens], device="npu", dtype=torch.int32),
                            has_initial_states=torch.tensor([initialized], device="npu"),
                        )
                        upstream_ple_conv(**kwargs, residual=expected_output, conv_state=expected_state)
                        _ple_conv_ascend(**kwargs, residual=actual_output, conv_state=actual_state)
                        torch.npu.synchronize()
                        pass_ratio, max_abs = _check_output_accuracy(expected_output, actual_output)
                        state_exact = torch.equal(expected_state, actual_state)
                        print(
                            f"PREFILL dtype={dtype} T={tokens} initial={initialized} null={null_slot} "
                            f"outer={outer_enabled} pass_ratio={pass_ratio:.6%} max_abs={max_abs:.6f} "
                            f"state_exact={state_exact}",
                            flush=True,
                        )
                        assert state_exact


def test_prefill_multiple_requests_across_token_chunks():
    torch.manual_seed(41)
    tokens, channels = 1031, 256
    query_start_loc = torch.tensor([0, 1, 1023, 1024, tokens], device="npu", dtype=torch.int32)
    state_indices = torch.tensor([1, 2, 3, NULL_BLOCK_ID], device="npu", dtype=torch.int32)
    has_initial_states = torch.tensor([True, False, True, True], device="npu")
    for dtype in (torch.bfloat16, torch.float16):
        inputs = torch.randn(tokens, channels, device="npu", dtype=dtype)
        weights = torch.randn(channels, 4, device="npu", dtype=dtype)
        state = torch.randn(4, 12, channels, device="npu", dtype=dtype).transpose(1, 2)
        residual = torch.randn_like(inputs)
        outer_residual = torch.randn_like(inputs)
        kwargs = dict(
            inputs=inputs,
            conv_weights=weights,
            state_indices=state_indices,
            outer_residual=outer_residual,
            mode="prefill",
            dilation=3,
            query_start_loc=query_start_loc,
            has_initial_states=has_initial_states,
        )
        expected, actual = residual.clone(), residual.clone()
        expected_state, actual_state = state.clone(), state.clone()
        upstream_ple_conv(**kwargs, residual=expected, conv_state=expected_state)
        _ple_conv_ascend(**kwargs, residual=actual, conv_state=actual_state)
        torch.npu.synchronize()
        _check_output_accuracy(expected, actual)
        assert torch.equal(expected_state, actual_state)
