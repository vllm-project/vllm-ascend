# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""NPU comparisons against upstream PLE, including state writeback."""

from itertools import accumulate

import torch
import torch_npu  # noqa: F401
from vllm.models.qwen4_exp.common.ops.ple import ple_conv as upstream_ple_conv
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

from vllm_ascend.ops.torch_ple import ple_conv_torch


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
                expected, actual = residual.clone(), residual.clone()
                expected_state, actual_state = state.clone(), state.clone()
                upstream_ple_conv(**kwargs, residual=expected, conv_state=expected_state)
                ple_conv_torch(**kwargs, residual=actual, conv_state=actual_state)
                torch.npu.synchronize()
                exact = (expected == actual).float().mean().item()
                max_abs = (expected.float() - actual.float()).abs().max().item()
                state_exact = torch.equal(expected_state, actual_state)
                print(
                    f"CASE mode={mode} dtype={dtype} outer={outer_enabled} "
                    f"exact={exact:.6f} max_abs={max_abs:.6f} state_exact={state_exact}",
                    flush=True,
                )
                assert exact >= 0.999 and state_exact


def test_prefill_one_request_on_generic_path():
    torch.manual_seed(37)
    for dtype in (torch.bfloat16, torch.float16):
        for tokens in (256, 8192):
            channels = 10240
            x = torch.randn(tokens, channels, device="npu", dtype=dtype)
            weights = torch.randn(channels, 4, device="npu", dtype=dtype)
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
                        ple_conv_torch(**kwargs, residual=actual_output, conv_state=actual_state)
                        torch.npu.synchronize()
                        exact = (expected_output == actual_output).float().mean().item()
                        max_abs = (expected_output.float() - actual_output.float()).abs().max().item()
                        state_exact = torch.equal(expected_state, actual_state)
                        print(
                            f"PREFILL dtype={dtype} T={tokens} initial={initialized} null={null_slot} "
                            f"outer={outer_enabled} exact={exact:.6f} max_abs={max_abs:.6f} "
                            f"state_exact={state_exact}",
                            flush=True,
                        )
                        assert exact >= 0.999 and state_exact


def test_prefill_multiple_requests_across_token_chunks():
    torch.manual_seed(41)
    tokens, channels = 1031, 256
    query_start_loc = torch.tensor([0, 1, 1023, 1024, tokens], device="npu", dtype=torch.int32)
    state_indices = torch.tensor([1, 2, 3, NULL_BLOCK_ID], device="npu", dtype=torch.int32)
    has_initial_states = torch.tensor([True, False, True, True], device="npu")
    for dtype in (torch.bfloat16, torch.float16):
        for mapped in (False, True):
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
            if mapped:
                kwargs["token_indices"] = torch.arange(tokens - 1, -1, -1, device="npu", dtype=torch.int32)
            expected, actual = residual.clone(), residual.clone()
            expected_state, actual_state = state.clone(), state.clone()
            upstream_ple_conv(**kwargs, residual=expected, conv_state=expected_state)
            ple_conv_torch(**kwargs, residual=actual, conv_state=actual_state)
            torch.npu.synchronize()
            assert (expected == actual).float().mean().item() >= 0.999
            assert torch.equal(expected_state, actual_state)
