#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

"""Exercise the official Kimi convolution with this repository's OPP loaded.

Run on an Ascend server after rebuilding the custom operators. Each dtype
uses a fresh process so earlier tests cannot cache a different OPP provider.
Missing dependencies or operator binaries fail this regression, never skip it.
"""

import importlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


def _reference(x, weight, state, lengths, cache_ids, initial_flags):
    """Independent grouped-conv reference, computed on the CPU in float32."""
    expected_state = state.clone()
    outputs = []
    offset = 0
    for length, cache_id, has_initial_state in zip(lengths, cache_ids, initial_flags):
        if length == 0:
            continue
        history = state[cache_id].float() if has_initial_state else torch.zeros_like(state[cache_id]).float()
        sequence = torch.cat((history, x[offset : offset + length].float()), dim=0)
        convolution = F.conv1d(
            sequence.T.unsqueeze(0),
            weight.float().T.unsqueeze(1),
            groups=x.shape[-1],
        )
        outputs.append(F.silu(convolution.squeeze(0).T).to(x.dtype))
        expected_state[cache_id] = sequence[-state.shape[1] :].to(state.dtype)
        offset += length
    return torch.cat(outputs), expected_state


def _check_on_npu(dtype_name):
    # Import and bootstrap before any allocation on the NPU. Import the
    # extension directly because enable_custom_op() is disabled on A5.
    from vllm_ascend import utils

    vendor = Path(utils.__file__).resolve().parent / "_cann_ops_custom" / "vendors" / "custom_transformer"
    assert vendor.is_dir(), f"Rebuild the custom operators first: {vendor} is missing"
    utils.bootstrap_custom_op_env()
    importlib.import_module("vllm_ascend.vllm_ascend_C")
    from vllm_ascend.ops.kimi_kda import AscendKimiGatedDeltaNetAttention

    run_conv = AscendKimiGatedDeltaNetAttention._run_causal_conv1d
    dtype = getattr(torch, dtype_name)
    generator = torch.Generator().manual_seed(123)
    channels, width = 128, 4
    weight = (torch.randn(width, channels, generator=generator) * 0.25).to(dtype)
    state = (torch.randn(6, width - 1, channels, generator=generator) * 0.25).to(dtype)
    weight_npu = weight.to("npu")
    state_npu = state.to("npu")
    cache_ids = [2, 4, 1, 0]
    cache_indices = torch.tensor(cache_ids, dtype=torch.int32, device="npu")
    rtol, atol = (1e-2, 2e-3) if dtype == torch.bfloat16 else (3e-3, 3e-4)

    for chunk, (lengths, initial_flags) in enumerate(
        [([1, 5, 2, 0], [True, False, True, False]), ([2, 1, 4, 0], [True, True, True, False])]
    ):
        x = (torch.randn(sum(lengths), channels, generator=generator) * 0.25).to(dtype)
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        metadata = SimpleNamespace(
            query_start_loc=torch.tensor(offsets, dtype=torch.int32, device="npu"),
            cache_indices=cache_indices,
            initial_state_mode=torch.tensor(initial_flags, dtype=torch.bool, device="npu"),
        )
        expected, next_state = _reference(x, weight, state, lengths, cache_ids, initial_flags)
        x_npu = x.to("npu")

        if chunk == 0:
            # Resolve and execute the separately named custom operator first.
            # Its registration must coexist with official CausalConv1d.
            custom_state = state_npu.clone()
            custom_output = torch.empty_like(x_npu)
            torch.ops._C_ascend.npu_causal_conv1d_custom(
                custom_output,
                x_npu,
                weight_npu,
                conv_state=custom_state,
                bias_opt=None,
                query_start_loc_opt=metadata.query_start_loc,
                cache_indices_opt=cache_indices,
                initial_state_mode_opt=metadata.initial_state_mode,
                num_accepted_tokens_opt=None,
                activation_mode=1,
                pad_slot_id=0,
                run_mode=0,
            )
            torch.testing.assert_close(custom_output.cpu(), expected, rtol=rtol, atol=atol)
            torch.testing.assert_close(custom_state.cpu(), next_state, rtol=0, atol=0)

        actual = run_conv(x_npu, weight_npu, state_npu, metadata, run_mode=0)
        torch.testing.assert_close(actual.cpu(), expected, rtol=rtol, atol=atol)
        # This includes null row 0 and unused rows, which must stay untouched.
        torch.testing.assert_close(state_npu.cpu(), next_state, rtol=0, atol=0)
        state = next_state

    # Ordinary decode must consume the history produced by official Prefill.
    x = (torch.randn(3, channels, generator=generator) * 0.25).to(dtype)
    expected, next_state = _reference(x, weight, state, [1, 1, 1], cache_ids[:3], [True] * 3)
    metadata = SimpleNamespace(
        query_start_loc=torch.arange(4, dtype=torch.int32, device="npu"),
        cache_indices=cache_indices[:3],
    )
    actual = run_conv(x.to("npu"), weight_npu, state_npu, metadata, run_mode=1)
    torch.testing.assert_close(actual.cpu(), expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(state_npu.cpu(), next_state, rtol=0, atol=0)


@pytest.mark.parametrize("dtype_name", ["bfloat16", "float16"])
def test_kimi_official_prefill_and_decode_with_custom_opp(dtype_name):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.e2e.nightly.single_node.ops.singlecard_ops.test_kimi_causal_conv1d_official",
            dtype_name,
        ],
        cwd=Path(__file__).resolve().parents[6],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    _check_on_npu(sys.argv[1])
