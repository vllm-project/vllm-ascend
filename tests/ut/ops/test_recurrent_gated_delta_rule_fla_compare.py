#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
import itertools
import statistics

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

# On CPU-only runners tests/ut/conftest.py installs a MagicMock torch_npu:
# torch_npu.npu.device_count() then returns a MagicMock which does not
# support comparison, so a real NPU passes and the mock raises -> skip.
try:
    _has_npu = torch_npu.npu.device_count() > 0
except Exception:
    _has_npu = False
if not _has_npu:
    pytest.skip("requires a real NPU (torch_npu is mocked on CPU-only runners)", allow_module_level=True)

pytest.importorskip("fla_npu")
try:
    from fla_npu.ops.ascendc import recurrent_gated_delta_rule  # type: ignore
except ImportError:
    pytest.skip("fla_npu.ops.ascendc.recurrent_gated_delta_rule unavailable", allow_module_level=True)

try:
    import vllm_ascend.vllm_ascend_C  # type: ignore # noqa: F401
except ImportError:
    pytest.skip("vllm_ascend_C extension not built/importable", allow_module_level=True)

if not hasattr(torch.ops._C_ascend, "npu_recurrent_gated_delta_rule"):
    pytest.skip("torch.ops._C_ascend.npu_recurrent_gated_delta_rule not registered", allow_module_level=True)

torch_npu.npu.set_compile_mode(jit_compile=False)

DIM = 128  # Dk == Dv
WARMUP = 2
ITERS = 7
# Perf sanity bound: median(fla) <= median(native) * RATIO_BOUND + SLACK_MS.
# Only catches catastrophic regressions; exact numbers are printed, not asserted.
RATIO_BOUND = 2.5
SLACK_MS = 1.0

# Alternates which backend is measured first per case to cancel drift.
_order_counter = itertools.count()


def _build_inputs(batch: int, mtp: int, hk: int, hv: int, device: torch.device) -> tuple[dict, tuple]:
    """Build MTP spec-decode style inputs (bf16 state, matching the online
    MambaStateDtypeCalculator default). Mirrors the input construction of the
    e2e op test tests/e2e/nightly/single_node/ops/singlecard_ops/
    test_recurrent_gated_delta_rule.py."""
    t = batch * mtp
    torch.manual_seed(42)
    query = torch.nn.functional.normalize(torch.rand((t, hk, DIM), device=device), p=2, dim=-1).to(torch.bfloat16)
    key = torch.nn.functional.normalize(torch.rand((t, hk, DIM), device=device), p=2, dim=-1).to(torch.bfloat16)
    value = torch.rand((t, hv, DIM), device=device).to(torch.bfloat16)
    g = torch.rand((t, hv), device=device, dtype=torch.float32)
    beta = torch.rand((t, hv), device=device).to(torch.bfloat16)
    # Per the kernel source (fla_npu vendor, arch35/recurrent_gated_delta_rule.h):
    # actual_seq_lengths holds per-sequence LENGTHS behind a leading prefix
    # entry; the kernel accumulates internally (seq0 = asl[0], seq1 += len).
    # Passing cumulative offsets instead trips the kernel's (seq0 + len) <= T
    # bounds check, which returns early leaving the output uninitialized (NaN).
    actual_seq_lengths = torch.zeros(batch + 1, device=device, dtype=torch.int32)
    actual_seq_lengths[1:] = mtp
    ssm_state_indices = torch.arange(t, device=device).to(torch.int32)
    # Kernel requires num_accepted_tokens in [1, seqLen] per sequence (an
    # out-of-range value makes it bail out early with uninitialized output);
    # randint(1, mtp + 1) keeps it valid for uniform lengths of mtp.
    num_accepted_tokens = torch.randint(1, mtp + 1, (batch,), device=device, dtype=torch.int32)
    kwargs = dict(
        query=query,
        key=key,
        value=value,
        g=g,
        beta=beta,
        scale=DIM**-0.5,
        actual_seq_lengths=actual_seq_lengths,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
    )
    state_shape = (t, hv, DIM, DIM)
    return kwargs, state_shape


def _median_ms(op, kwargs: dict, state_shape: tuple, device: torch.device) -> float:
    """Warm up, then time `op` with device events; returns median wall ms."""
    state = torch.zeros(state_shape, dtype=torch.bfloat16, device=device)
    for _ in range(WARMUP):
        state.zero_()
        op(**kwargs, state=state)
    torch.npu.synchronize()
    starts = [torch.npu.Event(enable_timing=True) for _ in range(ITERS)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(ITERS)]
    for i in range(ITERS):
        state.zero_()
        starts[i].record()
        op(**kwargs, state=state)
        ends[i].record()
    torch.npu.synchronize()
    del state
    return statistics.median(s.elapsed_time(e) for s, e in zip(starts, ends))


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("mtp", [1, 4])
@pytest.mark.parametrize("headnum", [(4, 8), (8, 16), (16, 32)])
def test_fla_vs_native_perf(batch, mtp, headnum):
    hk, hv = headnum
    device = torch.npu.current_device()
    kwargs, state_shape = _build_inputs(batch, mtp, hk, hv, device)
    native = torch.ops._C_ascend.npu_recurrent_gated_delta_rule

    # 1) Correctness: identical initial state, both host paths must agree on
    #    output and final state.
    torch.manual_seed(42)
    state_init = torch.rand(state_shape, dtype=torch.bfloat16, device=device)
    state_fla, state_nat = state_init.clone(), state_init.clone()
    out_fla = recurrent_gated_delta_rule(**kwargs, state=state_fla)
    out_nat = native(**kwargs, state=state_nat)
    torch.npu.synchronize()
    assert out_fla.shape == out_nat.shape == (batch * mtp, hv, DIM), (
        f"unexpected out shape fla={out_fla.shape} native={out_nat.shape}"
    )
    torch.testing.assert_close(out_fla.cpu().float(), out_nat.cpu().float(), rtol=3e-3, atol=1e-2, equal_nan=True)
    torch.testing.assert_close(state_fla.cpu().float(), state_nat.cpu().float(), rtol=3e-3, atol=1e-2, equal_nan=True)

    # 2) Performance: alternate measurement order between cases.
    reverse = next(_order_counter) % 2 == 1
    if reverse:
        native_ms = _median_ms(native, kwargs, state_shape, device)
        fla_ms = _median_ms(recurrent_gated_delta_rule, kwargs, state_shape, device)
    else:
        fla_ms = _median_ms(recurrent_gated_delta_rule, kwargs, state_shape, device)
        native_ms = _median_ms(native, kwargs, state_shape, device)
    ratio = fla_ms / native_ms if native_ms > 0 else float("inf")
    print(
        f"recurrent[batch={batch}, mtp={mtp}, heads=({hk},{hv})] "
        f"fla={fla_ms * 1e3:.1f}us native={native_ms * 1e3:.1f}us "
        f"ratio={ratio:.3f}"
    )

    assert fla_ms <= native_ms * RATIO_BOUND + SLACK_MS, (
        f"fla median {fla_ms * 1e3:.1f}us exceeds native "
        f"{native_ms * 1e3:.1f}us * {RATIO_BOUND} + {SLACK_MS}ms; "
        f"check for a host-path regression"
    )

    del state_init, state_fla, state_nat, out_fla, out_nat
    torch.npu.empty_cache()
