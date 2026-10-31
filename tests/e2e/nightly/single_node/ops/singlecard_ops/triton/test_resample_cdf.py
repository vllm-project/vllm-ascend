# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regressions for inverse-CDF rounding across vocabulary blocks."""

import json
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.ops.triton.v2.spec_decode.resample import resample


@pytest.mark.parametrize("case_index", [0, 1], ids=["bonus", "full-draft-residual"])
@torch.inference_mode()
def test_cdf_never_selects_zero_mass_token(case_index: int):
    """A block CDF and a separately reduced prefix can disagree in fp32."""
    init_device_properties_triton()
    path = Path(__file__).parent / "data/resample_cdf_zero_mass.json"
    case = json.loads(path.read_text(encoding="utf-8"))[case_index]
    vocab_size = 200064
    rows = len(case["pos"])
    target_logits = torch.full((rows, vocab_size), float("-inf"), dtype=torch.float32, device="npu")
    finite_logits = torch.tensor(case["finite_logits"], dtype=torch.float32, device="npu")
    target_logits[0, torch.tensor(case["finite_ids"], dtype=torch.int64, device="npu")] = finite_logits
    assert not bool(torch.isfinite(target_logits[0, case["expected_bad_token"]]))

    sampled = torch.full((1, rows), -1, dtype=torch.int64, device="npu")
    num_sampled = torch.zeros(1, dtype=torch.int32, device="npu")
    zero = torch.zeros(1, dtype=torch.float32, device="npu")
    target_lse = zero.clone()
    draft_logits = None
    if case["mode"] == "residual-full":
        target_lse.fill_(float(torch.logsumexp(finite_logits.double(), 0)))
        draft_logits = torch.full((1, 1, vocab_size), float("-inf"), device="npu")
        draft_logits[0, 0, 0] = 0.0

    resample(
        sampled,
        num_sampled,
        target_logits,
        target_lse,
        draft_logits,
        zero,
        torch.tensor([0, rows], dtype=torch.int32, device="npu"),
        torch.zeros(rows, dtype=torch.int32, device="npu"),
        torch.zeros(rows, dtype=torch.int32, device="npu"),
        torch.tensor([0.6], dtype=torch.float32, device="npu"),
        torch.tensor(case["seed"], dtype=torch.int64, device="npu"),
        torch.tensor(case["pos"], dtype=torch.int64, device="npu"),
    )
    torch.npu.synchronize()

    assert num_sampled.item() == 1
    token = sampled[0, 0].item()
    assert token != case["expected_bad_token"]
    assert torch.isfinite(target_logits[0, token]).item()
