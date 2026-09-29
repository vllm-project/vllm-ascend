# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import subprocess
import sys
import textwrap

import pytest
import torch

from vllm_ascend.ops.gdn_attn_builder import _stable_argsort_for_npu

_AICPU_FALLBACK_WARNING = "running on AiCpu"


def _run_argsort_in_subprocess(expression: str) -> subprocess.CompletedProcess[str]:
    code = textwrap.dedent(
        f"""
        import torch
        from vllm_ascend.ops.gdn_attn_builder import _stable_argsort_for_npu

        mask = torch.tensor([True, False, True, False], device="npu")
        indices = {expression}
        indices.cpu()
        """
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=True,
        text=True,
        timeout=120,
    )


def test_boolean_token_partition_avoids_aicpu_fallback():
    # Use isolated processes because the CANN warning is emitted only once per
    # process. First verify that this runtime exposes the integer fallback, then
    # ensure the helper does not take that path.
    integer_sort = _run_argsort_in_subprocess("torch.argsort(mask.to(torch.int32), stable=True)")
    if _AICPU_FALLBACK_WARNING not in integer_sort.stderr:
        pytest.skip("Integer argsort already runs without the AiCPU fallback on this runtime")

    boolean_partition = _run_argsort_in_subprocess("_stable_argsort_for_npu(mask)")
    assert _AICPU_FALLBACK_WARNING not in boolean_partition.stderr


@pytest.mark.parametrize("size", [0, 1, 32, 2048, 8192])
@pytest.mark.parametrize("pattern", ["zeros", "ones", "alternating", "random"])
@pytest.mark.parametrize("strided", [False, True])
def test_boolean_token_partition(size, pattern, strided):
    generator = torch.Generator().manual_seed(42)
    mask = torch.randint(0, 2, (size,), generator=generator).bool()
    if pattern == "zeros":
        mask.fill_(False)
    elif pattern == "ones":
        mask.fill_(True)
    elif pattern == "alternating":
        mask = torch.arange(size) % 2 == 0

    expected = torch.argsort(mask.to(torch.int32), stable=True)
    device_mask = mask.to("npu")
    if strided:
        device_mask = torch.stack((device_mask, device_mask), dim=1)[:, 0]
    actual = _stable_argsort_for_npu(device_mask)
    assert actual.device == device_mask.device
    assert actual.dtype == torch.int64
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
