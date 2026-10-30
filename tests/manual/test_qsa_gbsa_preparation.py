# SPDX-License-Identifier: Apache-2.0
"""Run explicitly on an idle NPU: python -m tests.manual.test_qsa_gbsa_preparation.

Checks compiled kernels, full GBSA output, and preparation graph replay with
changing inputs. This is not a substitute for full-model graph validation.
"""

import argparse
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401

from tests.ut.models.qwen4_exp.gbsa_reference import reference_prepare
from tests.ut.models.qwen4_exp.test_gbsa_preparation import make_inputs
from vllm_ascend.models.qwen4_exp import gbsa
from vllm_ascend.ops.triton.qwen4_exp.gbsa import prepare_qsa_gbsa_inputs


def compare(actual, expected):
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="Logical device within the authorized visible card set")
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    checks = 0
    for groups in (1, 31, 128, 511, 512):
        for tail in range(4):
            for pattern in ("full", "prefix", "holes", "invalid"):
                for dtype in (torch.int32, torch.int64):
                    cpu = make_inputs(groups, tail, pattern, dtype)
                    # Preserve noncontiguous input geometry after transfer.
                    inputs = []
                    for value in cpu:
                        backing = torch.empty_strided(value.shape, value.stride(), dtype=value.dtype, device="npu")
                        backing.copy_(value)
                        inputs.append(backing)
                    compare(prepare_qsa_gbsa_inputs(*inputs, 2), reference_prepare(*inputs, 2))
                    checks += 1
    print(f"PASS {checks} compiled preparation cases", flush=True)

    cpu = make_inputs(512, 3, "holes", torch.int32)
    inputs = [x.to("npu") for x in cpu]
    for _ in range(3):
        prepare_qsa_gbsa_inputs(*inputs, 2)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = prepare_qsa_gbsa_inputs(*inputs, 2)
    for tail in range(4):
        inputs[2][0] = 2048 + tail
        inputs[1][0] = 2047 + tail
        graph.replay()
        compare(captured, reference_prepare(*inputs, 2))
    print("PASS preparation graph replay with changed tail inputs", flush=True)

    # Use nonzero rows for native attention; empty/padded rows were checked above.
    for dtype in (torch.float16, torch.bfloat16):
        torch.manual_seed(803)
        query = torch.randn(7, 4, 128, dtype=dtype, device="npu")
        keys = torch.randn(100, 128, 2, 128, dtype=dtype, device="npu")
        values = torch.randn_like(keys)
        inputs[2].fill_(2051)
        inputs[1].fill_(2050)
        compact, positions, lengths, requests, table = inputs
        requests.clamp_(min=0)
        call_args = (query, keys, values, compact, table, requests, positions, lengths)
        actual = gbsa.qsa_gbsa_attention(*call_args, torch.empty_like(query))
        with patch.object(gbsa, "prepare_qsa_gbsa_inputs", reference_prepare):
            expected = gbsa.qsa_gbsa_attention(*call_args, torch.empty_like(query))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.npu.synchronize()
    print("PASS native GBSA FP16/BF16 output exact equality", flush=True)


if __name__ == "__main__":
    main()
