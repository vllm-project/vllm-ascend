# SPDX-License-Identifier: Apache-2.0
"""Run on Ascend with python benchmarks/vision_qk_pack.py."""

import statistics

import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.qwen_prefill import pack_vision_qk


def original(qkv):
    return qkv[:, :2].permute(1, 0, 2, 3).contiguous()


def latency_us(fn, qkv):
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    fn(qkv)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000


@torch.inference_mode()
def main():
    for tokens in (37836, 20048):
        qkv = torch.randn(tokens, 3, 16, 72, device="npu", dtype=torch.bfloat16)
        assert torch.equal(original(qkv), pack_vision_qk(qkv))
        for _ in range(5):
            original(qkv)
            pack_vision_qk(qkv)
        torch.npu.synchronize()
        baseline, packed = [], []
        cases = [(original, baseline), (pack_vision_qk, packed)]
        for i in range(40):
            for fn, samples in cases if i % 2 == 0 else reversed(cases):
                samples.append(latency_us(fn, qkv))
        print(f"T={tokens}: original={statistics.median(baseline):.2f} us, packed={statistics.median(packed):.2f} us")


if __name__ == "__main__":
    main()
