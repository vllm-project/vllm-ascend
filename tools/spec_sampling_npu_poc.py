#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

import vllm_ascend.sample.rejection_sampler as rejection_sampler_module
import vllm_ascend.sample.sampler as sampler_module
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.sample.rejection_sampler import AscendRejectionSampler
from vllm_ascend.sample.sampler import AscendSampler
from vllm_ascend.sample.spec_sampling_executor import SpecSamplingNPUExecutor
from vllm_ascend.sample.spec_sampling_poc import load_spec_sampling_case


def _move_tensors(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if hasattr(obj, "__dataclass_fields__"):
        kwargs = {name: _move_tensors(getattr(obj, name), device) for name in obj.__dataclass_fields__}
        return type(obj)(**kwargs)
    if hasattr(obj, "_fields") and hasattr(obj, "_asdict"):
        values = {name: _move_tensors(getattr(obj, name), device) for name in obj._fields}
        return type(obj)(**values)
    if isinstance(obj, dict):
        return {key: _move_tensors(value, device) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_move_tensors(item, device) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_move_tensors(item, device) for item in obj)
    return obj


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay one dumped MTP spec-sampling case on NPU.")
    parser.add_argument("case_path", type=Path, help="Path to case.pt or its containing directory")
    parser.add_argument("--device", default="npu", help="Replay device, default: npu")
    args = parser.parse_args()

    payload = load_spec_sampling_case(args.case_path)
    device = torch.device(args.device)
    if args.device == "npu":
        init_device_properties_triton()

    dummy_ascend_config = SimpleNamespace(
        enable_reduce_sample=False,
        enable_async_exponential=False,
    )
    sampler_module.get_ascend_config = lambda: dummy_ascend_config
    rejection_sampler_module.get_ascend_config = lambda: dummy_ascend_config

    logits = payload["logits"].to(device)
    draft_probs = payload["draft_probs"]
    draft_probs = None if draft_probs is None else draft_probs.to(device)
    sampling_metadata = _move_tensors(payload["sampling_metadata"], device)
    spec_decode_metadata = _move_tensors(payload["spec_decode_metadata"], device)
    expected = payload["sampler_output"].sampled_token_ids
    expected = None if expected is None else expected.to(device)
    prepared_top_k = payload.get("prepared_top_k")

    sampler = AscendSampler()
    rejection_sampler = AscendRejectionSampler(sampler)
    executor = SpecSamplingNPUExecutor(sampler, rejection_sampler)
    inputs = executor.build_inputs(
        metadata=spec_decode_metadata,
        sampling_metadata=sampling_metadata,
        logits=logits,
        draft_probs=draft_probs,
        prepared_top_k=prepared_top_k,
    )
    replay_output = executor.execute(inputs)

    actual = replay_output.sampled_token_ids
    if expected is not None and torch.equal(actual, expected):
        print("Replay matched expected sampled_token_ids.")
        return 0

    print("Replay mismatch.")
    if expected is not None:
        print(f"expected shape: {tuple(expected.shape)}")
        print(f"actual shape:   {tuple(actual.shape)}")
        print(f"expected: {expected}")
    print(f"actual:   {actual}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
