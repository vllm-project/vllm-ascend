#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Smoke test that vLLM-Omni NPU diffusion attention uses MindIE-SD."""

from __future__ import annotations

import os
from pathlib import Path


EXPECTED_BACKEND = "vllm_omni.diffusion.attention.backends.flash_attn.FlashAttentionBackend"


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_npu_available() -> None:
    import torch
    import torch_npu  # noqa: F401

    _assert(hasattr(torch, "npu"), "torch.npu is not available")
    _assert(torch.npu.is_available(), "torch.npu.is_available() is false")

    device_count = torch.npu.device_count()
    _assert(device_count >= 1, f"Expected at least 1 NPU device, got {device_count}")
    torch.npu.set_device(0)

    print(f"NPU available: device_count={device_count}")


def check_mindiesd_env() -> None:
    import mindiesd

    package_root = Path(mindiesd.__file__).resolve().parent
    ascend_opp_path = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    path_entries = {
        Path(entry).resolve(strict=False)
        for entry in ascend_opp_path.split(os.pathsep)
        if entry
    }

    expected_paths = {
        (package_root / "ops" / "vendors" / "aie_ascendc").resolve(strict=False),
        (package_root / "ops" / "vendors" / "customize").resolve(strict=False),
    }
    missing = expected_paths - path_entries
    _assert(
        not missing,
        "ASCEND_CUSTOM_OPP_PATH is missing MindIE-SD custom OPP paths: "
        + ", ".join(str(path) for path in sorted(missing, key=str)),
    )

    print(f"mindiesd import OK: {package_root}")
    print("ASCEND_CUSTOM_OPP_PATH includes MindIE-SD custom OPP entries")


def check_backend_selection() -> None:
    from vllm_omni.platforms.npu.platform import NPUOmniPlatform

    backend = NPUOmniPlatform.get_diffusion_attn_backend_cls(None, 128)
    _assert(
        backend == EXPECTED_BACKEND,
        f"Expected NPU diffusion attention backend {EXPECTED_BACKEND}, got {backend}",
    )

    print(f"vLLM-Omni NPU backend selection OK: {backend}")


def check_forward_npu_calls_mindiesd() -> None:
    import mindiesd
    import torch
    from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl

    original_attention_forward = mindiesd.attention_forward
    call_count = 0

    def counted_attention_forward(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_attention_forward(*args, **kwargs)

    mindiesd.attention_forward = counted_attention_forward
    try:
        device = torch.device("npu:0")
        query = torch.randn((1, 2, 16, 128), dtype=torch.float16, device=device)
        key = torch.randn((1, 2, 16, 128), dtype=torch.float16, device=device)
        value = torch.randn((1, 2, 16, 128), dtype=torch.float16, device=device)

        impl = FlashAttentionImpl(
            num_heads=2,
            head_size=128,
            softmax_scale=128**-0.5,
            causal=False,
        )
        output = impl.forward_npu(query, key, value, None)
        torch.npu.synchronize()
    finally:
        mindiesd.attention_forward = original_attention_forward

    _assert(call_count >= 1, "mindiesd.attention_forward was not called")
    _assert(output.shape == query.shape, f"Expected output shape {query.shape}, got {output.shape}")
    _assert(output.dtype == query.dtype, f"Expected output dtype {query.dtype}, got {output.dtype}")
    _assert(torch.isfinite(output).all().item(), "Output contains NaN or Inf")

    print(f"FlashAttentionImpl.forward_npu called mindiesd.attention_forward {call_count} time(s)")
    print(f"Output OK: shape={tuple(output.shape)}, dtype={output.dtype}")


def main() -> None:
    check_npu_available()
    check_mindiesd_env()
    check_backend_selection()
    check_forward_npu_calls_mindiesd()
    print("MindIE-SD/vLLM-Omni NPU backend smoke passed")


if __name__ == "__main__":
    main()
