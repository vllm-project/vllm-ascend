# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from contextlib import nullcontext

import torch

from vllm_ascend.utils import global_stream, npu_stream_switch


def random_sample(probs: torch.Tensor, generators: dict[int, torch.Generator]) -> torch.Tensor:
    """Sample one token per row on the input device, modifying probs in place."""
    rng_stream = global_stream() if probs.device.type == "npu" else None
    # Noise only needs the input metadata, so its generation can overlap preceding NPU work.
    with npu_stream_switch(rng_stream) if rng_stream is not None else nullcontext():
        noise = torch.empty(probs.shape, dtype=probs.dtype, device=probs.device)
        # Draw default noise in a batch, then overwrite rows with request-owned RNG.
        if len(generators) != probs.shape[0]:
            noise.exponential_()
        for row, generator in generators.items():
            noise[row].exponential_(generator=generator)
    if rng_stream is not None:
        compute_stream = torch.npu.current_stream(probs.device)
        compute_stream.wait_stream(rng_stream)
        noise.record_stream(compute_stream)
    return probs.div_(noise).argmax(dim=-1).view(-1)
