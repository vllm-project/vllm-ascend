# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Ascend suffix-decoding speculator (Model Runner V2).

Ported from vLLM PR #60339 (SuffixDecodingSpeculator for MRV2) with
Ascend-specific hardening:
  - masked-load address clamping in all four Triton kernels (NPU masked
    loads do not short-circuit address computation, so stale/corrupt
    index entries must never produce an out-of-range address)
  - rolling-hash ingest preserved (verified order-independent atomics)
"""
from vllm_ascend.worker.v2.spec_decode.suffix.speculator import (
    SuffixSpeculator,
)

__all__ = ["SuffixSpeculator"]
