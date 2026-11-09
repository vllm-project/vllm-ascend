# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Ascend suffix-decoding speculator (Model Runner V2).

Ported from vLLM PR #60339 (SuffixDecodingSpeculator for MRV2) with
Ascend-specific hardening:
  - bounded local suffix scans stop once no candidate remains active
  - masked-load address clamping for stale corpus index entries
  - non-atomic corpus ingestion: exact float32 address sorting followed by
    three int64 segmented-max kernels on the current stream

Requires the complete upstream #60339 configuration and request-lifecycle
integration. Model Runner V2 remains an explicit opt-in on Ascend. Optional
adaptive verification retains the known CANN 9.1 observe-kernel compiler gap.
"""

from vllm_ascend.worker.v2.spec_decode.suffix.speculator import (
    SuffixSpeculator,
)

__all__ = ["SuffixSpeculator"]
