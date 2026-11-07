# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-independent primitives for MemFabric GVA-backed tables."""

from vllm_ascend.distributed.gva.layout import GvaRowPartition, GvaRowRange

__all__ = ["GvaRowPartition", "GvaRowRange"]
