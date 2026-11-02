# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Store opaque KV bytes through the existing mapped Host allocation."""

from vllm.triton_utils import tl, triton


@triton.jit
def store_opaque_extents(descriptors, count, BLOCK: tl.constexpr):
    descriptor = tl.program_id(0)
    byte = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    source = tl.load(descriptors + descriptor).to(tl.pointer_type(tl.uint8))
    destination = tl.load(descriptors + count + descriptor).to(tl.pointer_type(tl.uint8))
    length = tl.load(descriptors + 2 * count + descriptor)
    payload = tl.load(source + byte, mask=byte < length, other=0)
    tl.store(destination + byte, payload, mask=byte < length)
