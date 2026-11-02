# SPDX-License-Identifier: Apache-2.0
"""Copy opaque byte extents between two retained NPU stage tensors."""

from vllm.triton_utils import tl, triton


@triton.jit
def _reorder(source, output, descriptors, BLOCK: tl.constexpr):
    segment = tl.program_id(0)
    source_start = tl.load(descriptors + segment * 3)
    output_start = tl.load(descriptors + segment * 3 + 1)
    length = tl.load(descriptors + segment * 3 + 2)
    offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(source + source_start + offset, mask=offset < length, other=0)
    tl.store(output + output_start + offset, value, mask=offset < length)


def reorder(source, output, descriptors, count, max_length):
    assert source.dtype == output.dtype and source.element_size() == 1
    assert source.device == output.device == descriptors.device
    assert source.data_ptr() != output.data_ptr()
    assert 8 <= count <= 256 and 0 < max_length <= source.numel()
    _reorder[(count, triton.cdiv(max_length, 32768))](source, output, descriptors, BLOCK=32768)
