# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.mamba.postprocess import (
    _copy_temporal_state,
    postprocess_mamba_fused_kernel,
)

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="NPU required")


@triton.jit
def _copy_temporal_test_kernel(src, dst, size, TILES: tl.constexpr):
    _copy_temporal_state(src.to(tl.int64), dst.to(tl.int64), size, tl.program_id(0), 1024, TILES)


@pytest.mark.parametrize("size", [0, 7, 1027, 786432, 786435, 786443])
@pytest.mark.parametrize("offsets", [(0, 0), (1, 1), (1, 3), (0, 8)])
@pytest.mark.parametrize("tiles", [1, 4])
def test_temporal_copy_preserves_bytes_and_guards(size, offsets, tiles):
    src_offset, dst_offset = offsets
    source = torch.randint(0, 256, (size + 64,), dtype=torch.uint8, device="npu")
    destination = torch.full_like(source, 123)
    expected = destination.cpu()
    expected[dst_offset : dst_offset + size] = source.cpu()[src_offset : src_offset + size]
    _copy_temporal_test_kernel[(tiles,)](source[src_offset:], destination[dst_offset:], size, tiles)
    assert torch.equal(destination.cpu(), expected)


@pytest.mark.parametrize(
    "state_shape,dtype",
    [
        pytest.param((1024, 16), torch.bfloat16, id="jamba-mamba1-tp8"),
        pytest.param((4, 128, 128), torch.bfloat16, id="qwen3-next-gdn-tp8"),
        pytest.param((32, 128, 128), torch.bfloat16, id="qwen3-next-gdn-tp1"),
        pytest.param((4, 128, 128), torch.float32, id="qwen3_5-gdn-tp8-fp32-cache"),
        pytest.param((16, 64, 128), torch.bfloat16, id="granite4-mamba2-tp8"),
        pytest.param((128, 64, 128), torch.bfloat16, id="granite4-mamba2-tp1"),
        pytest.param((3, 128, 256), torch.bfloat16, id="falcon-h1-mamba2-tp8"),
        pytest.param((2, 128, 128), torch.float32, id="ling3-kda-tp8"),
        pytest.param((2, 128, 128), torch.bfloat16, id="ring-lite-linear-tp8"),
    ],
)
@pytest.mark.parametrize("tiles", [1, 4])
def test_temporal_copy_config_derived_sizes(state_shape, dtype, tiles):
    # These are config-derived state sizes; no model class or weights are loaded.
    size = torch.Size(state_shape).numel() * torch.empty((), dtype=dtype).element_size()
    source = torch.randint(0, 256, (size + 64,), dtype=torch.uint8, device="npu")
    destination = torch.full_like(source, 123)
    expected = destination.cpu()
    expected[:size] = source.cpu()[:size]
    _copy_temporal_test_kernel[(tiles,)](source, destination, size, tiles)
    assert torch.equal(destination.cpu(), expected)


@pytest.mark.parametrize("precomputed", [False, True])
@pytest.mark.parametrize("graph_mode", [False, True])
@pytest.mark.parametrize(
    "state_types,state_bytes",
    [
        pytest.param(("conv", "temporal"), 1027, id="mixed-unaligned-tail"),
        pytest.param(("conv", "temporal"), 131072, id="gdn-aligned-tp8"),
        pytest.param(("temporal",), 65536, id="linear-only-tp8"),
    ],
)
def test_mixed_postprocess_replay_uses_updated_metadata(precomputed, graph_mode, state_types, state_bytes):
    """Check mixed state types, padded pages and changed decisions in one graph."""
    num_requests, num_blocks, num_layers = 4, 8, 2
    conv_width, conv_inner = 7, 12
    conv_bytes = conv_width * conv_inner * 2
    state_kinds = state_types * num_layers
    sizes = [conv_bytes if kind == "conv" else state_bytes for kind in state_kinds]
    strides = [
        conv_bytes + 64 if kind == "conv" else state_bytes + (125 if state_bytes == 1027 else 128)
        for kind in state_kinds
    ]
    offsets = [1 if kind == "temporal" and state_bytes == 1027 and i == 1 else 0 for i, kind in enumerate(state_kinds)]
    storage = [
        torch.randint(0, 256, (num_requests * num_blocks, stride), dtype=torch.uint8, device="npu")
        for stride in strides
    ]

    def tensor(values, dtype=torch.int64):
        return torch.tensor(values, dtype=dtype, device="npu")

    table = torch.arange(num_requests * num_blocks, dtype=torch.int32, device="npu").reshape(num_requests, num_blocks)
    accepted = tensor([1] * num_requests, torch.int32)
    computed = tensor([64] * num_requests, torch.int32)
    source_columns = tensor([3] * num_requests, torch.int32)
    mapping = tensor([0, 1, 2, 3], torch.int32)
    accepted_out = tensor([-99] * num_requests, torch.int32)
    args = (
        accepted,
        source_columns,
        tensor([4] * num_requests, torch.int32),
        computed,
        tensor([3] * num_requests, torch.int32),
        tensor([table.data_ptr()]),
        num_blocks,
        tensor([data.data_ptr() + offset for data, offset in zip(storage, offsets)]),
        tensor(strides),
        tensor([2 if kind == "conv" else 1 for kind in state_kinds]),
        tensor([conv_inner if kind == "conv" else state_bytes for kind in state_kinds]),
        tensor([conv_width if kind == "conv" else 0 for kind in state_kinds], torch.int32),
        tensor([0] * len(state_kinds), torch.int32),
        tensor([0] * len(state_kinds), torch.int32),
        tensor([0] * len(state_kinds)),
        accepted_out,
        mapping,
        num_requests,
    )

    def launch():
        postprocess_mamba_fused_kernel[(num_requests, len(state_kinds), 1)](
            *args,
            block_size=128,
            COPY_BLOCK_SIZE=1024,
            CONV_STATE_DIM_FIRST=False,
            HAS_IDX_MAPPING=True,
            PRECOMPUTED_NEW_COMPUTED=precomputed,
            TEMPORAL_TILES=1,
        )

    launch()
    torch.npu.synchronize()
    if graph_mode:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            launch()

    for reordering, new_counts, accept_counts, src_cols in [
        ([0, 1, 2, 3], [64, 128, 129, 128], [1, 2, 4, 4], [3, 3, 3, 3]),
        ([3, 0, -1, 1], [128, 128, 64, 128], [1, 4, 1, 2], [0, 3, 3, 3]),
        ([1, 2, 3, 0], [64, 64, 64, 64], [1, 1, 1, 1], [3, 3, 3, 3]),
    ]:
        for data in storage:
            data.random_(0, 256)
        initial = [data.cpu() for data in storage]
        expected = [data.clone() for data in initial]
        expected_out = torch.full((num_requests,), -99, dtype=torch.int32)
        mapping.copy_(tensor(reordering, torch.int32))
        accepted.copy_(tensor(accept_counts, torch.int32))
        source_columns.copy_(tensor(src_cols, torch.int32))
        values = new_counts if precomputed else [count - acc for count, acc in zip(new_counts, accept_counts)]
        computed.copy_(tensor(values, torch.int32))
        accepted_out.fill_(-99)
        for batch, req in enumerate(reordering):
            if req < 0:
                continue
            running = new_counts[req] - accept_counts[req] + 1
            aligned = new_counts[req] // 128 * 128
            if aligned < running:
                continue
            bias = aligned - running
            src, dst = src_cols[req], aligned // 128 - 1
            if src == dst:
                expected_out[req] = 1
            if src == dst and bias == 0:
                continue
            for state, offset in enumerate(offsets):
                if state_kinds[state] == "conv":
                    copy_size = (conv_width - bias) * conv_inner * 2
                    src_block = batch * num_blocks + src
                    src_offset = offset + bias * conv_inner * 2
                else:
                    copy_size = sizes[state]
                    src_block = batch * num_blocks + src + bias
                    src_offset = offset
                expected[state][batch * num_blocks + dst, offset : offset + copy_size] = initial[state][
                    src_block, src_offset : src_offset + copy_size
                ]
        if graph_mode:
            graph.replay()
        else:
            launch()
        for actual, reference in zip(storage, expected):
            assert torch.equal(actual.cpu(), reference)
        assert torch.equal(accepted_out.cpu(), expected_out)
