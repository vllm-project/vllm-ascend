# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch
from vllm.config import CUDAGraphMode

from vllm_ascend.attention.flash_attention_v3 import (
    AscendFlashAttentionBackend,
    AscendFlashAttentionImpl,
)


@pytest.fixture
def ops(monkeypatch):
    operators = NS(
        get_scheduler_metadata=Mock(side_effect=lambda **_: torch.zeros(8, dtype=torch.uint8)),
        flash_attn_with_kvcache=Mock(side_effect=lambda query, *_args, **_kwargs: query + 1),
    )
    monkeypatch.setitem(sys.modules, "flash_attn_npu_3", operators)
    return operators


def make_builder(mode, eager=False):
    impl = AscendFlashAttentionImpl(8, 192, 192**-0.5, num_kv_heads=2)
    config = NS(
        speculative_config=NS(method="dflash", enforce_eager=eager, num_speculative_tokens=7),
        scheduler_config=NS(max_num_seqs=4),
        compilation_config=NS(
            cudagraph_mode=mode,
            cudagraph_capture_sizes=[15],
            static_forward_context={f"attn.{i}": NS(impl=impl) for i in range(2)},
        ),
    )
    return AscendFlashAttentionBackend.get_builder_cls()(
        NS(dtype=torch.bfloat16, block_size=128), ["attn.0", "attn.1"], config, torch.device("cpu")
    ), impl


def make_common():
    offsets = torch.arange(5, dtype=torch.int32) * 4
    return NS(
        num_reqs=4,
        query_start_loc_cpu=offsets,
        query_start_loc=offsets.clone(),
        seq_lens=torch.tensor([21, 31, 41, 51], dtype=torch.int32),
        block_table_tensor=torch.zeros(4, 8, dtype=torch.int32),
        num_input_tokens=16,
        num_actual_tokens=16,
        max_query_len=4,
        causal=False,
        slot_mapping=torch.arange(16, dtype=torch.int32),
    )


@pytest.mark.parametrize("eager", [False, True])
def test_dflash_rounded_bucket_preserves_exact_device_lengths_and_tiling_storage(ops, eager):
    builder, _ = make_builder(CUDAGraphMode.FULL_DECODE_ONLY, eager)
    common = make_common()
    captured = builder.build_for_graph_capture(common)
    spec = next(iter(captured.scheduler_metadata))
    for offsets in ([0, 4, 4, 4, 4], [0, 4, 8, 12, 12], [0, 4, 8, 12, 16]):
        common.query_start_loc_cpu.copy_(torch.tensor(offsets))
        common.query_start_loc.copy_(common.query_start_loc_cpu)
        common.seq_lens.sub_(1)
        metadata = builder.build(0, common)
        active = offsets.index(offsets[-1])
        arguments = ops.get_scheduler_metadata.call_args.kwargs
        assert arguments["batch_size"] == active
        assert arguments["max_seqlen_q"] == 4
        assert arguments["cache_seqlens"].data_ptr() == common.seq_lens.data_ptr()
        torch.testing.assert_close(metadata.seq_lens, common.seq_lens[:active])
        torch.testing.assert_close(metadata.query_start_loc, common.query_start_loc[: active + 1])
        # The target's 15-token bucket rounds to a 16-token draft graph.
        assert (metadata.scheduler_metadata[spec] is captured.scheduler_metadata[spec]) is not eager
    assert ops.get_scheduler_metadata.call_count == 4  # Shared by two equivalent layers.
    assert bool(builder.scheduler_buffers) is not eager
