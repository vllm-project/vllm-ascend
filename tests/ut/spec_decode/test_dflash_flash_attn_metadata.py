# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exercise metadata update negotiation through the real DFlash2 propose path."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import vllm.v1.worker.gpu.cudagraph_utils as cudagraph_utils
import vllm.v1.worker.gpu.spec_decode.dflash.speculator as upstream_dflash
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.attention.attention_v1 import AscendAttentionBackend
from vllm_ascend.attention.fa_v1 import AscendFlashAttentionBackend, AscendFlashAttentionMetadata
from vllm_ascend.worker.v2.spec_decode.dflash.aclgraph import DFlashAclGraphManager
from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import AscendDFlash2Speculator


def _make_speculator(backends):
    """Keep real speculator/group/manager methods; supply CPU input buffers."""
    speculator = AscendDFlash2Speculator.__new__(AscendDFlash2Speculator)
    speculator.vllm_config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=1024))
    speculator.num_query_per_req = 4
    speculator.num_speculative_steps = 3
    speculator.max_num_reqs = 4
    speculator.max_num_tokens = 16
    speculator.dp_size = speculator.dcp_size = 1
    speculator.dp_rank = 0
    speculator.pcp_manager = None
    speculator.parallel_drafting_token_id = 0
    speculator.sample_from_anchor = False
    speculator.draft_kv_cache_group_ids = list(range(len(backends)))
    speculator.draft_kv_cache_group_id = 0
    speculator._group_causal = {i: False for i in range(len(backends))}
    speculator.attn_groups = []
    speculator.attn_backends = {}
    kv_spec = FullAttentionSpec(block_size=128, num_kv_heads=1, head_size=128, dtype=torch.bfloat16)
    for gid, backend in enumerate(backends):
        builder_cls = backend.get_builder_cls()
        builder = builder_cls.__new__(builder_cls)
        layer_name = f"draft.{gid}"
        speculator.attn_groups.append(
            [AttentionGroup(backend, [layer_name], kv_spec, gid, metadata_builders=[builder])]
        )
        speculator.attn_backends[layer_name] = backend
    # Negotiate through the actual AttentionGroup property and builder flag.
    # Hard-coding supports_decode_metadata_update would hide the regression.
    speculator._configure_decode_metadata_update()
    speculator.input_buffers = SimpleNamespace(
        seq_lens=torch.zeros(4, dtype=torch.int32),
        query_start_loc=torch.arange(5, dtype=torch.int32) * speculator.num_query_per_req,
    )
    speculator.block_tables = SimpleNamespace(
        slot_mappings=torch.zeros(len(backends), 16, dtype=torch.int32),
        input_block_tables=[torch.zeros(4, 8, dtype=torch.int32) for _ in backends],
        block_sizes=[128 for _ in backends],
        cp_size=1,
        cp_rank=0,
        cp_interleave=1,
    )
    speculator._context_slot_mappings = torch.zeros(len(backends), 16, dtype=torch.int32)
    speculator.context_positions = torch.zeros(16, dtype=torch.int64)
    speculator.sample_indices = speculator.sample_pos = speculator.sample_idx_mapping = torch.zeros(
        16, dtype=torch.int32
    )
    speculator.temperature = torch.ones(4)
    speculator.seeds = torch.zeros(4, dtype=torch.int64)
    speculator.draft_tokens = torch.zeros(4, 3, dtype=torch.int64)
    speculator._num_rejected = torch.zeros(4, dtype=torch.int32)
    speculator._aux_staging = None
    speculator.target_input_buffers = speculator.input_buffers
    speculator.kv_cache_config = object()
    speculator._num_graph_context_tokens = MagicMock(return_value=16)
    speculator._store_hidden_states = MagicMock()
    speculator.prepare_context_anchor = MagicMock()
    speculator._precompute_context_kv = MagicMock()
    speculator._prepare_eplb_forward = MagicMock()
    speculator._generate_draft = MagicMock()
    manager = DFlashAclGraphManager.__new__(DFlashAclGraphManager)
    manager.speculator = speculator
    speculator.query_cudagraph_manager = manager
    return speculator


def test_flash_attn_graph_forward_updates_metadata_without_replacing_device_views():
    speculator = _make_speculator([AscendFlashAttentionBackend])
    assert speculator.supports_decode_metadata_update
    builder = speculator.attn_groups[0][0].get_metadata_builder()
    metadata = AscendFlashAttentionMetadata(
        query_start_loc_gpu=speculator.input_buffers.query_start_loc,
        seq_lens_gpu=speculator.input_buffers.seq_lens,
        block_tables=speculator.block_tables.input_block_tables[0],
        slot_mapping=speculator.block_tables.slot_mappings[0],
    )
    original_fields = vars(metadata).copy()
    update = MagicMock(wraps=builder.update_draft_decode_metadata)
    builder.update_draft_decode_metadata = update

    # This is the real callback recorded before the draft attention forward.
    # It reaches Speculator._update_draft_decode_metadata and AttentionGroup.
    for lengths in ([100, 200, 0, 0], [197, 99, 0, 0]):
        speculator.input_buffers.seq_lens.copy_(torch.tensor(lengths, dtype=torch.int32))
        speculator._prepare_graph_forward(4, {"draft.0": metadata})
        assert all(getattr(metadata, name) is value for name, value in original_fields.items())
        torch.testing.assert_close(metadata.seq_lens_gpu, speculator.input_buffers.seq_lens)
    assert update.call_count == 2
    update.assert_called_with(metadata)


@pytest.mark.parametrize(
    "graph_mode,mixed_backends",
    [
        (CUDAGraphMode.FULL, False),
        (CUDAGraphMode.NONE, False),
        (CUDAGraphMode.PIECEWISE, False),
        (CUDAGraphMode.FULL, True),
    ],
    ids=["flash-full", "flash-eager", "flash-piecewise", "mixed-full"],
)
def test_propose_skips_metadata_build_only_for_supported_full_graphs(monkeypatch, graph_mode, mixed_backends):
    backends = [AscendFlashAttentionBackend]
    if mixed_backends:
        backends.append(AscendAttentionBackend)
    speculator = _make_speculator(backends)
    manager = speculator.query_cudagraph_manager
    desc = BatchExecutionDescriptor(cg_mode=graph_mode, num_tokens=16, num_reqs=4)
    graph = MagicMock()
    manager.graphs = {desc: graph}
    offloader = MagicMock()
    monkeypatch.setattr(cudagraph_utils, "get_offloader", lambda: offloader)
    # Only hardware input preparation/dispatch are replaced. Both propose()
    # implementations and both run_fullgraph() implementations remain real.
    prepare_inputs = MagicMock()
    monkeypatch.setattr(upstream_dflash, "prepare_dflash_inputs", prepare_inputs)
    monkeypatch.setattr(upstream_dflash, "dispatch_cg_and_sync_dp", lambda *_args, **_kwargs: (desc, None))
    slots = {"draft.0": speculator.block_tables.slot_mappings[0]}
    monkeypatch.setattr(upstream_dflash, "build_slot_mappings_by_layer", lambda *_args: slots)
    host_metadata = {name: object() for name in speculator.attn_backends}
    skip_build = graph_mode == CUDAGraphMode.FULL and not mixed_backends

    def build_metadata(**kwargs):
        if skip_build:
            raise AssertionError("propose rebuilt metadata before FULL FlashAttn replay")
        assert kwargs["batch_desc"] is desc
        return host_metadata

    speculator._build_uniform_attn_metadata = MagicMock(side_effect=build_metadata)
    speculator.build_draft_attn_metadatas = MagicMock(
        side_effect=AssertionError("FlashAttn graph manager rebuilt host metadata") if not mixed_backends else None,
        return_value=[host_metadata],
    )
    manager._graph_replay = MagicMock(side_effect=AssertionError("unexpected legacy FIA update"))
    manager._updatable_graph_replay = MagicMock(
        side_effect=AssertionError("unexpected host graph update") if not mixed_backends else None
    )

    for step, (num_reqs, rejected) in enumerate([(2, [0, 2]), (1, [3]), (3, [1, 0, 2])], start=1):
        batch = SimpleNamespace(
            num_reqs=num_reqs,
            num_tokens=num_reqs,
            seq_lens_cpu_upper_bound=torch.full((num_reqs,), 256, dtype=torch.int32),
            query_start_loc=torch.arange(num_reqs + 1, dtype=torch.int32),
        )
        result = speculator.propose(
            input_batch=batch,
            attn_metadata={},
            slot_mappings={},
            last_hidden_states=torch.zeros(num_reqs, 8),
            aux_hidden_states=None,
            num_sampled=torch.ones(num_reqs, dtype=torch.int32),
            num_rejected=torch.tensor(rejected, dtype=torch.int32),
            last_sampled=torch.zeros(4, dtype=torch.int64),
            next_prefill_tokens=torch.zeros(4, dtype=torch.int64),
            temperature=speculator.temperature,
            seeds=speculator.seeds,
        )
        torch.testing.assert_close(result, speculator.draft_tokens[:num_reqs])
        assert prepare_inputs.call_count == step * len(backends)
        assert speculator._build_uniform_attn_metadata.call_count == (0 if skip_build else step)
        if skip_build:
            assert graph.replay.call_count == step
            assert offloader.sync_prev_onload.call_count == step
            speculator.build_draft_attn_metadatas.assert_not_called()
            manager._updatable_graph_replay.assert_not_called()
            torch.testing.assert_close(speculator._num_rejected[:num_reqs], torch.tensor(rejected, dtype=torch.int32))
        elif mixed_backends:
            assert speculator.build_draft_attn_metadatas.call_count == step
            manager._updatable_graph_replay.assert_called_with(desc, [host_metadata])
            graph.replay.assert_not_called()
        else:
            assert speculator._generate_draft.call_count == step
            assert speculator._generate_draft.call_args.args[2] is host_metadata
            assert speculator._generate_draft.call_args.kwargs["cudagraph_runtime_mode"] == graph_mode
            graph.replay.assert_not_called()
        manager._graph_replay.assert_not_called()
    assert speculator.supports_decode_metadata_update is (not mixed_backends)
