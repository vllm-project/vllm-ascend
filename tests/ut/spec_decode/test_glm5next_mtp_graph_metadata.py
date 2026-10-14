# SPDX-License-Identifier: Apache-2.0
"""CPU regression coverage for the TP1 A5 merged-draft graph contract."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode

import vllm_ascend.spec_decode.llm_base_proposer as base_module
import vllm_ascend.spec_decode.multi_kv_cache_group_proposer as module
from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily
from vllm_ascend.spec_decode.multi_kv_cache_group_proposer import AscendMultiKVCacheGroupMTPProposer


def _proposer(monkeypatch, requests=2):
    proposer = AscendMultiKVCacheGroupMTPProposer.__new__(AscendMultiKVCacheGroupMTPProposer)
    proposer.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_seqs=30),
        model_config=SimpleNamespace(max_model_len=1024),
    )
    proposer._uses_multi_group_kv_cache = proposer.use_cuda_graph = True
    proposer.dcp_size = 1
    proposer.parallel_drafting = proposer.use_compress = proposer.uses_mrope = False
    proposer.num_speculative_tokens = 5
    proposer.max_model_len = 1024
    proposer.device = torch.device("cpu")
    proposer.arange = torch.arange(65)
    proposer.token_arange_np = np.arange(65, dtype=np.int32)
    proposer.seq_lens_group = [torch.zeros(64, dtype=torch.int32) for _ in range(5)]
    proposer.query_start_loc_group = [torch.zeros(65, dtype=torch.int32) for _ in range(5)]
    proposer.slot_mapping_group = [torch.full((64,), -1, dtype=torch.int32) for _ in range(5)]
    proposer._mtp_num_actual_reqs = torch.tensor(requests, dtype=torch.int32)
    proposer._mtp_graph_common_metadata = SimpleNamespace()
    proposer.block_size = 128
    table = torch.tensor([[2, 3, 4, 5], [7, 8, 9, 10]], dtype=torch.int32)
    builder = SimpleNamespace(nope=True, build_for_drafting=lambda common, step: common)
    group = SimpleNamespace(kv_cache_group_id=0, layer_names=["main"], get_metadata_builder=lambda: builder)
    proposer.draft_attn_groups = [group]
    proposer._get_primary_draft_attn_group = lambda: group
    proposer._common_attn_metadata_for_draft_group = lambda common, *args: common
    proposer.runner = SimpleNamespace(
        sparse_kv_offload_enabled=False,
        input_batch=SimpleNamespace(block_table=[SimpleNamespace(get_device_tensor=lambda: table)]),
    )
    monkeypatch.setattr(module, "enable_dsa_cp", lambda: False)
    monkeypatch.setattr(module, "lmhead_tp_enable", lambda: False)
    monkeypatch.setattr(module, "should_skip_allreduce_across_dp_group", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        module,
        "get_current_hardware_profile",
        lambda: SimpleNamespace(device_adaptor_family=DeviceAdaptorFamily.FP8_OPTIMIZED),
    )
    return proposer, table


@pytest.mark.parametrize("tokens,expected", [(1, 1), (6, 1), (12, 2), (180, 30), (192, 30)])
def test_decode_bucket_is_request_capacity_not_verify_tokens(monkeypatch, tokens, expected):
    proposer, _ = _proposer(monkeypatch)
    assert proposer._get_in_graph_draft_token_capacity(tokens, CUDAGraphMode.FULL) == expected


@pytest.mark.parametrize(
    "feature", ["tp", "cp", "pcp", "sp", "dcp", "offload", "uniform_mc2", "eager", "mrope", "rope"]
)
def test_unsupported_topologies_keep_existing_metadata_path(monkeypatch, feature):
    proposer, _ = _proposer(monkeypatch)
    mode = CUDAGraphMode.FULL
    if feature == "tp":
        proposer.vllm_config.parallel_config.tensor_parallel_size = 2
    elif feature == "cp":
        monkeypatch.setattr(module, "enable_dsa_cp", lambda: True)
    elif feature == "pcp":
        proposer.vllm_config.parallel_config.prefill_context_parallel_size = 2
    elif feature == "sp":
        proposer.vllm_config.parallel_config.use_sequence_parallel_moe = True
    elif feature == "dcp":
        proposer.dcp_size = 2
    elif feature == "offload":
        proposer.runner.sparse_kv_offload_enabled = True
    elif feature == "uniform_mc2":
        monkeypatch.setattr(module, "should_skip_allreduce_across_dp_group", lambda *args, **kwargs: False)
    elif feature == "mrope":
        proposer.uses_mrope = True
    elif feature == "rope":
        proposer._get_primary_draft_attn_group().get_metadata_builder().nope = False
    else:
        mode = CUDAGraphMode.NONE
    assert proposer._get_in_graph_draft_token_capacity(12, mode) is None


def test_nonuniform_prefill_cannot_derive_request_count_from_verify_bucket(monkeypatch):
    proposer, _ = _proposer(monkeypatch)
    # Two short prefill requests can occupy eight token rows. floor(8/6)
    # cannot describe those two requests, even when FULL graphs are enabled.
    assert proposer._get_in_graph_draft_token_capacity(8, CUDAGraphMode.FULL, uniform_decode=False) is None


@pytest.mark.parametrize("actual", [0, 1, 2])
def test_later_metadata_has_dynamic_active_prefix_and_stable_step_buffers(monkeypatch, actual):
    proposer, table = _proposer(monkeypatch, actual)
    snapshots = []
    for step in range(1, 5):
        pos = torch.tensor([105 + step, 205 + step])
        metadata = proposer._build_in_graph_draft_metadata(step, 2, pos, None)["main"]
        expected_lengths = [int(pos[i]) + 1 if i < actual else 0 for i in range(2)]
        assert metadata.seq_lens.tolist() == expected_lengths
        assert metadata.query_start_loc.tolist() == [0, min(1, actual), actual]
        assert metadata.seq_lens_cpu is None and metadata._seq_lens_cpu is None
        assert metadata.num_actual_tokens_tensor is proposer._mtp_num_actual_reqs
        assert metadata.max_seq_len == 1024
        for row in range(2):
            expected = int(table[row, pos[row] // 128]) * 128 + int(pos[row] % 128) if row < actual else -1
            assert metadata.slot_mapping[row].item() == expected
        snapshots.append((metadata, metadata.seq_lens.clone()))
    assert len({m.seq_lens.data_ptr() for m, _ in snapshots}) == 4
    assert len({m.slot_mapping.data_ptr() for m, _ in snapshots}) == 4
    for metadata, values in snapshots:
        torch.testing.assert_close(metadata.seq_lens, values)
    old = snapshots[-1][0]
    proposer._mtp_num_actual_reqs.zero_()
    empty = proposer._build_in_graph_draft_metadata(4, 2, torch.tensor([200, 300]), None)["main"]
    assert empty.seq_lens.data_ptr() == old.seq_lens.data_ptr()
    assert empty.seq_lens.tolist() == [0, 0]
    assert empty.slot_mapping.tolist() == [-1, -1]


def test_out_of_range_positions_never_write_kv(monkeypatch):
    proposer, _ = _proposer(monkeypatch)
    metadata = proposer._build_in_graph_draft_metadata(1, 2, torch.tensor([0, 205]), torch.tensor([True, False]))[
        "main"
    ]
    assert metadata.slot_mapping[0].item() == -1
    assert metadata.seq_lens.tolist() == [1, 206]


def test_graph_metadata_reads_persistent_tables_and_keeps_group_slots_separate(monkeypatch):
    proposer, primary_table = _proposer(monkeypatch)
    del proposer._common_attn_metadata_for_draft_group
    proposer.kv_cache_gid = 0
    proposer._draft_block_table_width = lambda group: {0: 4, 1: 8, 2: 1}[group.kv_cache_group_id]
    proposer._multi_group_slot_mapping_buffers = {(gid, 1): torch.empty(64, dtype=torch.int32) for gid in (1, 2)}

    proposer._multi_group_block_table_buffers = {
        gid: torch.zeros(64, width, dtype=torch.int32) for gid, width in ((0, 4), (1, 8), (2, 1))
    }

    class Table:
        def __init__(self, size, ids, width):
            self.size = size
            self.is_circular_group = False
            self.data = torch.tensor(ids, dtype=torch.int32)[:, None].expand(2, width).clone()
            self.slot_mapping = SimpleNamespace(gpu=torch.full((64,), -1, dtype=torch.int32))

        def get_device_tensor(self):
            return self.data

        def compute_slot_mapping(self, requests, query, positions):
            for row in range(requests):
                self.slot_mapping.gpu[row] = (
                    self.data[row, positions[row] // self.size] * self.size + positions[row] % self.size
                    if query[row + 1] > query[row]
                    else -1
                )

    proposer.runner.input_batch.block_table.extend([Table(32, [9, 19], 8), Table(256, [11, 12], 1)])
    for gid in (1, 2):
        builder = SimpleNamespace(build_for_drafting=lambda common, step: common)
        proposer.draft_attn_groups.append(
            SimpleNamespace(
                kv_cache_group_id=gid, layer_names=[f"cache{gid}"], get_metadata_builder=lambda b=builder: b
            )
        )
    positions = torch.tensor([106, 206])
    output = proposer._build_in_graph_draft_metadata(1, 2, positions, None)
    assert output["main"].slot_mapping.tolist() == [2 * 128 + 106, 8 * 128 + 78]
    assert output["cache1"].slot_mapping[:2].tolist() == [9 * 32 + 10, 19 * 32 + 14]
    assert output["cache2"].slot_mapping[:2].tolist() == [11 * 256 + 106, 12 * 256 + 206]
    assert bool((output["cache1"].slot_mapping[2:] == -1).all())
    assert len({metadata.slot_mapping.data_ptr() for metadata in output.values()}) == 3
    primary_table[0, 0] = 17
    proposer._mtp_num_actual_reqs.fill_(1)
    output = proposer._build_in_graph_draft_metadata(1, 2, positions, None)
    assert output["main"].slot_mapping.tolist() == [17 * 128 + 106, -1]
    assert output["cache1"].slot_mapping[:2].tolist() == [9 * 32 + 10, -1]


def test_merged_draft_preserves_first_pass_then_runs_four_small_steps(monkeypatch):
    proposer, _ = _proposer(monkeypatch)
    proposer.method = "mtp"
    proposer.input_ids = torch.arange(64, dtype=torch.int32)
    proposer.positions = torch.arange(64, dtype=torch.int32) + 100
    proposer.hidden_states = torch.ones(64, 4)
    proposer._get_positions = lambda count: proposer.positions[:count]
    proposer._set_positions = lambda count, positions: proposer.positions[:count].copy_(positions)
    proposer._share_mtp_indices = proposer.supports_mm_inputs = False
    proposer.pass_hidden_states_to_model = True
    proposer.max_model_len = 1024
    proposer._sample_draft_from_logits = lambda logits, sampling: (logits.argmax(-1), None)
    context = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.FULL, attn_metadata={"main": None})
    monkeypatch.setattr(base_module, "get_forward_context", lambda: context)
    monkeypatch.setattr(base_module, "_EXTRA_CTX", context)
    monkeypatch.setattr(base_module, "lmhead_tp_enable", lambda: False)
    seen = []

    class Model:
        model = None

        def __call__(self, input_ids, positions, hidden_states, **kwargs):
            seen.append((len(input_ids), positions.clone(), context.attn_metadata))
            return hidden_states + 1, hidden_states + 1

        def compute_logits(self, hidden):
            return torch.stack((hidden[:, 0], -hidden[:, 0]), dim=1)

    proposer.model = Model()
    result = proposer._run_merged_draft(
        num_input_tokens=12,
        batch_size=2,
        token_indices_to_sample=torch.tensor([5, 11]),
        target_positions=proposer.positions[:12],
        inputs_embeds=None,
        multi_steps_attn_metadata=[context.attn_metadata],
        num_tokens=12,
        sampling_metadata=SimpleNamespace(),
    )
    assert result.shape == (2, 5)
    assert [rows for rows, _, _ in seen] == [12, 2, 2, 2, 2]
    assert context.padded_num_tokens == 2
    for step, (_, positions, metadata) in enumerate(seen[1:], start=1):
        assert positions.tolist() == [105 + step, 111 + step]
        assert metadata["main"].seq_lens.tolist() == [106 + step, 112 + step]
