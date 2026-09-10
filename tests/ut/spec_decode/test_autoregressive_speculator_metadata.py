# SPDX-License-Identifier: Apache-2.0
"""Regression tests for speculative decode attention metadata."""

from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch

import vllm_ascend.worker.v2.spec_decode.autoregressive.speculator as speculator_module
from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import AscendAutoRegressiveSpeculator


def _new_speculator(*, mla: bool) -> SimpleNamespace:
    speculator = SimpleNamespace()
    speculator.attn_architecture = "MLA" if mla else None
    speculator.input_batch = SimpleNamespace(
        num_reqs=2,
        seq_lens_np=np.array([5, 7, 0, 0], dtype=np.int32),
        seq_lens_cpu_upper_bound=torch.tensor([5, 7, 0, 0], dtype=torch.int32),
    )
    speculator.input_buffers = SimpleNamespace(
        positions=torch.arange(4),
        draft_seq_lens_cpus=[torch.zeros(4, dtype=torch.int32) for _ in range(2)],
    )
    speculator.max_model_len = 32
    speculator._get_seq_lens_cpu = MethodType(AscendAutoRegressiveSpeculator._get_seq_lens_cpu, speculator)
    speculator._calc_next_seq_lens_cpu = MethodType(AscendAutoRegressiveSpeculator._calc_next_seq_lens_cpu, speculator)
    speculator._init_decode_draft_attn_metadatas = MethodType(
        AscendAutoRegressiveSpeculator._init_decode_draft_attn_metadatas, speculator
    )
    speculator._update_decode_attn_metadata = MethodType(
        AscendAutoRegressiveSpeculator._update_decode_attn_metadata, speculator
    )
    return speculator


def test_mla_fullgraph_rebuilds_per_step_decode_metadata(monkeypatch) -> None:
    monkeypatch.setattr(speculator_module, "vllm_version_is", lambda _: True)
    speculator = _new_speculator(mla=True)
    prefill_metadata = SimpleNamespace(decode=None)
    speculator.model_state = SimpleNamespace(attn_metadata={"layer": prefill_metadata})
    speculator.draft_attn_layer_names = {"layer"}

    padded_block_table = object()
    rebuilt_metadata = SimpleNamespace(
        attn_state=None,
        seq_lens_cpu=torch.zeros(4, dtype=torch.int32),
        decode=SimpleNamespace(
            seq_lens_list=[],
            actual_seq_lengths_q=None,
            block_table=padded_block_table,
        ),
    )
    speculator._build_draft_attn_metadata = MagicMock(return_value={"layer": rebuilt_metadata})

    per_step_metadata = AscendAutoRegressiveSpeculator.build_draft_attn_metadatas(speculator, 4, False)

    speculator._build_draft_attn_metadata.assert_called_once_with(
        num_reqs=2,
        num_reqs_padded=4,
        num_tokens_padded=4,
    )
    assert len(per_step_metadata) == 2

    first_decode = per_step_metadata[0]["layer"].decode
    second_decode = per_step_metadata[1]["layer"].decode
    assert first_decode is not None
    assert second_decode is not None
    assert first_decode is not second_decode
    assert first_decode is not rebuilt_metadata.decode
    assert second_decode is not rebuilt_metadata.decode
    assert first_decode.seq_lens_list == [6, 8, 0, 0]
    assert second_decode.seq_lens_list == [7, 9, 0, 0]
    assert first_decode.actual_seq_lengths_q == [1, 2, 3, 4]
    assert second_decode.actual_seq_lengths_q == [1, 2, 3, 4]
    assert first_decode.block_table is padded_block_table
    assert second_decode.block_table is padded_block_table


def test_gqa_decode_metadata_remains_top_level() -> None:
    speculator = _new_speculator(mla=False)
    metadata = SimpleNamespace(
        seq_lens_cpu=torch.zeros(4, dtype=torch.int32),
        seq_lens_list=[],
        actual_seq_lengths_q=None,
    )

    AscendAutoRegressiveSpeculator._update_decode_attn_metadata(speculator, {"layer": metadata}, step=1, num_reqs=2)

    assert metadata.seq_lens_list == [6, 8, 0, 0]
    assert metadata.actual_seq_lengths_q == [1, 2, 3, 4]


def test_draft_metadata_factory_injects_positions_and_restores_builder(monkeypatch) -> None:
    original_builder = MagicMock(return_value="metadata")
    monkeypatch.setattr(speculator_module.vllm_speculator, "build_attn_metadata", original_builder)
    positions = torch.arange(8)

    with speculator_module.build_draft_attn_metadata_factory(positions, 3):
        patched_builder = speculator_module.vllm_speculator.build_attn_metadata
        assert patched_builder is not original_builder
        assert patched_builder(example=True) == "metadata"

    assert speculator_module.vllm_speculator.build_attn_metadata is original_builder
    _, kwargs = original_builder.call_args
    assert kwargs["example"] is True
    assert torch.equal(kwargs["positions"], positions[:3])
