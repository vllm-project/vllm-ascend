# SPDX-License-Identifier: Apache-2.0
"""Regression tests for autoregressive speculator sequence lengths."""

from types import MethodType, SimpleNamespace

import numpy as np
import torch

from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import (
    AscendAutoRegressiveSpeculator,
)


def test_decode_metadata_pads_compact_seq_lens_to_graph_batch_size() -> None:
    speculator = SimpleNamespace(
        attn_architecture=None,
        input_batch=SimpleNamespace(seq_lens_np=np.array([5, 7], dtype=np.int32)),
        max_model_len=32,
    )
    speculator._get_seq_lens_cpu = MethodType(AscendAutoRegressiveSpeculator._get_seq_lens_cpu, speculator)
    speculator._calc_next_seq_lens_cpu = MethodType(AscendAutoRegressiveSpeculator._calc_next_seq_lens_cpu, speculator)
    metadata = SimpleNamespace(
        seq_lens_cpu=torch.full((8,), -1, dtype=torch.int32),
        seq_lens_list=[],
        actual_seq_lengths_q=None,
    )

    AscendAutoRegressiveSpeculator._update_decode_attn_metadata(
        speculator,
        {"layer": metadata},
        step=1,
    )

    assert torch.equal(
        metadata.seq_lens_cpu,
        torch.tensor([6, 8, 0, 0, 0, 0, 0, 0], dtype=torch.int32),
    )
    assert metadata.seq_lens_list == [6, 8, 0, 0, 0, 0, 0, 0]
    assert metadata.actual_seq_lengths_q == [1, 2, 3, 4, 5, 6, 7, 8]
