# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FIA's valid-length list must not include unwritten DSpark lookahead slots."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

import vllm_ascend.worker.v2.spec_decode.dspark.speculator as speculator_module
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator


def _spec(lengths, *, architecture="MLA", use_dcp=False):
    spec = AscendDSparkSpeculator.__new__(AscendDSparkSpeculator)
    spec.attn_architecture = architecture
    spec.use_dcp = use_dcp
    spec.input_buffers = SimpleNamespace(seq_lens=torch.tensor(lengths, dtype=torch.int32), positions=object())
    spec.target_input_buffers = SimpleNamespace(seq_lens_cpu=torch.tensor([99999], dtype=torch.int32))
    spec.max_model_len = 131072
    return spec


@pytest.mark.parametrize("length,padded", [(73, 1), (22666, 2), (81, 4), (22667, 4)])
def test_exact_valid_lengths_ignore_target_upper_bound(length, padded):
    # The observed target upper bound + width overstated the draft by 1 or 8.
    spec = _spec([length, 99999, 99999, 99999])
    actual, prefilling = spec._prepare_draft_dcp_metadata_inputs(1, padded, 8)
    assert actual is not None
    assert actual.device.type == "cpu"
    assert actual.dtype == torch.int32
    assert actual.tolist() == [length] + [0] * (padded - 1)
    assert not prefilling.any()
    assert spec.input_buffers.seq_lens.tolist() == [length, 99999, 99999, 99999]


def test_exact_lengths_refresh_after_rejection():
    spec = _spec([22666, 19051, 99999, 99999])
    first, _ = spec._prepare_draft_dcp_metadata_inputs(2, 4, 8)
    spec.input_buffers.seq_lens[:2].copy_(torch.tensor([22667, 19059]))
    second, _ = spec._prepare_draft_dcp_metadata_inputs(2, 4, 8)
    assert first is not None and second is not None
    assert first.tolist() == [22666, 19051, 0, 0]
    assert second.tolist() == [22667, 19059, 0, 0]


def test_padded_idle_has_zero_valid_lengths():
    actual, prefilling = _spec([99999] * 4)._prepare_draft_dcp_metadata_inputs(0, 4, 8)
    assert actual is not None
    assert actual.tolist() == [0] * 4
    assert not prefilling.any()


@pytest.mark.parametrize("architecture", ["GQA", "SFA", None])
def test_non_mla_unchanged(architecture):
    actual, prefilling = _spec([73], architecture=architecture)._prepare_draft_dcp_metadata_inputs(1, 2, 8)
    assert actual is None
    assert not prefilling.any()


def test_dcp_existing_delegate():
    spec = _spec([73], use_dcp=True)
    expected = torch.tensor([73, 0], dtype=torch.int32)
    spec.dcp_manager = SimpleNamespace(prepare_draft_dcp_metadata_inputs=MagicMock(return_value=(expected, object())))
    result = spec._prepare_draft_dcp_metadata_inputs(1, 2, 8)
    assert result[0] is expected
    assert spec.dcp_manager.prepare_draft_dcp_metadata_inputs.call_args.kwargs["target_seq_lens_cpu"] is (
        spec.target_input_buffers.seq_lens_cpu
    )


@pytest.mark.parametrize("entrypoint", ["regular", "ACL_update"])
def test_regular_and_graph_metadata_get_same_exact_cpu_view(monkeypatch, entrypoint):
    spec = _spec([22666, 99999, 99999, 99999])
    spec.num_query_per_req = 8
    spec.arange_np = np.arange(5, dtype=np.int32)
    spec.input_batch = SimpleNamespace(num_reqs=1)
    spec._group_causal = False
    spec.vllm_config = SimpleNamespace(parallel_config=object())
    monkeypatch.setattr(AscendDSparkSpeculator, "attn_vllm_config", property(lambda self: self.vllm_config))
    decode = SimpleNamespace(actual_seq_lengths_q=[8])
    metadata = {"draft.0": SimpleNamespace(decode=decode)}
    monkeypatch.setattr(DSparkSpeculator, "_build_attn_metadata", MagicMock(return_value=metadata))
    monkeypatch.setattr(speculator_module, "build_attn_metadata_wrapper", nullcontext)
    supplied = []

    @contextmanager
    def factory(*args, **kwargs):
        supplied.append(kwargs["seq_lens_cpu"])
        yield

    monkeypatch.setattr(speculator_module, "build_attn_metadata_factory", factory)
    # The upstream parent still adds width to this optimistic value. The
    # Ascend factory must supply the independent, exact draft-length view.
    upper_bound = torch.tensor([22666], dtype=torch.int32)
    if entrypoint == "ACL_update":
        result = spec.build_draft_attn_metadatas(4, upper_bound)[0]
    else:
        result = spec._build_uniform_attn_metadata(
            num_reqs=1,
            batch_desc=BatchExecutionDescriptor(cg_mode=CUDAGraphMode.FULL, num_tokens=32, num_reqs=4),
            num_query_per_req=8,
            seq_lens_cpu_upper_bound=upper_bound,
            step=8,
            causal=False,
        )
    assert result is metadata
    assert supplied and all(value is not None and value.tolist() == [22666, 0, 0, 0] for value in supplied)
    assert upper_bound.tolist() == [22666]
    assert decode.actual_seq_lengths_q == [8, 16, 24, 32]
