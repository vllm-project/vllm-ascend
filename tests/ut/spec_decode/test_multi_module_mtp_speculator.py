# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.spec_decode.multi_module_mtp.speculator import (
    MultiModuleMTPSpeculator,
)

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.spec_decode.multi_module_mtp import speculator as speculator_module
from vllm_ascend.worker.v2.spec_decode.multi_module_mtp.speculator import (
    AscendMultiModuleMTPSpeculator,
)


def _make_speculator(**attrs):
    """Build a speculator without running the heavy upstream __init__."""
    speculator = AscendMultiModuleMTPSpeculator.__new__(AscendMultiModuleMTPSpeculator)
    for name, value in attrs.items():
        setattr(speculator, name, value)
    return speculator


def test_build_draft_attn_metadatas_rebuilds_from_current_batch():
    """The replay hook rebuilds the draft metadata from the cached batch
    (DFlash/DSpark pattern) instead of reusing a cached one."""
    query_start_loc_np = np.array([0, 4, 8, 12, 16], dtype=np.int32)
    seq_lens_cpu_upper_bound = torch.arange(8, dtype=torch.int32)
    input_batch = SimpleNamespace(
        num_reqs=3,
        query_start_loc_np=query_start_loc_np,
        seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
    )
    speculator = _make_speculator(
        input_batch=input_batch,
        draft_attn_layer_names={"model.layers.45", "model.layers.46"},
    )
    built_metadata = {
        "model.layers.45": "m45",
        "model.layers.46": "m46",
        "model.layers.0": "target-only",
    }

    with (
        patch.object(
            AscendMultiModuleMTPSpeculator,
            "_build_attn_metadata",
            return_value=built_metadata,
        ) as mock_build,
        patch.object(speculator_module, "build_attn_metadata_wrapper", return_value=nullcontext()) as mock_wrapper,
    ):
        result = speculator.build_draft_attn_metadatas(
            num_reqs_padded=4, num_tokens_padded=16, is_draft_model_prefill=False
        )

    assert result == [{"model.layers.45": "m45", "model.layers.46": "m46"}]
    kwargs = mock_build.call_args.kwargs
    assert kwargs["num_reqs"] == 3
    assert kwargs["step"] == 0
    assert kwargs["query_start_loc_np"] is query_start_loc_np
    assert kwargs["seq_lens_cpu_upper_bound"] is seq_lens_cpu_upper_bound
    batch_desc = kwargs["batch_desc"]
    assert isinstance(batch_desc, BatchExecutionDescriptor)
    assert batch_desc.cg_mode == CUDAGraphMode.FULL
    assert batch_desc.num_tokens == 16
    assert batch_desc.num_reqs == 4
    mock_wrapper.assert_called_once()
    # The propose-built metadata is no longer cached on the instance.
    assert not hasattr(speculator, "_draft_attn_metadata")


def test_build_draft_attn_metadatas_requires_input_batch():
    """Replay without a cached batch fails loudly instead of building garbage."""
    speculator = _make_speculator(input_batch=None, draft_attn_layer_names=set())

    with pytest.raises(AssertionError):
        speculator.build_draft_attn_metadatas(num_reqs_padded=4, num_tokens_padded=16, is_draft_model_prefill=False)


@pytest.mark.parametrize(
    "any_prefilling,expected_state",
    [
        (True, AscendAttentionState.ChunkedPrefill),
        (False, AscendAttentionState.DecodeOnly),
    ],
)
def test_build_attn_metadata_selects_state_by_prefill(any_prefilling, expected_state):
    """Prefill-shaped windows build as ChunkedPrefill, decode as DecodeOnly."""
    is_prefilling_np = np.zeros(4, dtype=bool)
    is_prefilling_np[0] = any_prefilling
    input_batch = MagicMock(spec=AscendInputBatch)
    input_batch.is_prefilling_np = is_prefilling_np
    positions = torch.zeros(4)
    speculator = _make_speculator(
        input_batch=input_batch,
        input_buffers=SimpleNamespace(positions=positions),
        vllm_config=SimpleNamespace(parallel_config=object()),
    )
    batch_desc = SimpleNamespace(num_tokens=16)
    query_start_loc_np = np.array([0, 4, 8], dtype=np.int32)
    seq_lens_cpu_upper_bound = torch.arange(8, dtype=torch.int32)

    with (
        patch.object(speculator_module, "build_attn_metadata_factory", return_value=nullcontext()) as mock_factory,
        patch.object(
            MultiModuleMTPSpeculator, "_build_attn_metadata", return_value={"layer": "metadata"}
        ) as mock_parent,
    ):
        result = speculator._build_attn_metadata(2, batch_desc, query_start_loc_np, seq_lens_cpu_upper_bound, 0)

    assert result is mock_parent.return_value
    # The Ascend factory receives the prefill flags, the exact per-request
    # CPU lengths and the selected attention state.
    factory_args, factory_kwargs = mock_factory.call_args
    assert factory_args[0] is positions
    assert factory_args[1] == 16
    torch.testing.assert_close(factory_args[2], torch.from_numpy(is_prefilling_np))
    torch.testing.assert_close(factory_kwargs["seq_lens_cpu"], seq_lens_cpu_upper_bound[:2])
    assert factory_kwargs["attn_state"] == expected_state
    # All builder arguments pass through to the upstream implementation.
    mock_parent.assert_called_once_with(
        2, batch_desc, query_start_loc_np, seq_lens_cpu_upper_bound, 0, True, None, None
    )
