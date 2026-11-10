# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Use native graph descriptors and dispatch without allocating a device."""

from types import SimpleNamespace

import pytest
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)

from vllm_ascend.worker.v2.spec_decode.hardware_aware import extend_capture_descriptors


def make_manager():
    speculative_config = SimpleNamespace(
        num_speculative_tokens=7,
        uses_dynamic_speculative_decoding=lambda: False,
        use_dspark=lambda: True,
        draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(sample_from_anchor=True)),
    )
    manager = SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32, 64, 112, 128, 224, 256],
            max_cudagraph_capture_size=256,
        ),
        max_num_reqs=64,
        decode_query_len=7,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        varlen_decode=False,
        vllm_config=SimpleNamespace(
            speculative_config=speculative_config,
            additional_config={
                "dynamic_spec_config": {
                    "method": "dspark",
                    "physical_k": {"min_k": 3, "auto_tune": True},
                }
            },
        ),
        lora_capture_cases=[0],
        _capture_descs={},
        _candidates={},
        _maybe_ubatch_twin=lambda desc: None,
        _resolve_effective_loras=lambda count: count,
        _graphs_captured=True,
    )
    CudaGraphManager._init_candidates(manager)
    return manager


@pytest.mark.parametrize("batch,expected_tokens", [(4, 28), (6, 56), (7, 56), (8, 56)])
def test_native_dispatch_prefers_existing_smaller_k7_graph(batch, expected_tokens):
    manager = make_manager()
    extend_capture_descriptors(manager)

    selected = CudaGraphManager.dispatch(manager, batch, batch * 7, 7, 0)

    assert selected.cg_mode is CUDAGraphMode.FULL
    assert selected.num_tokens == expected_tokens


def test_native_dispatch_keeps_full_graph_ahead_of_smaller_piecewise():
    manager = make_manager()
    manager._candidates[(42, 0)].insert(
        0, BatchExecutionDescriptor(cg_mode=CUDAGraphMode.PIECEWISE, num_tokens=42, num_reqs=None)
    )
    extend_capture_descriptors(manager)

    selected = CudaGraphManager.dispatch(manager, 6, 42, 7, 0)

    assert selected.cg_mode is CUDAGraphMode.FULL
    assert selected.num_tokens == 56
