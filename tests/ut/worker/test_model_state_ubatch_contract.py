# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode

from vllm_ascend._310p.worker.v2.model_state import Ascend310PMambaHybridModelState, Ascend310PModelState
from vllm_ascend.worker.v2.model_states.default import AscendModelState
from vllm_ascend.worker.v2.model_states.mamba_hybrid import AscendMambaHybridModelState


@pytest.mark.parametrize(
    "model_cls, metadata_owner",
    [
        (AscendModelState, AscendModelState),
        (AscendMambaHybridModelState, AscendMambaHybridModelState),
        (Ascend310PModelState, AscendModelState),
        (Ascend310PMambaHybridModelState, AscendMambaHybridModelState),
    ],
)
@pytest.mark.parametrize("ubatch_idx", [0, 1])
def test_prepare_attn_accepts_single_batch_contract(model_cls, metadata_owner, ubatch_idx):
    state = model_cls.__new__(model_cls)
    state.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1),
        num_speculative_tokens=0,
    )
    state.max_model_len = 16
    state.pcp_manager = None
    batch = SimpleNamespace(
        num_reqs=1,
        num_tokens=2,
        query_start_loc_np=np.array([0, 2], dtype=np.int32),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        is_prefilling_np=np.array([True]),
        num_scheduled_tokens=np.array([2], dtype=np.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        seq_lens_np=np.array([2], dtype=np.int32),
        dcp_local_seq_lens=None,
        positions=torch.arange(2),
        attn_state=None,
    )
    metadata = {"layer": object()}
    with patch(f"{metadata_owner.__module__}.build_attn_metadata", return_value=metadata) as build:
        args = (batch, CUDAGraphMode.NONE, (), torch.empty(0), [], SimpleNamespace())
        if ubatch_idx:
            with pytest.raises(AssertionError, match="DBO is not supported"):
                state.prepare_attn(*args, ubatch_idx=ubatch_idx)
            build.assert_not_called()
        else:
            assert state.prepare_attn(*args, ubatch_idx=0) is metadata
            assert state.prepare_attn(*args) is metadata
            assert build.call_count == 2
