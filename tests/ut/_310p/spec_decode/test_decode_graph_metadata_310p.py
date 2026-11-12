# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""310P decode graphs retain addresses and refresh their contents before replay."""

import gc
import weakref
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor, CudaGraphManager

from vllm_ascend._310p.worker.v2.spec_decode import aclgraph as aclgraph_module
from vllm_ascend._310p.worker.v2.spec_decode.aclgraph import AutoRegressiveAclGraphManager310


def test_decode_capture_retains_only_recorded_metadata():
    manager = object.__new__(AutoRegressiveAclGraphManager310)
    manager.max_num_reqs = 2
    manager.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=1))
    tensor_refs = []

    def build_metadata(**kwargs):
        del kwargs
        seq_lens = torch.tensor([2, 2], dtype=torch.int32)
        tensor_refs.append(weakref.ref(seq_lens))
        return {"layer": SimpleNamespace(seq_lens=seq_lens, block_tables=torch.zeros((2, 3), dtype=torch.int32))}

    manager.speculator = SimpleNamespace(
        device=torch.device("cpu"),
        block_tables=SimpleNamespace(get_dummy_slot_mappings=lambda count: torch.zeros(count)),
        kv_cache_config=object(),
        current_draft_step=torch.zeros(1, dtype=torch.int32),
        _build_draft_attn_metadata=build_metadata,
    )
    desc = BatchExecutionDescriptor(cg_mode=CUDAGraphMode.FULL, num_tokens=2, num_reqs=2)
    warmup_desc = BatchExecutionDescriptor(cg_mode=CUDAGraphMode.FULL, num_tokens=1, num_reqs=1)

    def capture(self, create_forward_fn, **kwargs):
        del self, kwargs
        create_forward_fn(warmup_desc, True)
        assert not manager._decode_capture_attn_metadata
        create_forward_fn(desc, False)

    with (
        patch.object(aclgraph_module, "communicator_switch", return_value=nullcontext()),
        patch.object(aclgraph_module, "model_capture_wrapper", return_value=nullcontext()),
        patch.object(aclgraph_module, "prepare_inputs_to_capture"),
        patch.object(aclgraph_module, "build_slot_mappings_by_layer", return_value={}),
        patch.object(aclgraph_module, "set_draft_step_host"),
        patch.object(aclgraph_module.torch.npu, "current_stream"),
        patch.object(CudaGraphManager, "capture", capture),
    ):
        manager._capture_decode_per_step(
            object(),
            SimpleNamespace(seq_lens_cpu=torch.zeros(2, dtype=torch.int32)),
            object(),
            [],
            object(),
            "test capture",
        )
    gc.collect()
    assert set(manager._decode_capture_attn_metadata) == {desc}
    assert tensor_refs[0]() is None
    assert tensor_refs[1]() is manager._decode_capture_attn_metadata[desc]["layer"].seq_lens


@pytest.mark.parametrize("shared_block_table", [False, True])
def test_decode_replay_updates_captured_tensors_before_launch(shared_block_table):
    manager = object.__new__(AutoRegressiveAclGraphManager310)
    manager.is_draft_model_prefill = False
    desc = BatchExecutionDescriptor(cg_mode=CUDAGraphMode.FULL, num_tokens=2, num_reqs=2)
    captured = SimpleNamespace(
        seq_lens=torch.tensor([1, 1], dtype=torch.int32),
        block_tables=torch.zeros((2, 3), dtype=torch.int32),
    )
    runtime = SimpleNamespace(
        seq_lens=torch.tensor([11, 12], dtype=torch.int32),
        block_tables=captured.block_tables if shared_block_table else torch.ones((2, 3), dtype=torch.int32),
    )
    pending = {"layer": runtime, "mamba": None, "absent": None}
    manager._decode_capture_attn_metadata = {
        desc: {"layer": captured, "mamba": None, "absent": SimpleNamespace()},
    }
    model_state = SimpleNamespace(_refresh_capture_seq_lens=MagicMock())
    manager.speculator = SimpleNamespace(
        model_state=model_state,
        input_buffers=SimpleNamespace(seq_lens=runtime.seq_lens),
        _pending_draft_attn_metadata=pending,
    )
    seq_ptr = captured.seq_lens.data_ptr()
    block_ptr = captured.block_tables.data_ptr()
    result = torch.tensor([42])

    def replay(self, batch_desc):
        del self
        assert batch_desc == desc
        torch.testing.assert_close(captured.seq_lens, runtime.seq_lens)
        torch.testing.assert_close(captured.block_tables, runtime.block_tables)
        return result

    with patch.object(CudaGraphManager, "run_fullgraph", replay):
        assert manager.run_fullgraph(desc) is result
    assert captured.seq_lens.data_ptr() == seq_ptr
    assert captured.block_tables.data_ptr() == block_ptr
    assert model_state.attn_metadata is pending
    model_state._refresh_capture_seq_lens.assert_called_once_with(runtime.seq_lens)
