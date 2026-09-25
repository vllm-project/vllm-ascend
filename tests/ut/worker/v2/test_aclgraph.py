# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import SpeculatorCudaGraphManager

from vllm_ascend.compilation.updatable_graph import UpdatableGraph
from vllm_ascend.worker.v2.spec_decode.autoregressive.aclgraph import AutoRegressiveAclGraphManager


def _parent_init(
    self,
    vllm_config,
    device,
    cudagraph_mode,
    decode_query_len,
    lora_capture_cases=None,
):
    self._capture_descs = [object()]


def test_init_draft_prefill_graph():
    with (
        patch.object(SpeculatorCudaGraphManager, "__init__", _parent_init),
        patch.object(SpeculatorCudaGraphManager, "needs_capture", return_value=True),
        patch(
            "vllm_ascend.worker.v2.spec_decode.autoregressive.aclgraph.collect_sorted_captured_token_sizes",
            return_value=[4, 8],
        ),
        patch(
            "vllm_ascend.worker.v2.spec_decode.autoregressive.aclgraph.set_draft_graph_prefill_params"
        ) as set_prefill,
        patch("vllm_ascend.worker.v2.spec_decode.autoregressive.aclgraph.set_draft_graph_params") as set_decode,
    ):
        manager = AutoRegressiveAclGraphManager(object(), torch.device("cpu"), CUDAGraphMode.FULL, decode_query_len=2)
    assert manager.capture_sizes == [4, 8]
    assert manager.is_draft_model_prefill is True
    assert manager.speculator is None
    set_prefill.assert_called_once_with([4, 8])
    set_decode.assert_not_called()


def test_init_draft_decode_graph():
    with (
        patch.object(SpeculatorCudaGraphManager, "__init__", _parent_init),
        patch.object(SpeculatorCudaGraphManager, "needs_capture", return_value=True),
        patch(
            "vllm_ascend.worker.v2.spec_decode.autoregressive.aclgraph.collect_sorted_captured_token_sizes",
            return_value=[4, 8],
        ),
        patch(
            "vllm_ascend.worker.v2.spec_decode.autoregressive.aclgraph.set_draft_graph_prefill_params"
        ) as set_prefill,
        patch("vllm_ascend.worker.v2.spec_decode.autoregressive.aclgraph.set_draft_graph_params") as set_decode,
    ):
        manager = AutoRegressiveAclGraphManager(object(), torch.device("cpu"), CUDAGraphMode.FULL, decode_query_len=1)
    assert manager.is_draft_model_prefill is False
    set_decode.assert_called_once_with([4, 8])
    set_prefill.assert_not_called()


def test_run_fullgraph_uses_updatable_graph():
    manager = AutoRegressiveAclGraphManager.__new__(AutoRegressiveAclGraphManager)
    manager.update_stream = MagicMock()
    manager.is_draft_model_prefill = False
    backend = object()
    draft_vllm_config = object()
    draft_attn_metadatas = [{"draft": object()}]
    manager.speculator = SimpleNamespace(
        attn_backend=backend,
        draft_vllm_config=draft_vllm_config,
        build_draft_attn_metadatas=MagicMock(return_value=draft_attn_metadatas),
    )
    manager._updatable_graph_replay = MagicMock(return_value="result")
    manager._graph_replay = MagicMock()
    desc = SimpleNamespace(num_tokens=8, num_reqs=2)
    with patch(
        "vllm_ascend.worker.v2.spec_decode.autoregressive.aclgraph.use_updatable_graph",
        return_value=True,
    ):
        result = manager.run_fullgraph(desc)
    assert result == "result"
    manager.speculator.build_draft_attn_metadatas.assert_called_once_with(2, 8, False)
    manager._updatable_graph_replay.assert_called_once_with(
        desc,
        draft_attn_metadatas,
    )
    manager._graph_replay.assert_not_called()


def test_run_fullgraph_uses_legacy_graph():
    manager = AutoRegressiveAclGraphManager.__new__(AutoRegressiveAclGraphManager)
    manager.update_stream = MagicMock()
    manager.is_draft_model_prefill = False
    backend = object()
    draft_vllm_config = object()
    draft_attn_metadatas = [{"draft": object()}]
    manager.speculator = SimpleNamespace(
        attn_backend=backend,
        draft_vllm_config=draft_vllm_config,
        build_draft_attn_metadatas=MagicMock(return_value=draft_attn_metadatas),
    )
    manager._updatable_graph_replay = MagicMock()
    manager._graph_replay = MagicMock(return_value="result")
    desc = SimpleNamespace(num_tokens=8, num_reqs=2)
    with patch(
        "vllm_ascend.worker.v2.spec_decode.autoregressive.aclgraph.use_updatable_graph",
        return_value=False,
    ):
        result = manager.run_fullgraph(desc)
    assert result == "result"
    manager.speculator.build_draft_attn_metadatas.assert_called_once_with(2, 8, False)
    manager._graph_replay.assert_called_once_with(
        desc,
        backend,
        8,
        draft_vllm_config,
        draft_attn_metadatas,
    )
    manager._updatable_graph_replay.assert_not_called()


def test_capture_draft_prefill_delegates_to_parent():
    manager = AutoRegressiveAclGraphManager.__new__(AutoRegressiveAclGraphManager)
    manager.speculator = object()
    manager.is_draft_model_prefill = True
    forward_fn = MagicMock()
    model_state = object()
    input_buffers = object()
    block_tables = object()
    attn_groups = object()
    kv_cache_config = object()
    with (
        patch(
            "vllm_ascend.worker.v2.spec_decode.autoregressive.aclgraph.communicator_switch",
            return_value=nullcontext(),
        ),
        patch(
            "vllm_ascend.worker.v2.spec_decode.autoregressive.aclgraph.model_capture_wrapper",
            return_value=nullcontext(),
        ),
        patch.object(SpeculatorCudaGraphManager, "capture") as parent_capture,
    ):
        manager.capture(forward_fn, model_state, input_buffers, block_tables, attn_groups, kv_cache_config)
    parent_capture.assert_called_once_with(
        forward_fn,
        model_state,
        input_buffers,
        block_tables,
        attn_groups,
        kv_cache_config,
        progress_bar_desc="Capturing CUDA graphs",
    )


def test_capture_draft_decode_prepares_inputs_and_runs_forward():
    manager = AutoRegressiveAclGraphManager.__new__(AutoRegressiveAclGraphManager)
    manager.speculator = object()
    manager.is_draft_model_prefill = False
    manager.max_num_reqs = 3
    manager.dp_size = 2
    forward_fn = MagicMock()
    model_state = object()
    input_buffers = SimpleNamespace(seq_lens_cpu=torch.tensor([10, 20, 30], dtype=torch.int32))
    block_tables = object()
    attn_groups = object()
    kv_cache_config = object()
    desc = SimpleNamespace(num_tokens=4, num_reqs=None, cg_mode=CUDAGraphMode.FULL)

    def capture_side_effect(manager_arg, create_forward_fn, progress_bar_desc=None):
        runner = create_forward_fn(desc, False)
        runner(CUDAGraphMode.FULL)

    with (
        patch(
            "vllm_ascend.worker.v2.spec_decode.autoregressive.aclgraph.communicator_switch",
            return_value=nullcontext(),
        ),
        patch(
            "vllm_ascend.worker.v2.spec_decode.autoregressive.aclgraph.model_capture_wrapper",
            return_value=nullcontext(),
        ),
        patch("vllm_ascend.worker.v2.spec_decode.autoregressive.aclgraph.prepare_inputs_to_capture") as prepare_inputs,
        patch(
            "vllm_ascend.worker.v2.spec_decode.autoregressive.aclgraph.CudaGraphManager.capture",
            side_effect=capture_side_effect,
        ),
    ):
        manager.capture(forward_fn, model_state, input_buffers, block_tables, attn_groups, kv_cache_config)

    prepare_inputs.assert_called_once_with(
        3,
        4,
        model_state,
        input_buffers,
        block_tables,
        attn_groups,
        kv_cache_config,
        full_cudagraph=True,
    )
    args = forward_fn.call_args.args
    assert args[0] == 3
    assert args[1] is False
    assert args[2].cg_mode == CUDAGraphMode.FULL
    assert args[2].num_tokens == 4
    assert args[2].num_reqs == 3
    assert torch.equal(args[3], torch.tensor([4, 4], dtype=torch.int32))
    assert torch.equal(args[4], torch.tensor([10, 20, 30], dtype=torch.int32))


def test_graph_replay_updates_full_graph_params():
    manager = AutoRegressiveAclGraphManager.__new__(AutoRegressiveAclGraphManager)
    manager.update_stream = MagicMock()
    manager.is_draft_model_prefill = False
    draft_attn_metadatas = [{"draft": object()}]
    speculative_config = object()
    manager.speculator = SimpleNamespace(
        dp_size=2,
        model_state=SimpleNamespace(attn_metadata={"draft": object()}),
        speculative_config=speculative_config,
    )
    desc = SimpleNamespace(
        num_reqs=2,
        num_tokens=4,
        cg_mode=CUDAGraphMode.FULL,
    )
    attn_backend = object()
    draft_vllm_config = object()
    current_stream = object()
    forward_context = object()
    extra_ctx = SimpleNamespace()

    with (
        patch(
            "vllm_ascend.worker.v2.spec_decode.autoregressive.aclgraph.torch.npu.current_stream",
            return_value=current_stream,
        ),
        patch.object(
            SpeculatorCudaGraphManager,
            "run_fullgraph",
            return_value="result",
        ) as parent_replay,
        patch(
            "vllm_ascend.worker.v2.spec_decode.autoregressive.aclgraph.set_current_vllm_config",
            return_value=nullcontext(),
        ),
        patch(
            "vllm_ascend.worker.v2.spec_decode.autoregressive.aclgraph.set_forward_context",
            return_value=nullcontext(),
        ),
        patch(
            "vllm_ascend.worker.v2.spec_decode.autoregressive.aclgraph.get_forward_context",
            return_value=forward_context,
        ),
        patch(
            "vllm_ascend.worker.v2.spec_decode.autoregressive.aclgraph._EXTRA_CTX",
            extra_ctx,
        ),
        patch("vllm_ascend.worker.v2.spec_decode.autoregressive.aclgraph.update_full_graph_params") as update_params,
    ):
        result = manager._graph_replay(
            desc,
            attn_backend,
            4,
            draft_vllm_config,
            draft_attn_metadatas,
        )

    assert result == "result"
    assert extra_ctx.is_draft_model is True
    assert extra_ctx.is_draft_model_prefill is False
    manager.update_stream.wait_stream.assert_called_once_with(current_stream)
    parent_replay.assert_called_once_with(desc)
    update_params.assert_called_once_with(
        attn_backend,
        manager.update_stream,
        forward_context,
        4,
        draft_vllm_config,
        speculative_config,
        draft_attn_metadatas=draft_attn_metadatas,
    )


def test_updatable_graph_replay_updates_resolved_tasks():
    manager = AutoRegressiveAclGraphManager.__new__(AutoRegressiveAclGraphManager)
    manager.update_stream = MagicMock()
    manager.is_draft_model_prefill = False
    desc = MagicMock()
    desc.num_reqs = 2
    graph = MagicMock(spec=UpdatableGraph)
    resolved_tasks = object()
    graph.resolve_tasks.return_value = resolved_tasks
    manager.graphs = {desc: graph}
    fia_params = [{"layer_name": "draft"}]
    manager.speculator = SimpleNamespace(build_fia_params=MagicMock(return_value=fia_params))
    source = object()
    current_stream = object()
    draft_attn_metadatas = [{"draft": object()}]

    with (
        patch(
            "vllm_ascend.worker.v2.spec_decode.autoregressive.aclgraph.SharedSource",
            return_value=source,
        ) as shared_source,
        patch(
            "vllm_ascend.worker.v2.spec_decode.autoregressive.aclgraph.torch.npu.current_stream",
            return_value=current_stream,
        ),
        patch.object(
            SpeculatorCudaGraphManager,
            "run_fullgraph",
            return_value="result",
        ) as parent_replay,
    ):
        result = manager._updatable_graph_replay(
            desc,
            draft_attn_metadatas,
        )

    assert result == "result"
    manager.speculator.build_fia_params.assert_called_once_with(
        2,
        draft_attn_metadatas[0],
        False,
    )
    shared_source.assert_called_once_with(fia_params)
    graph.resolve_tasks.assert_called_once_with(source)
    manager.update_stream.wait_stream.assert_called_once_with(current_stream)
    parent_replay.assert_called_once_with(desc)
    graph.update.assert_called_once_with(
        manager.update_stream,
        resolved_tasks,
    )
