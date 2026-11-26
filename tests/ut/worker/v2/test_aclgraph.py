# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu import cudagraph_utils

from vllm_ascend import ascend_forward_context, utils
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.worker.v2.spec_decode.dflash.aclgraph import DFlashAclGraphManager
from vllm_ascend.worker.v2.spec_decode.target_dependent_ar import aclgraph
from vllm_ascend.worker.v2.spec_decode.target_dependent_ar.aclgraph import (
    AutoRegressiveAclGraphManager,
    SpeculatorCudaGraphManager,
    filter_draft_full_aclgraph_sizes,
)


def _parent_init(
    self,
    vllm_config,
    device,
    cudagraph_mode,
    decode_query_lens,
    lora_capture_cases=None,
    varlen_decode=False,
    **kwargs,
):
    """Mock the parent graph manager initialization."""
    self._capture_descs = [object()]
    self.cudagraph_mode = cudagraph_mode


def _graph_config(pcp_size: int = 1) -> SimpleNamespace:
    return SimpleNamespace(parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp_size))


def _query_shape(query_len):
    return {"decode_query_lens": {query_len: 8}}


def test_init_draft_prefill_graph():
    """Verify initialization for the draft prefill graph."""
    with (
        patch.object(SpeculatorCudaGraphManager, "__init__", _parent_init),
        patch.object(SpeculatorCudaGraphManager, "needs_capture", return_value=True),
        patch(
            "vllm_ascend.worker.v2.spec_decode.target_dependent_ar.aclgraph.collect_sorted_captured_token_sizes",
            return_value=[4, 8],
        ),
        patch(
            "vllm_ascend.worker.v2.spec_decode.target_dependent_ar.aclgraph.set_draft_graph_prefill_params"
        ) as set_prefill,
        patch("vllm_ascend.worker.v2.spec_decode.target_dependent_ar.aclgraph.set_draft_graph_params") as set_decode,
    ):
        manager = AutoRegressiveAclGraphManager(
            _graph_config(), torch.device("cpu"), CUDAGraphMode.FULL, **_query_shape(2)
        )
    assert manager.capture_sizes == [4, 8]
    assert manager.is_draft_model_prefill is True
    assert manager.speculator is None
    set_prefill.assert_called_once_with([4, 8])
    set_decode.assert_not_called()


@pytest.mark.parametrize(
    ("decode_query_len", "mode", "expected"),
    [
        (2, CUDAGraphMode.FULL_AND_PIECEWISE, CUDAGraphMode.PIECEWISE),
        (2, CUDAGraphMode.FULL_DECODE_ONLY, CUDAGraphMode.NONE),
        (1, CUDAGraphMode.FULL_DECODE_ONLY, CUDAGraphMode.FULL_DECODE_ONLY),
    ],
)
def test_sharded_pcp_draft_prefill_avoids_full_graphs(decode_query_len, mode, expected):
    """Sharded draft prefill stages inputs and all-gathers outside the forward."""
    with (
        patch.object(SpeculatorCudaGraphManager, "__init__", _parent_init),
        patch.object(SpeculatorCudaGraphManager, "needs_capture", return_value=False),
        patch(
            "vllm_ascend.worker.v2.spec_decode.target_dependent_ar.aclgraph.collect_sorted_captured_token_sizes",
            return_value=[4],
        ),
    ):
        manager = AutoRegressiveAclGraphManager(
            _graph_config(pcp_size=2), torch.device("cpu"), mode, **_query_shape(decode_query_len)
        )
    assert manager.cudagraph_mode == expected


def test_init_draft_decode_graph():
    """Verify initialization for the draft decode graph."""
    with (
        patch.object(SpeculatorCudaGraphManager, "__init__", _parent_init),
        patch.object(SpeculatorCudaGraphManager, "needs_capture", return_value=True),
        patch(
            "vllm_ascend.worker.v2.spec_decode.target_dependent_ar.aclgraph.collect_sorted_captured_token_sizes",
            return_value=[4, 8],
        ),
        patch(
            "vllm_ascend.worker.v2.spec_decode.target_dependent_ar.aclgraph.set_draft_graph_prefill_params"
        ) as set_prefill,
        patch("vllm_ascend.worker.v2.spec_decode.target_dependent_ar.aclgraph.set_draft_graph_params") as set_decode,
    ):
        manager = AutoRegressiveAclGraphManager(object(), torch.device("cpu"), CUDAGraphMode.FULL, **_query_shape(1))
    assert manager.is_draft_model_prefill is False
    set_decode.assert_called_once_with([4, 8])
    set_prefill.assert_not_called()


@pytest.fixture
def graph_runtime(monkeypatch):
    monkeypatch.setattr(cudagraph_utils, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True))
    monkeypatch.setattr(cudagraph_utils.current_platform, "get_global_graph_pool", lambda: None)
    monkeypatch.setattr(cudagraph_utils, "is_breakable_cudagraph_enabled", lambda *args: False)
    monkeypatch.setattr(aclgraph, "set_draft_graph_params", Mock())
    monkeypatch.setattr(aclgraph, "set_draft_graph_prefill_params", Mock())
    monkeypatch.setattr("vllm_ascend.worker.v2.spec_decode.dflash.aclgraph.set_draft_graph_params", Mock())


@pytest.mark.parametrize(
    "draft_moe,method,query_len,mode,limit,expected_max_requests,expected_mode",
    [
        (True, MoECommType.MC2, 1, CUDAGraphMode.FULL_DECODE_ONLY, 16, 16, CUDAGraphMode.FULL_DECODE_ONLY),
        (True, MoECommType.MC2, 1, CUDAGraphMode.FULL_DECODE_ONLY, 8, 8, CUDAGraphMode.FULL_DECODE_ONLY),
        (True, MoECommType.FUSED_MC2, 1, CUDAGraphMode.FULL_DECODE_ONLY, 512, 512, CUDAGraphMode.FULL_DECODE_ONLY),
        (True, MoECommType.ALLGATHER, 1, CUDAGraphMode.FULL_DECODE_ONLY, 512, 512, CUDAGraphMode.FULL_DECODE_ONLY),
        (False, MoECommType.MC2, 1, CUDAGraphMode.FULL_DECODE_ONLY, 16, 512, CUDAGraphMode.FULL_DECODE_ONLY),
        (True, MoECommType.MC2, 4, CUDAGraphMode.FULL_DECODE_ONLY, 16, 4, CUDAGraphMode.FULL_DECODE_ONLY),
        (True, MoECommType.MC2, 1, CUDAGraphMode.NONE, 16, 512, CUDAGraphMode.NONE),
        (True, MoECommType.ALLTOALL, 1, CUDAGraphMode.FULL_DECODE_ONLY, 512, 0, CUDAGraphMode.NONE),
        (True, MoECommType.MC2, 4, CUDAGraphMode.FULL_AND_PIECEWISE, 2, 0, CUDAGraphMode.NONE),
        (True, MoECommType.MC2, 8, CUDAGraphMode.FULL_AND_PIECEWISE, 4, 0, CUDAGraphMode.PIECEWISE),
        (True, MoECommType.MC2, 1, CUDAGraphMode.FULL_AND_PIECEWISE, 2, 1, CUDAGraphMode.FULL_DECODE_ONLY),
    ],
)
def test_draft_decode_capture_respects_moe_comm_policy(
    monkeypatch, graph_runtime, draft_moe, method, query_len, mode, limit, expected_max_requests, expected_mode
):
    sizes = [1, 4, 8, 16, 32, 512]
    original_sizes = sizes.copy()
    compilation = SimpleNamespace(
        prefill_cudagraph_capture_token_sizes=[4, 8, 12, 16],
        decode_cudagraph_capture_request_sizes=[1, 2, 3, 4],
        max_decode_cudagraph_capture_tokens=None,
        max_cudagraph_capture_size=4096,
    )
    config = SimpleNamespace(
        compilation_config=compilation,
        speculative_config=SimpleNamespace(),
        scheduler_config=SimpleNamespace(max_num_seqs=512),
        parallel_config=SimpleNamespace(tensor_parallel_size=1, prefill_context_parallel_size=1),
        cache_config=SimpleNamespace(use_kda_recoverssm=False),
        uniform_decode_query_len=query_len,
    )
    monkeypatch.setattr(aclgraph, "is_drafter_moe_model", lambda _: draft_moe)
    monkeypatch.setattr(
        aclgraph, "select_moe_comm_method", lambda size, _, **kwargs: method if size <= limit else MoECommType.ALLTOALL
    )
    manager = AutoRegressiveAclGraphManager(
        config, torch.device("cpu"), mode, decode_query_lens={query_len: 512}, decode_request_sizes=sizes
    )
    assert manager.cudagraph_mode == expected_mode
    assert manager.needs_capture() == (expected_mode != CUDAGraphMode.NONE)
    assert max(manager.decode_request_sizes, default=0) == expected_max_requests
    captured = manager._capture_descs.get(CUDAGraphMode.FULL, [])
    assert max((desc.num_reqs for desc in captured), default=0) == (expected_max_requests if expected_mode else 0)
    if draft_moe and query_len == 1 and method == MoECommType.MC2 and mode == CUDAGraphMode.FULL_DECODE_ONLY:
        manager._graphs_captured = True
        at_limit = manager.dispatch(limit, limit, uniform_token_count=1, num_active_loras=0)
        above_limit = manager.dispatch(limit + 1, limit + 1, uniform_token_count=1, num_active_loras=0)
        assert at_limit.cg_mode == CUDAGraphMode.FULL
        assert above_limit.cg_mode == CUDAGraphMode.NONE
        assert above_limit.num_reqs == limit + 1
        assert above_limit.num_tokens == limit + 1
    assert sizes == original_sizes
    assert compilation.decode_cudagraph_capture_request_sizes == [1, 2, 3, 4]
    assert compilation.prefill_cudagraph_capture_token_sizes == [4, 8, 12, 16]


def test_varlen_draft_decode_prunes_mixed_and_decode_grids(monkeypatch, graph_runtime):
    capacity = 256
    config = VllmConfig()
    config.speculative_config = SimpleNamespace(num_speculative_tokens=3)
    config.scheduler_config.max_num_seqs = 128
    config.scheduler_config.max_num_batched_tokens = 512
    compilation = config.compilation_config
    compilation.prefill_cudagraph_capture_token_sizes = [256, 512]
    compilation.max_decode_cudagraph_capture_requests = 128
    compilation.max_decode_cudagraph_capture_tokens = 512
    monkeypatch.setattr(aclgraph, "is_drafter_moe_model", lambda _: True)
    monkeypatch.setattr(
        aclgraph,
        "select_moe_comm_method",
        lambda size, _, **kwargs: MoECommType.MC2 if size <= capacity else MoECommType.ALLTOALL,
    )
    manager = AutoRegressiveAclGraphManager(
        config,
        torch.device("cpu"),
        CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_lens={4: 128},
        varlen_decode=True,
    )
    assert manager.capture_sizes[-1] == capacity
    assert manager.compilation_config.max_decode_cudagraph_capture_tokens == capacity
    assert compilation.max_decode_cudagraph_capture_tokens == 512
    manager._graphs_captured = True
    for max_query_len, expected in ((4, CUDAGraphMode.FULL), (5, CUDAGraphMode.PIECEWISE)):
        assert manager.dispatch(64, 256, None, 0, max_query_len=max_query_len).cg_mode == expected
        assert manager.dispatch(128, 512, None, 0, max_query_len=max_query_len).cg_mode == CUDAGraphMode.NONE


@pytest.mark.parametrize(("comm_alg", "capacity"), [("fullmesh_v1", 512), ("fullmesh_v2", 256)])
@pytest.mark.parametrize(
    ("method", "manager_cls", "query_len", "mode", "target_tp", "pcp_size"),
    [
        pytest.param("mtp", AutoRegressiveAclGraphManager, 1, CUDAGraphMode.FULL_DECODE_ONLY, 1, 1, id="mtp-decode"),
        pytest.param(
            "draft_model", AutoRegressiveAclGraphManager, 1, CUDAGraphMode.FULL_DECODE_ONLY, 2, 1, id="draft-tp1-decode"
        ),
        pytest.param(
            "eagle", AutoRegressiveAclGraphManager, 1, CUDAGraphMode.FULL_DECODE_ONLY, 1, 1, id="eagle-decode"
        ),
        pytest.param("eagle", AutoRegressiveAclGraphManager, 4, CUDAGraphMode.FULL, 1, 1, id="eagle-prefill"),
        *(
            pytest.param("draft_model", AutoRegressiveAclGraphManager, 4, mode, tp, pcp, id=f"{mode}-tp{tp}-pcp{pcp}")
            for tp in (1, 2)
            for mode, pcp in (
                (CUDAGraphMode.FULL_AND_PIECEWISE, 1),
                (CUDAGraphMode.PIECEWISE, 1),
                (CUDAGraphMode.FULL_AND_PIECEWISE, 2),
            )
        ),
        pytest.param("dspark", DFlashAclGraphManager, 2, CUDAGraphMode.FULL_DECODE_ONLY, 1, 1, id="dspark-decode"),
        pytest.param("dspark", DFlashAclGraphManager, 2, CUDAGraphMode.FULL_AND_PIECEWISE, 1, 1, id="dspark-mixed"),
        pytest.param("dspark", DFlashAclGraphManager, 2, CUDAGraphMode.PIECEWISE, 1, 1, id="dspark-piecewise"),
    ],
)
def test_draft_graphs_exclude_batches_requiring_alltoall(
    monkeypatch, graph_runtime, comm_alg, capacity, method, manager_cls, query_len, mode, target_tp, pcp_size
):
    token_sizes = [128, 256, 512, 1024]
    request_sizes = [size // query_len for size in token_sizes]
    compilation = SimpleNamespace(
        cudagraph_capture_sizes=token_sizes,
        max_cudagraph_capture_size=1024,
        decode_cudagraph_capture_request_sizes=request_sizes,
        prefill_cudagraph_capture_token_sizes=token_sizes,
        max_decode_cudagraph_capture_tokens=None,
    )
    draft_config = SimpleNamespace(
        dtype=torch.bfloat16,
        quantization=None,
        is_moe=True,
        hf_text_config=SimpleNamespace(to_dict=lambda: {"num_experts": 8}),
    )
    # MoE construction publishes this after loading. MTP tests the earlier fallback.
    if method != "mtp":
        draft_config.draft_moe_quant_type = QuantType.NONE
    config = SimpleNamespace(
        model_config=SimpleNamespace(is_moe=True, quantization="ascend"),
        speculative_config=SimpleNamespace(
            method=method, draft_model_config=draft_config, uses_dynamic_speculative_decoding=lambda: False
        ),
        compilation_config=compilation,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=target_tp, enable_expert_parallel=True, prefill_context_parallel_size=pcp_size
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=1024),
        cache_config=SimpleNamespace(use_kda_recoverssm=False),
        uniform_decode_query_len=query_len,
    )
    if method == "draft_model":
        config.speculative_config.draft_parallel_config = SimpleNamespace(
            tensor_parallel_size=1, enable_expert_parallel=True
        )
    monkeypatch.setattr(utils, "_IS_DRAFTER_MOE_MODEL", None)
    monkeypatch.setattr(ascend_forward_context, "_mc2_tokens_capacity", 1024)
    monkeypatch.setattr(ascend_forward_context, "is_mega_moe_supported", lambda: True)
    monkeypatch.setattr(
        ascend_forward_context, "get_current_hardware_profile", lambda: get_hardware_profile(AscendDeviceType.A3)
    )
    monkeypatch.setattr(ascend_forward_context, "get_ep_group", lambda: SimpleNamespace(world_size=8))
    monkeypatch.setattr(
        ascend_forward_context,
        "get_ascend_config",
        lambda: SimpleNamespace(enable_fused_mc2=1, get_mc2_comm_alg=lambda: comm_alg),
    )
    # The target would have admitted each rejected size under its MegaMoE policy.
    rejected_size = 2 * capacity
    assert ascend_forward_context.select_moe_comm_method(rejected_size, config) == MoECommType.FUSED_MC2
    shape: dict[str, Any] = {"decode_query_lens": {query_len: 1024}}
    if manager_cls is AutoRegressiveAclGraphManager and query_len == 1:
        shape["decode_request_sizes"] = request_sizes
    manager = manager_cls(config, torch.device("cpu"), mode, **shape)
    expected_mode = CUDAGraphMode.PIECEWISE if pcp_size > 1 else mode
    assert manager.cudagraph_mode == expected_mode
    assert manager.capture_sizes == [size for size in token_sizes if size <= capacity]
    assert compilation.max_cudagraph_capture_size == 1024
    assert compilation.cudagraph_capture_sizes == [128, 256, 512, 1024]
    assert compilation.prefill_cudagraph_capture_token_sizes == [128, 256, 512, 1024]
    assert compilation.decode_cudagraph_capture_request_sizes == [size // query_len for size in token_sizes]

    manager._graphs_captured = True
    for uniform_query_len, runtime_mode in (
        (query_len, expected_mode.decode_mode()),
        (None, expected_mode.mixed_mode()),
    ):
        admitted = manager.dispatch(capacity // query_len, capacity, uniform_query_len, num_active_loras=0)
        eager = manager.dispatch(rejected_size // query_len, rejected_size, uniform_query_len, num_active_loras=0)
        assert admitted.cg_mode == runtime_mode
        assert eager.cg_mode == CUDAGraphMode.NONE
        assert (eager.num_reqs, eager.num_tokens) == (rejected_size // query_len, rejected_size)


def test_filter_draft_full_aclgraph_sizes_uses_clipped_decode_token_budget():
    """Uniform decode buckets clipped by max_decode_cudagraph_capture_tokens stay admissible."""
    draft_config = SimpleNamespace(parallel_config=SimpleNamespace(prefill_context_parallel_size=1))
    compilation = SimpleNamespace(
        cudagraph_capture_sizes=[256, 512],
        max_cudagraph_capture_size=512,
        prefill_cudagraph_capture_token_sizes=[],
        decode_cudagraph_capture_request_sizes=[128],
        max_decode_cudagraph_capture_tokens=256,
    )
    vllm_config = SimpleNamespace(
        compilation_config=compilation,
        speculative_config=SimpleNamespace(draft_model_config=SimpleNamespace()),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1),
    )

    with (
        patch(
            "vllm_ascend.worker.v2.spec_decode.target_dependent_ar.aclgraph.is_drafter_moe_model",
            return_value=True,
        ),
        patch(
            "vllm_ascend.worker.v2.spec_decode.target_dependent_ar.aclgraph.resolve_speculator_draft_vllm_config",
            return_value=draft_config,
        ),
        patch(
            "vllm_ascend.worker.v2.spec_decode.target_dependent_ar.aclgraph.resolve_draft_moe_quant_type",
            return_value=QuantType.NONE,
        ),
        patch(
            "vllm_ascend.worker.v2.spec_decode.target_dependent_ar.aclgraph.select_moe_comm_method",
            side_effect=lambda num_tokens, *_, **__: MoECommType.MC2 if num_tokens <= 256 else MoECommType.ALLTOALL,
        ),
    ):
        _, mode, req_sizes = filter_draft_full_aclgraph_sizes(
            vllm_config,
            CUDAGraphMode.FULL_DECODE_ONLY,
            {4: 128},
            [128],
        )

    assert mode == CUDAGraphMode.FULL_DECODE_ONLY
    assert req_sizes == [128]


def test_run_fullgraph_uses_updatable_graph():
    """Verify full-graph replay uses the updatable graph path."""
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
    manager._updatable_graph_replay = MagicMock(return_value="result")  # type: ignore[method-assign]
    manager._graph_replay = MagicMock()  # type: ignore[method-assign]
    desc = SimpleNamespace(num_tokens=8, num_reqs=2)
    with patch(
        "vllm_ascend.worker.v2.spec_decode.target_dependent_ar.aclgraph.use_updatable_graph",
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
    """Verify full-graph replay falls back to the legacy graph path."""
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
    manager._updatable_graph_replay = MagicMock()  # type: ignore[method-assign]
    manager._graph_replay = MagicMock(return_value="result")  # type: ignore[method-assign]
    desc = SimpleNamespace(num_tokens=8, num_reqs=2)
    with patch(
        "vllm_ascend.worker.v2.spec_decode.target_dependent_ar.aclgraph.use_updatable_graph",
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


@pytest.mark.parametrize("warmup", [False, True])
def test_capture_draft_prefill_prepares_pcp_inputs_and_runs_forward(warmup):
    """Draft prefill capture stages PCP metadata before invoking its forward."""
    manager = AutoRegressiveAclGraphManager.__new__(AutoRegressiveAclGraphManager)
    manager.speculator = object()
    manager.is_draft_model_prefill = True
    manager.max_num_reqs = 4
    manager.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=1))
    forward_fn = MagicMock()
    model_state, input_buffers, block_tables, kv_cache_config = (object() for _ in range(4))
    attn_groups: Any = object()
    pcp_manager = MagicMock()
    metadata, slots = object(), object()
    desc = SimpleNamespace(num_tokens=8, num_reqs=2, cg_mode=CUDAGraphMode.FULL, max_query_len=4, uniform_token_count=4)

    def capture_side_effect(manager_arg, create_forward_fn, progress_bar_desc=None):
        assert manager_arg is manager
        create_forward_fn(desc, warmup)(CUDAGraphMode.FULL)

    with (
        patch(
            "vllm_ascend.worker.v2.spec_decode.target_dependent_ar.aclgraph.communicator_switch",
            return_value=nullcontext(),
        ),
        patch(
            "vllm_ascend.worker.v2.spec_decode.target_dependent_ar.aclgraph.model_capture_wrapper",
            return_value=nullcontext(),
        ),
        patch(
            "vllm_ascend.worker.v2.spec_decode.target_dependent_ar.aclgraph.prepare_inputs_to_capture",
            return_value=(metadata, slots),
        ) as prepare_inputs,
        patch(
            "vllm_ascend.worker.v2.spec_decode.target_dependent_ar.aclgraph.CudaGraphManager.capture",
            side_effect=capture_side_effect,
        ),
    ):
        manager.capture(
            forward_fn, model_state, input_buffers, block_tables, attn_groups, kv_cache_config, pcp_manager=pcp_manager
        )
    prepare_inputs.assert_called_once_with(
        2,
        8,
        model_state,
        input_buffers,
        block_tables,
        attn_groups,
        kv_cache_config,
        full_cudagraph=True,
        max_query_len=4,
        pcp_manager=pcp_manager,
    )
    forward_fn.assert_called_once_with(2, 8, metadata, slots, None, CUDAGraphMode.FULL)


def test_capture_draft_decode_prepares_inputs_and_runs_forward():
    """Verify draft decode capture prepares inputs and invokes the forward path."""
    manager = AutoRegressiveAclGraphManager.__new__(AutoRegressiveAclGraphManager)
    manager.speculator = SimpleNamespace(num_speculative_steps=3)
    manager.is_draft_model_prefill = False
    manager.max_num_reqs = 3
    # vLLM #53934 turned `dp_size` into a read-only property backed by
    # `vllm_config.parallel_config.data_parallel_size`.
    manager.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=2))
    forward_fn = MagicMock()
    model_state = object()
    input_buffers = SimpleNamespace(seq_lens_cpu=torch.tensor([10, 20, 30], dtype=torch.int32))
    block_tables = object()
    attn_groups: Any = object()
    kv_cache_config = object()
    desc_fields = {"num_tokens": 4, "num_reqs": None, "cg_mode": CUDAGraphMode.FULL}
    # vLLM main (#58275) reads max_query_len/uniform_token_count off the
    # descriptor when preparing capture inputs.
    desc_fields.update(max_query_len=None, uniform_token_count=None)
    desc = SimpleNamespace(**desc_fields)

    def capture_side_effect(manager_arg, create_forward_fn, progress_bar_desc=None):
        """Execute the captured forward function for the test descriptor."""
        runner = create_forward_fn(desc, False)
        runner(CUDAGraphMode.FULL)

    with (
        patch(
            "vllm_ascend.worker.v2.spec_decode.target_dependent_ar.aclgraph.communicator_switch",
            return_value=nullcontext(),
        ),
        patch(
            "vllm_ascend.worker.v2.spec_decode.target_dependent_ar.aclgraph.model_capture_wrapper",
            return_value=nullcontext(),
        ),
        patch(
            "vllm_ascend.worker.v2.spec_decode.target_dependent_ar.aclgraph.prepare_inputs_to_capture",
            return_value=(None, None),
        ) as prepare_inputs,
        patch(
            "vllm_ascend.worker.v2.spec_decode.target_dependent_ar.aclgraph.CudaGraphManager.capture",
            side_effect=capture_side_effect,
        ),
    ):
        manager.capture(forward_fn, model_state, input_buffers, block_tables, attn_groups, kv_cache_config)

    expected_capture_kwargs: dict[str, Any] = {"full_cudagraph": True}
    expected_capture_kwargs["max_query_len"] = None
    expected_capture_kwargs["pcp_manager"] = None
    prepare_inputs.assert_called_once_with(
        3,
        4,
        model_state,
        input_buffers,
        block_tables,
        attn_groups,
        kv_cache_config,
        **expected_capture_kwargs,
    )
    args = forward_fn.call_args.args
    assert args[0] == 3
    assert args[1] is False
    assert args[2].cg_mode == CUDAGraphMode.FULL
    assert args[2].num_tokens == 4
    assert args[2].num_reqs == 3
    assert torch.equal(args[3], torch.tensor([4, 4], dtype=torch.int32))
    assert torch.equal(args[4], torch.tensor([10, 20, 30], dtype=torch.int32))
    # vLLM #60335 passes `num_speculative_steps` on main.
    assert args[5] == 3


def test_graph_replay_updates_full_graph_params():
    """Verify legacy graph replay updates full-graph parameters correctly."""
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
            "vllm_ascend.worker.v2.spec_decode.target_dependent_ar.aclgraph.torch.npu.current_stream",
            return_value=current_stream,
        ),
        patch.object(
            SpeculatorCudaGraphManager,
            "run_fullgraph",
            return_value="result",
        ) as parent_replay,
        patch(
            "vllm_ascend.worker.v2.spec_decode.target_dependent_ar.aclgraph.set_current_vllm_config",
            return_value=nullcontext(),
        ),
        patch(
            "vllm_ascend.worker.v2.spec_decode.target_dependent_ar.aclgraph.set_forward_context",
            return_value=nullcontext(),
        ),
        patch(
            "vllm_ascend.worker.v2.spec_decode.target_dependent_ar.aclgraph.get_forward_context",
            return_value=forward_context,
        ),
        patch(
            "vllm_ascend.worker.v2.spec_decode.target_dependent_ar.aclgraph._EXTRA_CTX",
            extra_ctx,
        ),
        patch(
            "vllm_ascend.worker.v2.spec_decode.target_dependent_ar.aclgraph.update_full_graph_params"
        ) as update_params,
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


@pytest.mark.parametrize("use_dcp,architecture", [(False, "GQA"), (True, "MLA")])
@pytest.mark.parametrize("is_prefill", [False, True])
def test_updatable_graph_replay_uses_param_source(use_dcp, architecture, is_prefill):
    """Resolve the parameter source built for the current draft graph."""
    manager = AutoRegressiveAclGraphManager.__new__(AutoRegressiveAclGraphManager)
    manager.update_stream = MagicMock()
    manager.is_draft_model_prefill = is_prefill
    desc = MagicMock(num_reqs=2)
    graph = MagicMock()
    resolved_tasks = object()
    graph.resolve_tasks.return_value = resolved_tasks
    manager.graphs = {desc: graph}
    steps = [{"draft": object()} for _ in range(1 if is_prefill else 2)]
    source = object()
    manager.speculator = SimpleNamespace(
        use_dcp=use_dcp,
        attn_architecture=architecture,
        build_param_source=MagicMock(return_value=source),
    )
    current_stream = object()

    with (
        patch(
            "vllm_ascend.worker.v2.spec_decode.target_dependent_ar.aclgraph.UpdatableGraph",
            MagicMock,
        ),
        patch(
            "vllm_ascend.worker.v2.spec_decode.target_dependent_ar.aclgraph.torch.npu.current_stream",
            return_value=current_stream,
        ),
        patch.object(
            SpeculatorCudaGraphManager,
            "run_fullgraph",
            return_value="result",
        ) as parent_replay,
    ):
        result = manager._updatable_graph_replay(desc, steps)

    manager.speculator.build_param_source.assert_called_once_with(2, steps[0], is_prefill, num_tokens=desc.num_tokens)
    assert result == "result"
    graph.resolve_tasks.assert_called_once_with(source)
    manager.update_stream.wait_stream.assert_called_once_with(current_stream)
    parent_replay.assert_called_once_with(desc)
    graph.update.assert_called_once_with(manager.update_stream, resolved_tasks)


@pytest.mark.parametrize(
    "use_dcp,architecture,is_prefill",
    [(True, "MLA", True), (True, "MLA", False), (True, "GQA", True), (False, "MLA", True)],
)
def test_target_dependent_param_source_selects_metadata_or_fia(use_dcp, architecture, is_prefill):
    from vllm_ascend.compilation.updatable_graph import ContextSource, SharedSource
    from vllm_ascend.worker.v2.spec_decode.target_dependent_ar.speculator import AscendTargetDependentARSpeculator

    metadata = {"layer": object()}
    params = [{"layer_name": "layer", "actual_seq_kvlen": [3]}]
    speculator = SimpleNamespace(
        use_dcp=use_dcp, attn_architecture=architecture, build_fia_params=MagicMock(return_value=params)
    )
    source = AscendTargetDependentARSpeculator.build_param_source(speculator, 2, metadata, is_prefill, num_tokens=4)

    if use_dcp and architecture == "MLA" and is_prefill:
        assert isinstance(source, ContextSource)
        assert source.context is metadata
        speculator.build_fia_params.assert_not_called()
    else:
        assert isinstance(source, SharedSource)
        assert source.params is params
        speculator.build_fia_params.assert_called_once_with(2, metadata, is_prefill, num_tokens=4)


def test_multi_module_param_source_keeps_fia_parameters():
    from vllm_ascend.compilation.updatable_graph import SharedSource
    from vllm_ascend.worker.v2.spec_decode.multi_module_mtp.speculator import AscendMultiModuleMTPSpeculator

    metadata = {"layer": object()}
    params = [{"layer_name": "layer", "actual_seq_kvlen": [3]}]
    speculator = SimpleNamespace(build_fia_params=MagicMock(return_value=params))
    source = AscendMultiModuleMTPSpeculator.build_param_source(speculator, 2, metadata, False, num_tokens=4)

    assert isinstance(source, SharedSource)
    assert source.params is params
    speculator.build_fia_params.assert_called_once_with(2, metadata, False, num_tokens=4)
