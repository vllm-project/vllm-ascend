# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from copy import copy
from types import SimpleNamespace

import pytest
import torch
from vllm.config import CUDAGraphMode

from vllm_ascend.models.hy_v4_flash import dfly_speculator


def make_speculator():
    spec = object.__new__(dfly_speculator.FlashDFlySpeculator)
    parallel = SimpleNamespace(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        data_parallel_size=4,
        is_moe_model=True,
    )
    spec.vllm_config = SimpleNamespace(
        additional_config={"hy4flash_dfly_dp_graph": True},
        parallel_config=parallel,
        compilation_config=SimpleNamespace(decode_cudagraph_capture_request_sizes=[1, 3]),
    )
    spec.speculative_config = SimpleNamespace(enforce_eager=False, draft_parallel_config=copy(parallel))
    spec.dp_size = 4
    spec.max_num_reqs = 3
    spec.max_num_tokens = 18
    spec.num_query_per_req = 6
    spec.num_speculative_steps = 5
    spec.hidden_size = 4
    spec.device = torch.device("cpu")
    spec.dtype = torch.bfloat16
    spec.cudagraph_manager = None
    spec.model_state = SimpleNamespace(rope_state=None)
    spec.target = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=SimpleNamespace(tp_size=1, token_exchange_group=None)),
        lm_head=SimpleNamespace(tp_size=1),
    )
    spec.model = torch.nn.Identity()
    return spec


def test_local_tp1_draft_can_capture_without_changing_target(monkeypatch):
    spec = make_speculator()
    monkeypatch.setattr(dfly_speculator, "CudaGraphManager", lambda *a, **kw: SimpleNamespace())
    monkeypatch.setattr(torch.npu, "graph_pool_handle", lambda: "private-pool")
    spec.init_cudagraph_manager(CUDAGraphMode.FULL_DECODE_ONLY)
    assert spec.cudagraph_manager is not None
    assert spec.cudagraph_manager.pool == "private-pool"
    assert spec.graph_vllm_config.parallel_config.is_moe_model is False
    assert spec.vllm_config.parallel_config.is_moe_model is True
    assert spec.graph_vllm_config.parallel_config is not spec.vllm_config.parallel_config
    assert spec.graph_ids.shape == (3, 6)


@pytest.mark.parametrize(
    "field",
    [
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
    ],
)
@pytest.mark.parametrize("draft", [False, True])
def test_parallel_draft_stays_eager(field, draft):
    spec = make_speculator()
    parallel = spec.speculative_config.draft_parallel_config if draft else spec.vllm_config.parallel_config
    setattr(parallel, field, 2)
    assert not spec._dp_graph_is_local()
    spec.init_cudagraph_manager(CUDAGraphMode.FULL_DECODE_ONLY)
    assert spec.cudagraph_manager is None


@pytest.mark.parametrize("case", ["head", "embedding", "exchange", "reduce", "draft_linear"])
def test_dp_axis_collectives_stay_eager(case):
    spec = make_speculator()
    if case == "head":
        spec.target.lm_head.tp_size = 4
    elif case == "embedding":
        spec.target.model.embed_tokens.tp_size = 4
    elif case == "exchange":
        spec.target.model.embed_tokens.token_exchange_group = SimpleNamespace(world_size=4)
    elif case == "reduce":
        spec.target.model.embed_tokens.output_reduce_group = SimpleNamespace(world_size=4)
    else:
        spec.model.tp_size = 2
    assert not spec._dp_graph_is_local()
    spec.init_cudagraph_manager(CUDAGraphMode.FULL_DECODE_ONLY)
    assert spec.cudagraph_manager is None


@pytest.mark.parametrize("case", ["disabled", "eager", "no_graphs", "no_sizes"])
def test_graph_opt_in_and_runtime_modes(case):
    spec = make_speculator()
    mode = CUDAGraphMode.FULL_DECODE_ONLY
    if case == "disabled":
        spec.vllm_config.additional_config.clear()
    elif case == "eager":
        spec.speculative_config.enforce_eager = True
    elif case == "no_graphs":
        mode = CUDAGraphMode.NONE
    else:
        spec.vllm_config.compilation_config.decode_cudagraph_capture_request_sizes = []
    spec.init_cudagraph_manager(mode)
    assert spec.cudagraph_manager is None


def test_graph_option_rejects_string_boolean():
    spec = make_speculator()
    spec.vllm_config.additional_config["hy4flash_dfly_dp_graph"] = "false"
    with pytest.raises(ValueError, match="must be a boolean"):
        spec.init_cudagraph_manager(CUDAGraphMode.FULL_DECODE_ONLY)


def test_single_dp_replica_retains_existing_tp_graph_support(monkeypatch):
    spec = make_speculator()
    spec.dp_size = spec.vllm_config.parallel_config.data_parallel_size = 1
    spec.vllm_config.parallel_config.tensor_parallel_size = 4
    spec.vllm_config.additional_config.clear()
    monkeypatch.setattr(dfly_speculator, "CudaGraphManager", lambda *a, **kw: SimpleNamespace())
    monkeypatch.setattr(torch.npu, "graph_pool_handle", lambda: "private-pool")
    spec.init_cudagraph_manager(CUDAGraphMode.FULL_DECODE_ONLY)
    assert spec.cudagraph_manager is not None


def test_capture_uses_npu_communicator_and_dense_context(monkeypatch):
    spec = make_speculator()
    spec.vllm_config.model_config = SimpleNamespace(max_model_len=256)
    spec.kv_layers = (SimpleNamespace(kv_cache=torch.empty(1, 1, 16, 576)),)
    spec.offsets = torch.arange(6)
    spec.input_buffers = SimpleNamespace(positions=torch.empty(18, dtype=torch.long))
    spec.draft_tokens = torch.empty(3, 5, dtype=torch.long)
    spec.model.config = SimpleNamespace(mask_token_id=2)
    state = {"switched": False, "forward": False}

    @contextmanager
    def switch():
        state["switched"] = True
        try:
            yield
        finally:
            state["switched"] = False

    @contextmanager
    def context(metadata, config, **kwargs):
        assert state["switched"]
        assert config.parallel_config.is_moe_model is False
        assert kwargs["num_tokens"] == 6
        yield

    def capture(prepare, **kwargs):
        assert state["switched"]
        forward = prepare(SimpleNamespace(num_reqs=1, num_tokens=6), False)
        forward(CUDAGraphMode.NONE)

    def forward(*args):
        state["forward"] = True

    monkeypatch.setattr(dfly_speculator, "communicator_switch", switch)
    monkeypatch.setattr(dfly_speculator, "set_forward_context", context)
    monkeypatch.setattr(dfly_speculator, "CudaGraphManager", lambda *a, **kw: SimpleNamespace(capture=capture))
    monkeypatch.setattr(torch.npu, "graph_pool_handle", lambda: "private-pool")
    spec._forward_queries = forward
    spec.init_cudagraph_manager(CUDAGraphMode.FULL_DECODE_ONLY)
    spec.capture()
    assert state == {"switched": False, "forward": True}
