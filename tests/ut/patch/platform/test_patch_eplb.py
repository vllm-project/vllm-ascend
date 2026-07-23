# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from vllm.config import EPLBConfig, ParallelConfig, VllmConfig
from vllm.config import parallel as parallel_module
from vllm.platforms import current_platform

from vllm_ascend.patch.platform import patch_eplb


class _FakeNpuPlatform:
    device_type = "npu"

    def __getattr__(self, name):
        return getattr(current_platform, name)


@contextmanager
def _npu_parallel_config_platform():
    proxy = parallel_module.current_platform
    assert isinstance(proxy, patch_eplb._CudaAlikeEplbPlatformProxy)
    original_platform = proxy._platform
    proxy._platform = _FakeNpuPlatform()
    try:
        yield
    finally:
        proxy._platform = original_platform


def test_parallel_and_vllm_config_keep_upstream_validation():
    with _npu_parallel_config_platform():
        parallel_config = ParallelConfig(
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            enable_eplb=True,
            eplb_config=EPLBConfig(use_async=False),
        )
        vllm_config = VllmConfig(parallel_config=parallel_config)

    assert vllm_config.parallel_config.enable_eplb
    assert vllm_config.parallel_config.eplb_config.communicator == "torch_nccl"


def test_communicator_factory_maps_torch_distributed_to_hccl(monkeypatch):
    communicator = object()
    communicator_cls = MagicMock(return_value=communicator)
    monkeypatch.setattr(patch_eplb, "HcclEplbCommunicator", communicator_cls)
    coordinator = MagicMock()

    with _npu_parallel_config_platform():
        result = patch_eplb._eplb_communicator.create_eplb_communicator(
            coordinator,
            "torch_nccl",
            [],
            [],
        )

    assert result is communicator
    communicator_cls.assert_called_once_with(coordinator.device_group)


def test_parallel_config_platform_patch_is_idempotent():
    proxy = parallel_module.current_platform

    patch_eplb._patch_parallel_config()

    assert parallel_module.current_platform is proxy


def test_router_patch_calls_npu_custom_op(monkeypatch):
    topk_ids = torch.tensor([[0, 1]], dtype=torch.int32)
    physical_ids = torch.tensor([[2, 1]], dtype=torch.int32)
    custom_op = MagicMock(return_value=physical_ids)
    monkeypatch.setattr(
        patch_eplb.torch.ops.vllm,
        "ascend_eplb_map_and_record",
        custom_op,
    )
    monkeypatch.setattr(patch_eplb, "dbo_current_ubatch_id", lambda: 0)
    layer_state = SimpleNamespace(
        logical_to_physical_map=torch.tensor([[0], [1]], dtype=torch.int32),
        logical_replica_count=torch.ones(2, dtype=torch.int32),
        expert_load_view=torch.zeros(3, dtype=torch.int64),
        should_record_tensor=torch.tensor(True),
        num_unpadded_tokens_tensors=[torch.tensor(1, dtype=torch.int32)],
    )
    router = SimpleNamespace(
        eplb_state=layer_state,
        _validate_eplb_state=MagicMock(),
    )

    result = patch_eplb.BaseRouter._apply_eplb_mapping(router, topk_ids)

    assert result is physical_ids
    router._validate_eplb_state.assert_called_once_with()
    assert custom_op.call_args.args[-1] is layer_state.num_unpadded_tokens_tensors[0]


def test_non_matching_scope_discards_pass_without_advancing_load_window(monkeypatch):
    model_state = SimpleNamespace(expert_load_pass=torch.ones(2, dtype=torch.int64))
    eplb_state = patch_eplb._eplb_state.EplbState.__new__(patch_eplb._eplb_state.EplbState)
    eplb_state.model_states = {"model": model_state}
    eplb_state.parallel_config = SimpleNamespace(
        eplb_config=SimpleNamespace(
            log_balancedness_interval=1,
        )
    )
    eplb_state.expert_rearrangement_step = 0
    eplb_state.expert_rearrangement_step_interval = 10
    eplb_state.expert_load_window_step = 0
    eplb_state.expert_load_window_size = 2
    eplb_state.should_record_tensor = None
    eplb_state.is_async = False
    eplb_state._ascend_scope_matched = False
    ep_group = SimpleNamespace(device_group=MagicMock())
    monkeypatch.setattr(patch_eplb._eplb_state, "get_ep_group", lambda: ep_group)

    eplb_state.step()

    torch.testing.assert_close(model_state.expert_load_pass, torch.zeros(2, dtype=torch.int64))
    assert eplb_state.expert_load_window_step == 0
    assert eplb_state.expert_rearrangement_step == 1
