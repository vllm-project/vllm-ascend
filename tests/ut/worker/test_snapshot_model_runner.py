from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class _TopKHolder(torch.nn.Module):
    def __init__(self, buffer: torch.Tensor) -> None:
        super().__init__()
        self.topk_indices_buffer = buffer


class _BackendSpecificReloadTarget:
    def __init__(self) -> None:
        self.reloaded = False

    def reload_derived_weights_after_restore(self, act_dtype: torch.dtype) -> None:
        self.reloaded = True

    def get_derived_weight_sanity_tensors(self) -> dict[str, torch.Tensor]:
        return {"backend_specific_weight": torch.zeros(1)}


class _ImplHolder(torch.nn.Module):
    def __init__(self, impl: object) -> None:
        super().__init__()
        self.impl = impl


def test_reset_resume_sfa_runtime_buffers_clears_shared_state():
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.group_len = SimpleNamespace(
        gpu=torch.full((4,), 3, dtype=torch.int32),
        cpu=torch.full((4,), 5, dtype=torch.int32),
    )
    runner.group_key_idx = SimpleNamespace(
        gpu=torch.full((4,), 7, dtype=torch.int32),
        cpu=torch.full((4,), 11, dtype=torch.int32),
    )
    runner.group_key_cache_idx = SimpleNamespace(
        gpu=torch.full((4,), 13, dtype=torch.int32),
        cpu=torch.full((4,), 17, dtype=torch.int32),
    )

    shared_topk = torch.full((4, 8), 23, dtype=torch.int32)
    model = _TopKHolder(shared_topk)
    model.child = _TopKHolder(shared_topk)
    drafter = _TopKHolder(shared_topk)
    runner.get_model = lambda: model
    runner._get_drafter_model = lambda: drafter

    runner._reset_resume_sfa_runtime_buffers()

    for staged in (
        runner.group_len,
        runner.group_key_idx,
        runner.group_key_cache_idx,
    ):
        assert torch.count_nonzero(staged.gpu) == 0
        assert torch.count_nonzero(staged.cpu) == 0
    assert torch.all(shared_topk == -1)


def test_reload_derived_weights_uses_backend_specific_sanity_tensors():
    target = _BackendSpecificReloadTarget()
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.model_config = SimpleNamespace(dtype=torch.bfloat16)

    with patch("vllm_ascend.worker.model_runner_v1.logger") as logger:
        runner._reload_non_persistent_derived_weights(
            model=_ImplHolder(target),
            label="model",
        )

    assert target.reloaded
    logger.error.assert_called_once()
    assert "backend_specific_weight" in str(logger.error.call_args)
