from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.snapshot.model_runner_lifecycle.checkpoint import dump_model_runner
from vllm_ascend.snapshot.model_runner_lifecycle.module_lifecycle import (
    reset_modules_runtime_state,
    restore_model_non_checkpoint_state,
)
from vllm_ascend.snapshot.model_runner_lifecycle.restore import (
    _reset_target_and_drafter_modules_after_restore,
    _restore_model_runner_runtime_state,
    restore_model_runner,
)


class _TopKHolder(torch.nn.Module):
    def __init__(self, buffer: torch.Tensor) -> None:
        super().__init__()
        self.topk_indices_buffer = buffer

    def reset_runtime_state_after_snapshot_restore(self) -> None:
        self.topk_indices_buffer.fill_(-1)


class _BackendSpecificReloadTarget:
    def __init__(self) -> None:
        self.reloaded = False
        self.runtime_reset = False

    def restore_lazy_tensors_after_snapshot_restore(self, act_dtype: torch.dtype, *, restored_tensor_ids) -> None:
        self.reloaded = True

    def reset_runtime_state_after_snapshot_restore(self) -> None:
        self.runtime_reset = True


class _ImplHolder(torch.nn.Module):
    def __init__(self, impl: object) -> None:
        super().__init__()
        self.impl = impl


class _FailingReloadTarget:
    def restore_lazy_tensors_after_snapshot_restore(self, act_dtype: torch.dtype, *, restored_tensor_ids) -> None:
        raise RuntimeError("restore failed")


def _make_runner(model, drafter_model):
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            use_v2_model_runner=True,
            parallel_config=SimpleNamespace(tensor_parallel_size=8, pipeline_parallel_size=1),
            model_config=SimpleNamespace(model="/models/test-model"),
        ),
        model_config=SimpleNamespace(dtype=torch.bfloat16, hf_config=object()),
        dp_size=2,
        dp_rank=1,
        device=torch.device("cpu"),
        kv_block_zeroer=None,
        get_model=lambda: model,
        get_draft_model=lambda: drafter_model,
    )


def test_dump_model_runner_dumps_target_and_drafter(tmp_path):
    runner = _make_runner(torch.nn.Module(), torch.nn.Module())

    with (
        patch("vllm_ascend.snapshot.model_runner_lifecycle.checkpoint.get_tp_group") as tp_group,
        patch("vllm_ascend.snapshot.model_runner_lifecycle.checkpoint.dump_state_dict") as dump,
    ):
        tp_group.return_value.rank_in_group = 3
        dump_model_runner(runner, str(tmp_path))

    assert dump.call_count == 2
    assert str(dump.call_args_list[0].args[1]).endswith("model_ckpt.1tp3.pth")
    assert str(dump.call_args_list[1].args[1]).endswith("model_ckpt_drafter.1tp3.pth")


def test_restore_model_runner_restores_target_and_drafter(tmp_path):
    model = torch.nn.Module()
    drafter_model = torch.nn.Module()
    runner = _make_runner(model, drafter_model)

    with (
        patch("vllm_ascend.snapshot.model_runner_lifecycle.checkpoint.get_tp_group") as tp_group,
        patch("vllm_ascend.snapshot.model_runner_lifecycle.restore._restore_model_checkpoint") as restore_one,
        patch(
            "vllm_ascend.snapshot.model_runner_lifecycle.restore._restore_model_runner_runtime_state"
        ) as restore_runtime,
    ):
        tp_group.return_value.rank_in_group = 3
        restore_model_runner(runner, str(tmp_path))

    assert restore_one.call_count == 2
    assert restore_one.call_args_list[0].args[1] is model
    assert restore_one.call_args_list[0].args[3] == "model"
    assert restore_one.call_args_list[1].args[1] is drafter_model
    assert restore_one.call_args_list[1].args[3] == "drafter"
    restore_runtime.assert_called_once_with(runner)


def test_restore_model_runner_runtime_state_runs_all_phases():
    runner = _make_runner(torch.nn.Module(), torch.nn.Module())
    model = runner.get_model()

    with (
        patch("vllm_ascend.snapshot.model_runner_lifecycle.restore.reload_cos_and_sin_after_restore") as reload_rope,
        patch(
            "vllm_ascend.snapshot.model_runner_lifecycle.restore.reset_moe_padding_cache_after_snapshot_restore"
        ) as reset_padding,
        patch("vllm_ascend.snapshot.model_runner_lifecycle.restore._reset_v2_runner_runtime_state") as reset_runner,
        patch(
            "vllm_ascend.snapshot.model_runner_lifecycle.restore._reset_target_and_drafter_modules_after_restore"
        ) as reset_modules,
    ):
        _restore_model_runner_runtime_state(runner)

    reload_rope.assert_called_once_with(model)
    reset_padding.assert_called_once_with()
    reset_runner.assert_called_once_with(runner)
    reset_modules.assert_called_once_with(runner)


def test_reset_target_and_drafter_modules_after_restore():
    shared_topk = torch.full((4, 8), 23, dtype=torch.int32)
    model = _TopKHolder(shared_topk)
    model.child = _TopKHolder(shared_topk)
    backend = _BackendSpecificReloadTarget()
    model.backend = _ImplHolder(backend)
    drafter = _TopKHolder(shared_topk)
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(use_v2_model_runner=True),
        get_model=lambda: model,
        get_draft_model=lambda: drafter,
    )

    _reset_target_and_drafter_modules_after_restore(runner)

    assert torch.all(shared_topk == -1)
    assert backend.runtime_reset


def test_reload_derived_weights_uses_backend_specific_hook():
    target = _BackendSpecificReloadTarget()

    restore_model_non_checkpoint_state(_ImplHolder(target), torch.bfloat16, "model")

    assert target.reloaded


def test_module_lifecycle_dispatches_module_before_impl():
    calls = []
    impl = MagicMock()
    impl.reset_runtime_state_after_snapshot_restore.side_effect = lambda: calls.append("impl")
    module = _ImplHolder(impl)
    module.reset_runtime_state_after_snapshot_restore = lambda: calls.append("module")

    reset_modules_runtime_state((module,))

    assert calls == ["module", "impl"]


def test_reload_derived_weights_propagates_failure():
    with pytest.raises(RuntimeError, match="restore failed"):
        restore_model_non_checkpoint_state(
            _ImplHolder(_FailingReloadTarget()),
            torch.bfloat16,
            "model",
        )
