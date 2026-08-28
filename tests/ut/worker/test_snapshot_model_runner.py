from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.snapshot.model_restore import (
    _reset_block_table_device_buffers,
    _reset_runtime_tensor_states,
    dump_model_runner,
    restore_model_runner,
)
from vllm_ascend.snapshot.tensor_state import restore_derived_tensor_state


class _TopKHolder(torch.nn.Module):
    def __init__(self, buffer: torch.Tensor) -> None:
        super().__init__()
        self.topk_indices_buffer = buffer

    def reset_snapshot_runtime_state(self) -> None:
        self.topk_indices_buffer.fill_(-1)


class _BackendSpecificReloadTarget:
    def __init__(self) -> None:
        self.reloaded = False

    def restore_snapshot_derived_state(self, act_dtype: torch.dtype) -> None:
        self.reloaded = True

    def get_snapshot_derived_tensors(self) -> dict[str, torch.Tensor]:
        return {"backend_specific_weight": torch.zeros(1)}


class _ImplHolder(torch.nn.Module):
    def __init__(self, impl: object) -> None:
        super().__init__()
        self.impl = impl


class _FailingReloadTarget:
    def restore_snapshot_derived_state(self, act_dtype: torch.dtype) -> None:
        raise RuntimeError("restore failed")


def _make_runner(model, drafter_model):
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(tensor_parallel_size=8),
            model_config=SimpleNamespace(model="/models/test-model"),
        ),
        model_config=SimpleNamespace(dtype=torch.bfloat16, hf_config=object()),
        dp_size=2,
        dp_rank=1,
        device=torch.device("cpu"),
        drafter=SimpleNamespace(model=drafter_model),
        get_model=lambda: model,
    )


def test_dump_model_runner_dumps_target_and_drafter(tmp_path):
    runner = _make_runner(torch.nn.Module(), torch.nn.Module())

    with (
        patch("vllm_ascend.snapshot.model_restore.get_tp_group") as tp_group,
        patch("vllm_ascend.snapshot.model_restore.dump_state_dict") as dump,
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
        patch("vllm_ascend.snapshot.model_restore.get_tp_group") as tp_group,
        patch("vllm_ascend.snapshot.model_restore._restore_one_model") as restore_one,
        patch("vllm_ascend.snapshot.model_restore.restore_global_tensor_state"),
        patch("vllm_ascend.snapshot.model_restore._clear_spec_decode_carryover"),
        patch("vllm_ascend.snapshot.model_restore.restore_drafter_runtime_buffers"),
        patch("vllm_ascend.snapshot.model_restore._reset_attention_builder_runtime_states"),
        patch("vllm_ascend.snapshot.model_restore._reset_runtime_tensor_states"),
        patch("vllm_ascend.snapshot.model_restore._reset_block_table_device_buffers"),
    ):
        tp_group.return_value.rank_in_group = 3
        restore_model_runner(runner, str(tmp_path))

    assert restore_one.call_count == 2
    assert restore_one.call_args_list[0].args[1] is model
    assert restore_one.call_args_list[0].args[3] == "model"
    assert restore_one.call_args_list[1].args[1] is drafter_model
    assert restore_one.call_args_list[1].args[3] == "drafter"


def test_reset_resume_runtime_tensor_states_clears_shared_state():
    runner = SimpleNamespace()
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
    runner.drafter = SimpleNamespace(model=drafter)

    _reset_runtime_tensor_states(runner)

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

    with patch("vllm_ascend.snapshot.tensor_state.logger") as logger:
        restore_derived_tensor_state(_ImplHolder(target), torch.bfloat16, "model")

    assert target.reloaded
    logger.error.assert_called_once()
    assert "backend_specific_weight" in str(logger.error.call_args)


def test_reload_derived_weights_propagates_failure():
    with pytest.raises(RuntimeError, match="restore failed"):
        restore_derived_tensor_state(
            _ImplHolder(_FailingReloadTarget()),
            torch.bfloat16,
            "model",
        )


def test_reset_block_tables_delegates_to_owner():
    runner = SimpleNamespace()
    block_table = SimpleNamespace(clear=Mock(), block_tables=[object(), object()])
    runner.input_batch = SimpleNamespace(block_table=block_table)

    _reset_block_table_device_buffers(runner)

    block_table.clear.assert_called_once_with()
