from types import SimpleNamespace

import torch

from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class _TopKHolder(torch.nn.Module):
    def __init__(self, buffer: torch.Tensor) -> None:
        super().__init__()
        self.topk_indices_buffer = buffer


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
