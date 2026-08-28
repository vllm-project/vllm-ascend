# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_ascend.snapshot.model_state import restore_state_dict


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2))
        self.register_buffer("scale", torch.zeros(2))


def test_restore_state_dict_copies_parameters_and_buffers(tmp_path) -> None:
    path = tmp_path / "model.pth"
    torch.save(
        {
            "weight": torch.tensor([1.0, 2.0]),
            "scale": torch.tensor([3.0, 4.0]),
        },
        path,
    )
    model = _Model()

    restore_state_dict(model, str(path), "model")

    torch.testing.assert_close(model.weight, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(model.scale, torch.tensor([3.0, 4.0]))
