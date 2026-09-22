# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import torch

from vllm_ascend.models.glm5next.ops.causal_conv1d import _staging_geometry


def test_staging_geometry_preserves_every_speculative_state_column():
    cache_indices = torch.empty_strided((3, 4), (7, 1), dtype=torch.int32)

    assert _staging_geometry(cache_indices) == (3, 4, 12, 1)


def test_staging_geometry_keeps_one_dimensional_prefill_indices():
    cache_indices = torch.empty(3, dtype=torch.int32)

    assert _staging_geometry(cache_indices) == (3, 1, 3, 0)


def test_causal_conv_schema_declares_mutated_tensors():
    source = (Path(__file__).parents[3] / "csrc" / "torch_binding.cpp").read_text(encoding="utf-8")

    assert '"npu_causal_conv1d_custom(Tensor(a!) output, Tensor x, "' in source
    assert '"                         Tensor(b!) conv_state, "' in source
    assert '") -> Tensor(a!)");' in source
