# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Contract tests for vllm_ascend.ops.rope_dsv4 components.

``ComplexExpRotaryEmbedding.forward`` writes its rotated result back into
the input storage (``y.copy_``). ``DeepseekV4DSparkModel._project_shared_kv``
relies on this in-place behavior (it discards the rope return value), so a
contract test locks it: a future out-of-place refactor fails loudly here
instead of silently disabling draft RoPE.
"""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend.ops.rope_dsv4 import ComplexExpRotaryEmbedding


def test_dsv4_rope_writes_back_inplace():
    # DeepseekV4DSparkModel._project_shared_kv discards the return value of
    # _apply_dsv4_rope; RoPE only takes effect because forward writes the
    # rotated values back into the input storage (``y.copy_``). Lock that
    # contract so a future out-of-place refactor fails loudly instead of
    # silently disabling draft RoPE.
    with (
        patch("vllm_ascend.ops.rope_dsv4.current_platform") as fake_platform,
        # create=True: the torch_npu mock installed by tests/ut/conftest.py on
        # CPU-only CI has no npu_rotary_mul attribute.
        patch(
            "torch_npu.npu_rotary_mul",
            lambda x, cos, sin, rotary_mode=None: x * 2 + 1,
            create=True,
        ),
    ):
        fake_platform.device_type = "cpu"
        vllm_config = SimpleNamespace(
            speculative_config=None,
            scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        )
        rotary = ComplexExpRotaryEmbedding(
            vllm_config=vllm_config,
            layername="ut.dspark_rope",
            head_size=8,
            rotary_dim=8,
            max_position_embeddings=64,
            base=10000,
            scaling_factor=1.0,
        )
        x = torch.randn(4, 1, 8)
        snapshot = x.clone()
        out = rotary(x, torch.ones(1), torch.zeros(1))

    assert not torch.equal(x, snapshot)
    assert out is x
