# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU-reachable tests for the Ascend fused mamba "align" block-table gather.

Port of vLLM's ``tests/v1/attention/test_mamba_align_block_table.py`` for
``vllm_ascend.ops.triton.mamba.align_block_table``. ``tests/ut`` runs on CPU
runners (``tests/ut/conftest.py`` pins ``torch.npu.is_available()`` to False
there), and the "align" branch launches the fused NPU Triton kernel, so the
only behavior reachable here is the ``"none"`` passthrough. The on-device
kernel equivalence runs live in
``tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_mamba_align_block_table.py``,
which imports this module's builders the way ``test_gdn_prepare.py`` reuses
``tests/ut/ops/test_gdn_attn_builder.py``.
"""

import torch
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_ascend.ops.triton.mamba.align_block_table import (
    mamba_get_block_table_tensor,
)

WIDTH = 64


def _spec(block_size: int, num_speculative_blocks: int) -> MambaSpec:
    # Only block_size and num_speculative_blocks are read by the align branch;
    # shapes/dtypes just have to be well formed for the frozen dataclass.
    return MambaSpec(
        block_size=block_size,
        shapes=((1,),),
        dtypes=(torch.float32,),
        num_speculative_blocks=num_speculative_blocks,
    )


def _reference(block_table, seq_lens, block_size, num_spec_blocks):
    """The align branch exactly as it was before the fusion."""
    start_indices = (seq_lens - 1) // block_size
    start_indices = start_indices.clamp(min=0)
    offsets = torch.arange(1 + num_spec_blocks, device=block_table.device, dtype=torch.int32)
    indices_to_gather = (start_indices.unsqueeze(1) + offsets).to(torch.int64)
    return torch.gather(block_table, 1, indices_to_gather)


def _make(nreq, block_size, nspec, device, noncontig=False):
    torch.manual_seed(0)
    width = WIDTH * 2 if noncontig else WIDTH
    bt = torch.randint(0, 5000, (nreq, width), dtype=torch.int32, device=device)
    if noncontig:
        # Every other column: stride(1) == 2, which a kernel assuming a unit
        # column stride would read straight through, silently returning the
        # wrong blocks.
        bt = bt[:, ::2]
        assert not bt.is_contiguous()
    # Keep start + nspec inside the row, which the gather already requires.
    hi = max(1, (WIDTH - 1 - nspec) * block_size)
    sl = torch.randint(0, hi, (nreq,), dtype=torch.int32, device=device)
    return bt, sl


def test_cpu_none_mode_passes_through():
    bt = torch.arange(12, dtype=torch.int32).reshape(3, 4)
    sl = torch.tensor([1, 2, 3], dtype=torch.int32)
    assert mamba_get_block_table_tensor(bt, sl, _spec(16, 3), "none") is bt
