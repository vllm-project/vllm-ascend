# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integer metadata movement must be exact, including strided/padded inputs."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops.triton.mamba import stage_gdn_commit as staging


@pytest.mark.parametrize("batch,capacity,width", [(1, 8, 1), (3, 8, 7), (125, 192, 141), (192, 192, 1)])
@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("align", [False, True])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("copy_accepted", [False, True])
def test_stage_kernel(batch, capacity, width, mapped, align, dtype, copy_accepted):
    check_stage_case(batch, capacity, width, mapped, align, dtype, device="npu", copy_accepted=copy_accepted)


def check_stage_case(batch, capacity, width, mapped, align, dtype, *, device, copy_accepted=True):
    bucket = min(1 << (batch - 1).bit_length(), capacity)
    rows = min(batch + 2, capacity)

    def strided(values):
        out = torch.full((2 * len(values),), -111, dtype=dtype, device=device)
        out[::2] = torch.tensor(values, dtype=dtype, device=device)
        return out[::2]

    accepted = strided([i % 5 for i in range(rows)])
    source = SimpleNamespace(
        indices=strided([i + 1 for i in range(batch)]),
        offsets=strided([4 * i for i in range(batch + 1)]),
        mapping=strided([(i + 1) % rows for i in range(batch)]) if mapped else None,
        # More metadata rows than real sampler rows emulate graph padding.
        table=torch.arange((capacity + 4) * width * 2, dtype=dtype, device=device).reshape(capacity + 4, width * 2)[
            :, ::2
        ]
        if align
        else None,
        computed=strided([1000 + i for i in range(capacity + 4)]) if align else None,
    )
    target = SimpleNamespace(
        indices=torch.full((bucket,), -1, dtype=dtype, device=device),
        offsets=torch.full((bucket + 1,), -1, dtype=dtype, device=device),
        mapping=torch.full((bucket,), -1, dtype=dtype, device=device),
        table=torch.full((capacity, width), -1, dtype=dtype, device=device) if align else None,
        computed=torch.full((capacity,), -1, dtype=dtype, device=device) if align else None,
    )
    output = torch.full((capacity,), -1, dtype=dtype, device=device)
    staging.stage_gdn_commit(source, target, accepted, output, copy_accepted=copy_accepted)
    if copy_accepted:
        torch.testing.assert_close(output[:rows], accepted)
        assert not output[rows:].count_nonzero()
    else:
        assert (output == -1).all()
    torch.testing.assert_close(target.indices[:batch], source.indices)
    assert not target.indices[batch:].count_nonzero()
    torch.testing.assert_close(target.offsets[: batch + 1], source.offsets)
    assert (target.offsets[batch + 1 :] == source.offsets[-1]).all()
    expected_map = source.mapping if mapped else torch.arange(batch, dtype=dtype, device=device)
    torch.testing.assert_close(target.mapping[:batch], expected_map)
    assert not target.mapping[batch:].count_nonzero()
    if align:
        torch.testing.assert_close(target.table[:rows], source.table[:rows])
        torch.testing.assert_close(target.computed[:rows], source.computed[:rows])
        assert not target.table[rows:].count_nonzero()
        assert not target.computed[rows:].count_nonzero()


def test_batch_counts_do_not_create_jit_specializations():
    assert {"batch", "sampled_rows"}.issubset(set(staging._stage_gdn_commit_kernel.do_not_specialize))
