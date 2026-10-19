# SPDX-License-Identifier: Apache-2.0
"""Reject incompatible cache/metadata contracts before launching a kernel."""

import pytest
import torch

from vllm_ascend.ops.triton.mamba.gdn_recover import (
    _check_cache,
    _check_conv,
    _check_metadata,
    gdn_recover_verify,
)


def _cache():
    state = torch.empty(5, 6, 17, 33)
    records = torch.empty(5, 6, 4, 51)
    return state, records


@pytest.mark.parametrize(
    "case,error",
    [
        ("bf16-state", "FP32"),
        ("bf16-record", "FP32"),
        ("wrong-record", "records must hold"),
        ("different-device", "same device"),
        ("inner-stride", "contiguous inner"),
        ("overlap", "must not overlap"),
        ("zero-heads", "positive"),
        ("window-too-wide", "supported window"),
        ("dimension-too-wide", "supported window"),
    ],
)
def test_invalid_checkpoint_layout(case, error):
    state, records = _cache()
    if case == "bf16-state":
        state = state.bfloat16()
    elif case == "bf16-record":
        records = records.bfloat16()
    elif case == "wrong-record":
        records = records[..., :-1]
    elif case == "different-device":
        records = records.to("meta")
    elif case == "inner-stride":
        state = state.transpose(-1, -2)
    elif case == "overlap":
        state = state.as_strided(state.shape, (0, 17 * 33, 33, 1))
    elif case == "zero-heads":
        state = state[:, :0]
        records = records[:, :0]
    elif case == "window-too-wide":
        records = torch.empty(5, 6, 17, 51)
    elif case == "dimension-too-wide":
        state = torch.empty(5, 6, 17, 257)
        records = torch.empty(5, 6, 4, 275)
    with pytest.raises(ValueError, match=error):
        _check_cache(state, records)


@pytest.mark.parametrize("case", ["shape", "float", "stride", "device"])
def test_invalid_metadata(case):
    state, _ = _cache()
    ids = torch.tensor([2, 1], dtype=torch.int32)
    offsets = torch.tensor([0, 2, 6], dtype=torch.int64)
    if case == "shape":
        offsets = offsets[:-1]
    elif case == "float":
        ids = ids.float()
    elif case == "stride":
        ids = torch.tensor([2, 0, 1, 0], dtype=torch.int32)[::2]
    else:
        ids = ids.to("meta")
    with pytest.raises(ValueError):
        _check_metadata(state, ids, offsets)


@pytest.mark.parametrize("case", ["shape", "dtype", "history", "window", "stride"])
def test_invalid_conv_layout(case):
    conv = torch.empty(5, 6, 257, dtype=torch.bfloat16)
    saved = torch.empty(5, 3, 257, dtype=torch.bfloat16)
    if case == "shape":
        saved = saved[:, :, :-1]
    elif case == "dtype":
        saved = saved.float()
    elif case == "history":
        saved = torch.empty(5, 17, 257, dtype=torch.bfloat16)
    elif case == "window":
        conv = torch.empty(5, 32, 257, dtype=torch.bfloat16)
    else:
        conv = conv.as_strided(conv.shape, (0, 257, 1))
    with pytest.raises(ValueError):
        _check_conv(conv, saved)


@pytest.mark.parametrize("case", ["heads", "q-shape", "g-shape", "output", "scale"])
def test_invalid_verify_arguments(case):
    state, records = _cache()
    q = torch.empty(6, 2, 33)
    k = torch.empty_like(q)
    v = torch.empty(6, 6, 17)
    g = torch.empty(6, 6)
    beta = torch.empty_like(g)
    ids = torch.tensor([2, 1], dtype=torch.int32)
    offsets = torch.tensor([0, 2, 6], dtype=torch.int32)
    kwargs = {}
    if case == "heads":
        q = torch.empty(6, 4, 33)
        k = torch.empty_like(q)
    elif case == "q-shape":
        q = q[..., :-1]
    elif case == "g-shape":
        g = g[:, :-1]
    elif case == "output":
        kwargs["out"] = torch.empty(6, 6, 34)[..., ::2]
    else:
        kwargs["scale"] = float("nan")
    with pytest.raises(ValueError):
        gdn_recover_verify(q, k, v, g, beta, state, records, ids, offsets, **kwargs)
