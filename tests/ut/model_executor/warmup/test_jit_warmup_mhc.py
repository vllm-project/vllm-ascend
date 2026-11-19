# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import inspect
from unittest.mock import MagicMock

import pytest
import torch

from tests.ut.model_executor.warmup.helpers import make_registry
from vllm_ascend.ops.triton import mhc_post_collapse_norm as mhc_ops


def test_mhc_warmup_registration():
    owner = mhc_ops.MhcPostCollapseNormKernel()
    registry = make_registry()

    owner.register_warmup(torch.bfloat16, False)
    assert len(registry) == 0

    with registry.activate():
        owner.register_warmup(torch.bfloat16, False)
        owner.register_warmup(torch.bfloat16, True)
        owner.register_warmup(torch.bfloat16, False)

    registrations = registry._registrations[owner]
    assert len(registrations) == 2
    assert {key for args, kwargs in registrations for key in owner.get_warmup_keys(*args, **kwargs)} == {
        (torch.bfloat16, False),
        (torch.bfloat16, True),
    }


@pytest.mark.parametrize("write_fp32", [False, True], ids=["attention", "moe"])
@pytest.mark.parametrize("rows", [1, 72])
def test_mhc_warmup_matches_runtime_launch(monkeypatch, write_fp32, rows):
    owner = mhc_ops.MhcPostCollapseNormKernel()
    definition = type(owner).kernel
    kernel = MagicMock()
    kernel.arg_names = tuple(inspect.signature(getattr(definition, "fn", definition)).parameters)
    owner.kernel = kernel
    monkeypatch.setattr(mhc_ops, "MHC_POST_COLLAPSE_NORM", owner)

    owner.compile((torch.bfloat16, write_fp32))
    kernel.warmup.assert_called_once()
    kernel.__getitem__.assert_not_called()

    width = mhc_ops.MHC_FUSION_HIDDEN_SIZE
    sublayer_out = torch.empty((width, rows), dtype=torch.bfloat16).t()
    residual = torch.empty((rows, 4, width), dtype=torch.bfloat16)
    post = torch.empty((rows, 4), dtype=torch.float32)
    comb = torch.empty((rows, 4, 4), dtype=torch.float32)
    pre_mix = torch.empty_like(post)
    weight = torch.empty((width,), dtype=torch.bfloat16)
    updated, normalized, routing_input = mhc_ops.mhc_post_collapse_norm(
        sublayer_out, residual, post, comb, pre_mix, weight, 1e-6, write_fp32=write_fp32
    )

    kernel.__getitem__.assert_called_once_with((rows,))
    launch = kernel.__getitem__.return_value
    launch.assert_called_once()
    actual = launch.call_args.kwargs
    warmed = kernel.warmup.call_args.kwargs
    assert warmed["grid"] == (1,)
    assert actual.keys() == warmed.keys() - {"grid"}
    assert actual.keys() == set(kernel.arg_names) | {"enable_fp_fusion"}
    for name, value in actual.items():
        if name.endswith("_ptr"):
            assert value.dtype == warmed[name].dtype
        else:
            assert value == warmed[name]

    assert actual["sublayer_out_ptr"].is_contiguous()
    assert (actual["sublayer_out_ptr"] is sublayer_out) == sublayer_out.is_contiguous()
    assert actual["residual_ptr"] is residual
    assert actual["updated_ptr"] is updated and updated is not residual
    assert actual["normalized_ptr"] is normalized and normalized is not sublayer_out
    assert actual["normalized_fp32_ptr"] is routing_input
    assert (routing_input is normalized) == (not write_fp32)
    assert routing_input.dtype == (torch.float32 if write_fp32 else torch.bfloat16)
