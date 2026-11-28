# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ops.tilelang_recoverssm import model_verify


def _inputs(rows, tokens):
    mixed = torch.empty((1, tokens, 40, 128), dtype=torch.bfloat16, device="meta")
    gates = torch.empty((tokens, 48), dtype=torch.bfloat16, device="meta")
    checkpoint = torch.empty((73, 28, 128, 128), dtype=torch.bfloat16, device="meta")[:, :24]
    q, k, v = mixed.split((8, 8, 24), dim=2)
    a, b = gates.split(24, dim=1)
    return dict(
        A_log=torch.empty((24,), dtype=torch.float32, device="meta"),
        a=a,
        b=b,
        dt_bias=torch.empty((24,), dtype=torch.bfloat16, device="meta"),
        q=q,
        k=k,
        v=v,
        checkpoint_state=checkpoint,
        replay_cache=torch.empty((73, 24, 4, 257), dtype=torch.float32, device="meta"),
        query_start_loc=torch.empty((rows + 1,), dtype=torch.int32, device="meta"),
        state_indices=torch.empty((rows, 4), dtype=torch.int32, device="meta")[:, 0],
        spec_query_len=4,
        out=torch.empty(v.shape, dtype=v.dtype, device="meta"),
    )


@pytest.fixture
def model_launcher(monkeypatch):
    launch = Mock(side_effect=lambda **kwargs: kwargs["out"])
    monkeypatch.setattr(model_verify, "_is_a3", lambda device: True)
    monkeypatch.setattr(model_verify, "_run_model_verify", launch)
    return launch


@pytest.mark.parametrize("rows", [1, 2, 3, 5, 7, 8, 9, 15, 16, 17, 20, 23, 31, 32, 33, 63, 64, 65, 127, 129])
def test_model_verify_preserves_unpadded_shape(model_launcher, rows):
    inputs = _inputs(rows, 4 * rows - 1)
    z = inputs["v"]
    weight = torch.empty((128,), dtype=torch.bfloat16, device="meta")
    out = model_verify.verify_model(**inputs, z=z, norm_weight=weight, norm_eps=1e-6)
    assert out is inputs["out"]
    model_launcher.assert_called_once()
    args = model_launcher.call_args.kwargs
    assert args["state_indices"] is inputs["state_indices"]
    assert args["query_start_loc"] is inputs["query_start_loc"]
    assert args["q"] is inputs["q"]
    assert args["z"] is z
    assert args["norm_weight"] is weight
    assert args["norm_eps"] == 1e-6


@pytest.mark.parametrize("rows", [0, 1, 33])
def test_empty_verify_skips_kernel(model_launcher, rows):
    inputs = _inputs(rows, 0)
    assert model_verify.verify_model(**inputs) is inputs["out"]
    model_launcher.assert_not_called()


def test_model_verify_without_norm(model_launcher):
    inputs = _inputs(17, 67)
    assert model_verify.verify_model(**inputs) is inputs["out"]
    assert model_launcher.call_args.kwargs["norm_weight"] is None
    assert model_launcher.call_args.kwargs["norm_eps"] is None


def test_unsupported_model_layout_and_norm_raise_before_launch(model_launcher):
    inputs = _inputs(33, 131)
    with pytest.raises(ValueError, match="Verify layout"):
        model_verify.verify_model(**(inputs | {"spec_query_len": 5}))
    with pytest.raises(ValueError, match="Verify layout"):
        model_verify.verify_model(**(inputs | {"use_qk_l2norm_in_kernel": False}))
    with pytest.raises(ValueError, match="Fused sigmoid RMSNorm"):
        model_verify.verify_model(**inputs, z=inputs["v"], norm_eps=1e-6)
    model_launcher.assert_not_called()
