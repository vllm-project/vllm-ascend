# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm_ascend.worker.v2.spec_decode import greedy_verify_experiment


def _make_verify_args():
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    draft_sampled = torch.tensor([0, 1])
    verify_draft_sampled = torch.tensor([0, 1])
    return (
        logits,
        None,
        draft_sampled,
        torch.tensor([0, 1]),
        torch.tensor([0, 2], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        np.array([0], dtype=np.int32),
        torch.tensor([0, 0], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        np.array([2], dtype=np.int32),
        verify_draft_sampled,
    )


def _make_rejection_sampler_cls():
    class FakeRejectionSampler:
        def __init__(self, sampler=None, spec_config=None, device=None):
            pass

        def _verify(self, *args):
            self.original_verify_args = args
            return "original"

    return FakeRejectionSampler


@pytest.fixture(autouse=True)
def _disable_entropy_verify(monkeypatch):
    config = SimpleNamespace(rejection_sampler_config=SimpleNamespace(enable_entropy_verify=False))
    monkeypatch.setattr(greedy_verify_experiment, "get_ascend_config", lambda: config)


def test_install_uses_greedy_fast_path(monkeypatch):
    rejection_sampler_cls = _make_rejection_sampler_cls()
    greedy_verify_experiment.install(rejection_sampler_cls)

    processed_logits = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    sampled = torch.tensor([[1, 0]])
    counts = torch.tensor([2], dtype=torch.int32)
    greedy_verify = MagicMock(return_value=(sampled, counts))
    monkeypatch.setattr(greedy_verify_experiment, "greedy_verify", greedy_verify)

    sampler = MagicMock()
    sampler.apply_sampling_params.return_value = processed_logits
    sampler.sampling_states.temperature.np = np.array([0.0], dtype=np.float32)
    rejection_sampler = rejection_sampler_cls(spec_config=SimpleNamespace(method="mtp"))
    rejection_sampler.sampler = sampler
    rejection_sampler.watermark_key = None
    rejection_sampler.synthetic_conditional_rates = None
    rejection_sampler.use_block_verification = False
    rejection_sampler.fly_window_size = 0
    rejection_sampler.num_speculative_steps = 1
    args = _make_verify_args()

    result = rejection_sampler._verify(*args)

    assert result[0] is processed_logits
    assert result[1] is sampled
    assert result[2] is counts
    sampler.apply_sampling_params.assert_called_once_with(
        args[0],
        args[7],
        args[5],
        args[6],
        args[3],
        args[2],
        args[8],
        args[9],
    )
    greedy_verify.assert_called_once_with(
        processed_logits,
        args[10],
        args[4],
        1,
    )


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("watermark_key", 1),
        ("synthetic_conditional_rates", torch.tensor([1.0])),
        ("use_block_verification", True),
        ("fly_window_size", 1),
    ],
)
def test_install_falls_back_for_unsupported_modes(attribute, value):
    rejection_sampler_cls = _make_rejection_sampler_cls()
    greedy_verify_experiment.install(rejection_sampler_cls)

    rejection_sampler = rejection_sampler_cls(spec_config=SimpleNamespace(method="mtp"))
    rejection_sampler.sampler = SimpleNamespace(
        sampling_states=SimpleNamespace(
            temperature=SimpleNamespace(
                np=np.array([0.0], dtype=np.float32),
            )
        )
    )
    rejection_sampler.watermark_key = None
    rejection_sampler.synthetic_conditional_rates = None
    rejection_sampler.use_block_verification = False
    rejection_sampler.fly_window_size = 0
    setattr(rejection_sampler, attribute, value)
    args = _make_verify_args()

    assert rejection_sampler._verify(*args) == "original"
    assert rejection_sampler.original_verify_args == args


@pytest.mark.parametrize("fallback_reason", ["random_sampling", "smaller_draft_vocab"])
def test_install_falls_back_when_batch_is_not_supported(fallback_reason):
    rejection_sampler_cls = _make_rejection_sampler_cls()
    greedy_verify_experiment.install(rejection_sampler_cls)

    temperature = 1.0 if fallback_reason == "random_sampling" else 0.0
    rejection_sampler = rejection_sampler_cls(spec_config=SimpleNamespace(method="mtp"))
    rejection_sampler.sampler = SimpleNamespace(
        sampling_states=SimpleNamespace(
            temperature=SimpleNamespace(
                np=np.array([temperature], dtype=np.float32),
            )
        )
    )
    rejection_sampler.watermark_key = None
    rejection_sampler.synthetic_conditional_rates = None
    rejection_sampler.use_block_verification = False
    rejection_sampler.fly_window_size = 0
    args = list(_make_verify_args())
    if fallback_reason == "smaller_draft_vocab":
        args[1] = torch.empty(1, 1, 1)
    args = tuple(args)

    assert rejection_sampler._verify(*args) == "original"
    assert rejection_sampler.original_verify_args == args


@pytest.mark.parametrize("method", ["eagle", "dflash", None])
def test_non_mtp_falls_back(method):
    cls = _make_rejection_sampler_cls()
    greedy_verify_experiment.install(cls)
    sampler = cls(spec_config=SimpleNamespace(method=method))
    args = _make_verify_args()
    assert sampler._verify(*args) == "original"
    assert sampler.original_verify_args == args


def test_entropy_verify_falls_back(monkeypatch):
    cls = _make_rejection_sampler_cls()
    greedy_verify_experiment.install(cls)
    sampler = cls(spec_config=SimpleNamespace(method="mtp"))
    config = SimpleNamespace(rejection_sampler_config=SimpleNamespace(enable_entropy_verify=True))
    monkeypatch.setattr(greedy_verify_experiment, "get_ascend_config", lambda: config)
    args = _make_verify_args()
    assert sampler._verify(*args) == "original"
    assert sampler.original_verify_args == args


def test_install_is_idempotent():
    cls = _make_rejection_sampler_cls()
    greedy_verify_experiment.install(cls)
    verify = cls._verify
    greedy_verify_experiment.install(cls)
    assert cls._verify is verify
