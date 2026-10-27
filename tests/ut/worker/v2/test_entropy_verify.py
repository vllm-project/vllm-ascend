# SPDX-License-Identifier: Apache-2.0

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.worker.gpu.spec_decode import rejection_sampler

from vllm_ascend.ascend_config import RejectionSamplerConfig
from vllm_ascend.worker.v2.spec_decode import rejection_sampler_utils as utils


@pytest.fixture
def inputs():
    return dict(
        target_logits=torch.zeros(2, 3),
        draft_logits=None,
        draft_sampled=torch.zeros(2, dtype=torch.int64),
        cu_num_logits=torch.tensor([0, 2], dtype=torch.int32),
        pos=torch.arange(2, dtype=torch.int32),
        idx_mapping=torch.zeros(1, dtype=torch.int32),
        expanded_idx_mapping=torch.zeros(2, dtype=torch.int32),
        expanded_local_pos=torch.arange(2, dtype=torch.int32),
        temperature=torch.ones(1),
        seed=torch.zeros(1, dtype=torch.int64),
        num_speculative_steps=1,
    )


@pytest.fixture
def launches(monkeypatch):
    """Exercise host dispatch on CPU; numerical kernels have separate NPU tests."""
    # The CPU Triton placeholder does not provide next_power_of_2.
    monkeypatch.setattr(
        utils,
        "triton",
        SimpleNamespace(cdiv=utils.triton.cdiv, next_power_of_2=lambda n: 1 << (n - 1).bit_length()),
    )
    mocks = {}
    for name in (
        "_compute_block_stats_kernel",
        "_entropy_partials_kernel",
        "_compute_cumulative_log_p_kernel",
        "_compute_local_residual_mass_kernel",
        "_probabilistic_rejection_kernel",
    ):
        kernel = MagicMock()
        monkeypatch.setattr(utils, name, kernel)
        mocks[name] = kernel.__getitem__.return_value
    mocks["resample"] = MagicMock()
    monkeypatch.setattr(utils, "resample", mocks["resample"])
    return mocks


def set_config(monkeypatch, **kwargs):
    config = RejectionSamplerConfig(**kwargs)
    getter = MagicMock(return_value=SimpleNamespace(rejection_sampler_config=config))
    monkeypatch.setattr(utils, "get_ascend_config", getter)
    return getter


def test_worker_config_is_resolved_per_call(monkeypatch, inputs, launches):
    # Reinitializing worker configuration must not reuse the previous thresholds.
    for enabled, threshold, alpha in ((True, 0.6, 0.0), (False, 0.95, 0.4), (True, 0.8, 1.2)):
        getter = set_config(
            monkeypatch, enable_entropy_verify=enabled, posterior_threshold=threshold, posterior_alpha=alpha
        )
        for mock in launches.values():
            mock.reset_mock()
        utils.rejection_sample(**inputs)
        getter.assert_called_once_with()
        kwargs = launches["_probabilistic_rejection_kernel"].call_args.kwargs
        assert kwargs["USE_ENTROPY_VERIFY"] is enabled
        assert launches["_entropy_partials_kernel"].called is (enabled and alpha > 0)
        assert kwargs["POSTERIOR_THRESHOLD"] == (threshold if enabled else 1.0)
        assert kwargs["POSTERIOR_ALPHA"] == (alpha if enabled else 0.0)
        assert launches["resample"].call_args.kwargs["use_block_verification"] is False


@pytest.mark.parametrize("enabled", [False, True])
def test_explicit_override_does_not_read_worker_config(monkeypatch, inputs, launches, enabled):
    getter = set_config(monkeypatch, enable_entropy_verify=not enabled)
    utils.rejection_sample(**inputs, use_entropy_verify=enabled, posterior_threshold=0.7, posterior_alpha=0)
    getter.assert_not_called()
    assert launches["_probabilistic_rejection_kernel"].call_args.kwargs["USE_ENTROPY_VERIFY"] is enabled
    if enabled:
        assert launches["_probabilistic_rejection_kernel"].call_args.kwargs["POSTERIOR_THRESHOLD"] == 0.7


@pytest.mark.parametrize("enabled,alpha,synthetic", [(False, 0.4, False), (True, 0.0, False), (True, 0.4, True)])
def test_unused_entropy_skips_allocations(monkeypatch, inputs, launches, enabled, alpha, synthetic):
    set_config(monkeypatch, enable_entropy_verify=enabled, posterior_alpha=alpha)
    empty_like = MagicMock(side_effect=AssertionError("Unexpected entropy allocation"))
    monkeypatch.setattr(utils.torch, "empty_like", empty_like)
    utils.rejection_sample(**inputs, synthetic_conditional_rates=torch.ones(1) if synthetic else None)
    launches["_entropy_partials_kernel"].assert_not_called()
    kwargs = launches["_probabilistic_rejection_kernel"].call_args.kwargs
    assert kwargs["SYNTHETIC_MODE"] is synthetic
    assert kwargs["USE_ENTROPY_VERIFY"] is (enabled and not synthetic)
    assert kwargs["POSTERIOR_ALPHA"] == 0.0


def test_entropy_rejects_upstream_block(monkeypatch, inputs, launches):
    set_config(monkeypatch, enable_entropy_verify=True)
    with pytest.raises(ValueError, match="cannot be combined"):
        utils.rejection_sample(**inputs, use_block_verification=True)
    for mock in launches.values():
        mock.assert_not_called()


@pytest.mark.parametrize("threshold,alpha", [(0, 0.4), (1.01, 0.4), (0.95, -0.1), (float("nan"), 0.4)])
def test_invalid_entropy_parameters_fail_before_launch(inputs, launches, threshold, alpha):
    with pytest.raises(ValueError, match="posterior_threshold"):
        utils.rejection_sample(**inputs, use_entropy_verify=True, posterior_threshold=threshold, posterior_alpha=alpha)
    for mock in launches.values():
        mock.assert_not_called()


def test_disabled_entropy_preserves_upstream_block(monkeypatch, inputs, launches):
    set_config(monkeypatch)
    utils.rejection_sample(**inputs, use_block_verification=True)
    launches["_compute_cumulative_log_p_kernel"].assert_called_once()
    launches["_entropy_partials_kernel"].assert_not_called()
    assert launches["_probabilistic_rejection_kernel"].call_args.kwargs["USE_BLOCK_VERIFICATION"] is True
    assert launches["resample"].call_args.kwargs["use_block_verification"] is True


def test_upstream_verify_uses_processed_logits_without_class_patch(monkeypatch, inputs, launches):
    set_config(monkeypatch, enable_entropy_verify=True)
    # This is the existing module-function replacement from patch_triton.py.
    monkeypatch.setattr(rejection_sampler, "rejection_sample", utils.rejection_sample)
    verifier = rejection_sampler.RejectionSampler.__new__(rejection_sampler.RejectionSampler)
    verifier.sampler = MagicMock()
    processed = inputs["target_logits"] + 1
    verifier.sampler.apply_sampling_params.return_value = processed
    verifier.sampler.sampling_states.temperature.gpu = inputs["temperature"]
    verifier.sampler.sampling_states.seeds.gpu = inputs["seed"]
    verifier.sampler.use_fp64_gumbel = False
    verifier.num_speculative_steps = 1
    verifier.synthetic_conditional_rates = None
    verifier.use_block_verification = False
    verifier.watermark_key = None
    verifier.fly_window_size = 0
    verifier.fly_entropy_threshold = 0.3
    verifier.fly_entropy_top_k = 3
    # Upstream main adds this argument; the supported release omits it.
    verify_kwargs = {}
    if "verify_draft_sampled" in inspect.signature(verifier._verify).parameters:
        verify_kwargs["verify_draft_sampled"] = inputs["draft_sampled"]
    result, _, _ = verifier._verify(
        inputs["target_logits"],
        inputs["draft_logits"],
        inputs["draft_sampled"],
        inputs["pos"],
        inputs["cu_num_logits"],
        inputs["idx_mapping"],
        inputs["idx_mapping"].numpy(),
        inputs["expanded_idx_mapping"],
        inputs["expanded_local_pos"],
        inputs["cu_num_logits"][1:].numpy(),
        **verify_kwargs,
    )
    assert result is processed
    verifier.sampler.apply_sampling_params.assert_called_once()
    assert launches["_entropy_partials_kernel"].call_args.args[0] is processed
    assert launches["_probabilistic_rejection_kernel"].call_args.args[5] is processed
