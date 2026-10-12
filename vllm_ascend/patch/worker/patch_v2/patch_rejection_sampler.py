# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Register the prefill sampling adapters through the Ascend worker patches."""

from functools import wraps
from typing import Any

from vllm.logger import logger
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

from vllm_ascend.patch.worker.patch_v2.prefill_rejection_sampler import install_prefill_placeholder_mask
from vllm_ascend.patch.worker.patch_v2.recompute_rejection_sampler import RecomputeAwareRejectionSampler
from vllm_ascend.utils import is_pd_decode_recompute_scheduler_enabled
from vllm_ascend.worker.v2.model_runner import NPUModelRunner


def install_rejection_sampler_patch(runner_cls: type[Any]) -> None:
    """Wrap only Ascend V2 loading, including its existing inherited paths.

    Capturing the original method in a closure makes re-imports safe. Keep
    upstream GPU/V1 runners and the upstream sampler class untouched.
    """
    original_load_model = runner_cls.load_model
    if getattr(original_load_model, "_ascend_prefill_boundary", False):
        return

    @wraps(original_load_model)
    def load_model(self, load_dummy_weights: bool = False, *args, **kwargs) -> None:
        original_load_model(self, load_dummy_weights, *args, **kwargs)
        sampler = self.rejection_sampler
        if sampler is None:
            return
        inner_sampler = sampler._sampler if isinstance(sampler, RecomputeAwareRejectionSampler) else sampler
        if isinstance(inner_sampler, RejectionSampler):
            # Ordinary prefill does not depend on PD roles or recompute.
            # Custom entry points that bypass verification stay unchanged.
            install_prefill_placeholder_mask(inner_sampler)
        if is_pd_decode_recompute_scheduler_enabled(self.vllm_config) and not isinstance(
            sampler, RecomputeAwareRejectionSampler
        ):
            self.rejection_sampler = RecomputeAwareRejectionSampler(sampler)
            logger.info("Recompute sampler boundary adapter active: source=%s", __file__)

    load_model._ascend_prefill_boundary = True  # type: ignore[attr-defined]
    runner_cls.load_model = load_model


install_rejection_sampler_patch(NPUModelRunner)
