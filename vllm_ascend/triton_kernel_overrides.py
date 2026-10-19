#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""NPU Triton kernel overrides registered with the vLLM kernel dispatcher.

This table replaces the retired ``patch/worker/patch_v2/patch_triton.py``
monkey-patch file. Each entry maps one upstream Triton kernel to its Ascend
implementation; ``register_triton_kernels()`` hands the whole table to
``vllm.triton_utils.dispatcher.register_kernels``, which swaps the defining
attribute and sweeps every ``from``-import copy by identity.

Two rules when editing the table:

- Keys must target the module where the kernel is *defined*
  (``func.__module__``), never a from-import copy site. The identity sweep
  matches the original object; a key pointing at a copy silently defeats it.
- Values only need their parameter *order* to match the original kernel;
  parameter names may differ (the dispatcher reorders launch kwargs).
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from vllm_ascend.logger import init_logger_ascend
from vllm_ascend.ops.triton.v2.apply_grammar_bitmask import apply_grammar_bitmask_npu
from vllm_ascend.ops.triton.v2.mamba.precopy import precopy_mamba_align_fused_npu
from vllm_ascend.ops.triton.v2.metrics.num_nans import get_num_nans
from vllm_ascend.ops.triton.v2.sample.categorical_sample import categorical_sample
from vllm_ascend.ops.triton.v2.sample.fill_logprob_token_idx import (
    fill_logprob_token_ids_npu,
)
from vllm_ascend.ops.triton.v2.sample.sampling_mask import sampling_mask_from_logits_npu
from vllm_ascend.ops.triton.v2.sample.thinking_budget import (
    _load_effective_token_ascend,
    update_committed_marker_cache_npu,
)
from vllm_ascend.worker.v2.sample.apply_top_k_top_p import apply_top_k_top_p_npu
from vllm_ascend.worker.v2.sample.bad_words import apply_bad_words
from vllm_ascend.worker.v2.sample.gumbel import apply_temperature
from vllm_ascend.worker.v2.sample.logprob import (
    compute_token_logprobs,
    compute_topk_logprobs,
)
from vllm_ascend.worker.v2.sample.min_p import apply_min_p
from vllm_ascend.worker.v2.sample.penalties import apply_penalties, bincount
from vllm_ascend.worker.v2.sample.trace_replay import apply_trace_tokens
from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import (
    rejection_sample as npu_rejection_sample,
)

# A stdlib logger on the vllm_ascend tree, immune to the UCM monkey-patch
# of vllm.logger.init_logger; configure_ascend_logging() (NPUPlatform
# startup) attaches the console handler that makes these logs visible.
logger = init_logger_ascend(__name__)

TRITON_KERNEL_OVERRIDES: dict[str, Callable[..., Any]] = {
    # ---- vllm.v1.worker.gpu.sample.* host wrappers ------------------------
    # gumbel_sample is defined in gumbel.py; the eagle / spec-decode base /
    # dspark speculators and the sampler only from-import it. The identity
    # sweep covers every copy (the retired patch needed five assignments).
    "vllm.v1.worker.gpu.sample.gumbel.gumbel_sample": categorical_sample,
    # apply_temperature is also defined in gumbel.py; states.py is a copy
    # site (retired patch: two assignments).
    "vllm.v1.worker.gpu.sample.gumbel.apply_temperature": apply_temperature,
    # apply_min_p is defined in min_p.py; states.py only from-imports it.
    # The identity sweep covers every copy site.
    "vllm.v1.worker.gpu.sample.min_p.apply_min_p": apply_min_p,
    "vllm.v1.worker.gpu.sample.penalties.apply_penalties": apply_penalties,
    "vllm.v1.worker.gpu.sample.penalties.bincount": bincount,
    "vllm.v1.worker.gpu.sample.bad_words.apply_bad_words": apply_bad_words,
    "vllm.v1.worker.gpu.sample.logprob.compute_token_logprobs": (compute_token_logprobs),
    # Upstream renamed compute_topk_logprobs to compute_topk_scores and
    # moved the definition into logprob.py; the other three modules
    # from-import it. One registration covers the three from-import copies
    # of the retired patch. The impl keeps our historical name; its
    # signature matches the upstream compute_topk_scores.
    "vllm.v1.worker.gpu.sample.logprob.compute_topk_scores": (compute_topk_logprobs),
    # TODO: Remove when upstream guards empty launch grids (Ascend
    # rejects an empty grid with coreDim=0).
    "vllm.v1.worker.gpu.sample.trace_replay.apply_trace_tokens": (apply_trace_tokens),
    # TODO: Remove after Triton-Ascend 3.6.0 is the minimum supported
    # version.
    "vllm.v1.worker.gpu.sample.thinking_budget._update_committed_marker_cache_kernel": (  # noqa: E501
        update_committed_marker_cache_npu
    ),
    # Class mount. KernelOverride is not a descriptor, so no implicit `cls`
    # is passed at call time; sampling_mask_from_logits_npu therefore keeps
    # a cls-free signature (the retired patch wrapped it in classmethod()).
    # Rationale: strided vocab-dimension loads over-allocate NPU UB.
    # TODO: Remove when Triton-Ascend can compile and efficiently lower
    # the upstream kernel.
    "vllm.v1.worker.gpu.sample.output.SamplingMaskTensors.from_logits": (sampling_mask_from_logits_npu),
    # TODO: Remove once the compiler and Triton-Ascend toolkit support the
    # upstream fill_logprob_token_ids_kernel.
    "vllm.v1.worker.gpu.sample.logprob._fill_logprob_token_ids_kernel": (fill_logprob_token_ids_npu),
    # Defined in metrics/logits.py; sampler.py and rejection_sampler.py are
    # copy sites (retired patch: two assignments).
    # TODO: Remove once the compiler and Triton-Ascend toolkit support the
    # upstream get_num_nans.
    "vllm.v1.worker.gpu.metrics.logits.get_num_nans": get_num_nans,
    # ---- vllm.v1.worker.gpu.spec_decode ------------------------------------
    # rejection_sample is defined in rejection_sampler_utils; the
    # rejection_sampler module only from-imports it. One registration
    # covers both retired assignments.
    "vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils.rejection_sample": (npu_rejection_sample),
    # ---- other upstream modules --------------------------------------------
    # Defined in topk_topp_triton.py; topk_topp_sampler only from-imports
    # it. The identity sweep covers every copy site.
    "vllm.v1.sample.ops.topk_topp_triton.apply_top_k_top_p_triton": (apply_top_k_top_p_npu),
    "vllm.v1.worker.gpu.structured_outputs._apply_grammar_bitmask_kernel": (apply_grammar_bitmask_npu),
    "vllm.v1.worker.mamba_utils.precopy_mamba_align_fused_kernel": (precopy_mamba_align_fused_npu),
}


# Triton *device functions* cannot go through the dispatcher: they are
# inlined at kernel-compile time (the callee is resolved from the
# defining module's globals), and the dispatcher's KernelOverride is not
# a JITFunction -- compiling a kernel that calls the swapped symbol
# would fail with "not a Triton function". These swaps keep the retired
# patch file's plain-attribute semantics; only the defining module is
# patched, which is exactly where the inliner looks.
_DEVICE_FUNCTION_SWAPS: dict[str, Callable[..., Any]] = {
    # TODO: Remove after the new Q4 Triton-Ascend release is available.
    "vllm.v1.worker.gpu.sample.thinking_budget._load_effective_token": (_load_effective_token_ascend),
}


# One-way latches: vLLM invokes the platform hook from every
# WorkerWrapperBase.init_worker, one process may create several workers
# (in-process TP), and the e2e conftest registers directly before the
# in-process engine starts -- so both entry points must stay idempotent.
# Re-registering would double-wrap kernels, re-run the identity sweep on
# a possibly polluted module set, and corrupt the serve smoke signals.
_kernels_registered = False
_device_functions_swapped = False


def swap_device_functions() -> None:
    """Swap Triton device functions the dispatcher cannot wrap.
    
    Device functions are ``@triton.jit`` helpers called from inside other
    kernels' bodies and inlined at kernel-compile time; there is no host
    launch for the dispatcher to intercept, and a ``KernelOverride`` is
    not a ``JITFunction``, so a wrapped name would break the outer
    kernel's compilation. The raw attribute swap below keeps the retired
    patch's semantics: only the defining module is patched, which is
    where the inliner resolves the callee.
    """
    global _device_functions_swapped
    if _device_functions_swapped:
        return
    for dotted, impl in _DEVICE_FUNCTION_SWAPS.items():
        module_name, attr = dotted.rsplit(".", 1)
        setattr(importlib.import_module(module_name), attr, impl)
        logger.debug("Swapped device function %s", dotted)
    _device_functions_swapped = True
    logger.info("Swapped %d NPU Triton device functions", len(_DEVICE_FUNCTION_SWAPS))


def register_triton_kernels() -> None:
    """Register all NPU Triton kernel overrides with the vLLM dispatcher."""
    global _kernels_registered
    if _kernels_registered:
        return
    # Fail fast: an ImportError here means the installed vLLM predates the
    # kernel dispatcher (vllm-project/vllm#43048) and this plugin cannot
    # install its kernel overrides.
    from vllm.triton_utils.dispatcher import register_kernels

    register_kernels(TRITON_KERNEL_OVERRIDES)
    _kernels_registered = True
    logger.info(
        "Registered %d NPU Triton kernel overrides",
        len(TRITON_KERNEL_OVERRIDES),
    )
