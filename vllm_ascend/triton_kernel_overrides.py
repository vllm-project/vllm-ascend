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

This table replaces the retired ``patch/worker/patch_triton.py``
monkey-patch file. Each entry maps one upstream Triton kernel to its Ascend
implementation; ``register_triton_kernels()`` hands the whole table to
``vllm.triton_utils.dispatcher.register_kernels``, which swaps the defining
attribute and sweeps every ``from``-import copy by identity.

Three rules when editing the table:

- Keys must target the module where the kernel is *defined*
  (``func.__module__``), never a from-import copy site. The identity sweep
  matches the original object; a key pointing at a copy silently defeats it.
- Parameter contract: mirror the original's parameter names or order
  -- either one suffices for the call shape the impl receives, and
  mirroring both covers every shape. The dispatcher forwards two call
  shapes verbatim and adapts only the third:

  * Plain calls (``kernel(...)``) and positional ``kernel[grid]``
    calls reach the impl untouched -- no renaming, no reordering. The
    impl must accept every argument name its callers pass, via an
    explicit parameter or ``**kwargs``.
  * Keyword-only ``kernel[grid]`` calls are adapted: matching names
    forward by name; otherwise the dispatcher forwards positionally
    in the original's order -- arguments the caller omitted are
    skipped (later values shift left), and names outside the
    original's signature are rejected.

  Two consequences: never rename an original parameter -- a verbatim
  call site that passes it by name hits a TypeError, and a keyword-only
  ``[grid]`` call silently shifts the value into the wrong positional
  slot. Keep NPU-only parameters in the impl's tail with defaults so a
  keyword-only ``[grid]`` relaunch still binds every original argument
  to its slot (see ``rejection_sample``; plain-call hosts bind by name
  and tolerate any position, the tail rule covers both shapes).
- Values are either plain callables or raw ``@triton.jit`` kernels. The
  dispatcher swaps a raw kernel in directly -- never wrapping it -- so it
  launches natively with the caller's grid and must mirror the original
  kernel's parameter names or order; this is also the only shape that
  works for Triton *device functions*, which are inlined into other
  kernels at compile time. A plain callable is wrapped in
  ``KernelOverride`` and invoked as a plain function: launch arguments
  are forwarded, but the caller's grid reaches the impl only when it
  reimplements ``__getitem__`` -- so impls that relaunch with a different
  grid are written as launch adapters (see the ``ops/triton/`` modules).

Warmup: ``KernelOverride.warmup`` delegates only to values that are
themselves Triton kernels, so raw swaps keep compiling before ACL graph
capture while plain callables compile lazily on first launch. That
matches the retired patch: the JIT warmup registry only ever registered
vllm-ascend-owned kernels, so no table entry had pre-capture compilation
to lose.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vllm_ascend.logger import init_logger_ascend
from vllm_ascend.ops.triton.apply_grammar_bitmask import apply_grammar_bitmask_npu
from vllm_ascend.ops.triton.fla.chunk import chunk_gated_delta_rule
from vllm_ascend.ops.triton.kda.kda import (
    chunk_kda,
    chunk_kda_with_fused_gate,
    fused_kda_gate,
    fused_recurrent_kda,
)
from vllm_ascend.ops.triton.mamba.precopy import precopy_mamba_align_fused_kernel
from vllm_ascend.ops.triton.metrics.num_nans import get_num_nans
from vllm_ascend.ops.triton.sample.categorical_sample import categorical_sample
from vllm_ascend.ops.triton.sample.fill_logprob_token_idx import (
    _fill_logprob_token_ids_kernel,
)
from vllm_ascend.ops.triton.sample.sampling_mask import sampling_mask_from_logits_npu
from vllm_ascend.ops.triton.sample.thinking_budget import (
    _load_effective_token_ascend,
    _update_committed_marker_cache_kernel_ascend,
)
from vllm_ascend.worker.v2.sample.apply_top_k_top_p import apply_top_k_top_p
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
    "vllm.v1.worker.gpu.sample.logprob.compute_token_logprobs": compute_token_logprobs,
    # Upstream renamed compute_topk_logprobs to compute_topk_scores and
    # moved the definition into logprob.py; the other three modules
    # from-import it. One registration covers the three from-import copies
    # of the retired patch. The impl keeps our historical name; its
    # signature matches the upstream compute_topk_scores.
    "vllm.v1.worker.gpu.sample.logprob.compute_topk_scores": compute_topk_logprobs,
    # TODO: Remove when upstream guards empty launch grids (Ascend
    # rejects an empty grid with coreDim=0).
    "vllm.v1.worker.gpu.sample.trace_replay.apply_trace_tokens": apply_trace_tokens,
    # TODO: Remove after Triton-Ascend 3.6.0 is the minimum supported
    # version.
    "vllm.v1.worker.gpu.sample.thinking_budget._update_committed_marker_cache_kernel": (  # noqa: E501
        _update_committed_marker_cache_kernel_ascend
    ),
    # A Triton *device function* (inlined into other kernels at compile
    # time), so the value must itself be a Triton kernel: the dispatcher
    # swaps such values in directly and never wraps them, and the impl
    # mirrors the original's parameter names and order. The Ascend
    # rewrite replaces the upstream branch-local return with a masked
    # tl.load.
    # TODO: Remove after the new Q4 Triton-Ascend release is available.
    "vllm.v1.worker.gpu.sample.thinking_budget._load_effective_token": (  # noqa: E501
        _load_effective_token_ascend
    ),
    # Class mount. KernelOverride is not a descriptor, so no implicit `cls`
    # is passed at call time; sampling_mask_from_logits_npu therefore keeps
    # a cls-free signature (the retired patch wrapped it in classmethod()).
    # Rationale: strided vocab-dimension loads over-allocate NPU UB.
    # TODO: Remove when Triton-Ascend can compile and efficiently lower
    # the upstream kernel.
    "vllm.v1.worker.gpu.sample.output.SamplingMaskTensors.from_logits": sampling_mask_from_logits_npu,
    # Production-dead today: upstream's only call site sits inside
    # compute_topk_scores, which the registration above replaces with
    # the Ascend host path -- that path warns and falls back to default
    # topk (worker/v2/sample/logprob.py) and never launches this kernel.
    # Revives as-is once the Ascend host grows the custom
    # logprob_token_ids branch.
    # TODO: Remove once the compiler and Triton-Ascend toolkit support
    # the upstream fill_logprob_token_ids_kernel.
    "vllm.v1.worker.gpu.sample.logprob._fill_logprob_token_ids_kernel": _fill_logprob_token_ids_kernel,
    # Defined in metrics/logits.py; sampler.py and rejection_sampler.py are
    # copy sites (retired patch: two assignments).
    # TODO: Remove once the compiler and Triton-Ascend toolkit support the
    # upstream get_num_nans.
    "vllm.v1.worker.gpu.metrics.logits.get_num_nans": get_num_nans,
    # ---- vllm.v1.worker.gpu.spec_decode ------------------------------------
    # rejection_sample is defined in rejection_sampler_utils; the
    # rejection_sampler module only from-imports it. One registration
    # covers both retired assignments.
    "vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils.rejection_sample": npu_rejection_sample,
    # ---- other upstream modules --------------------------------------------
    # Defined in topk_topp_triton.py; topk_topp_sampler only from-imports
    # it. The identity sweep covers every copy site.
    # Latent mismatch: upstream's ``mask_value`` has no parameter on the
    # Ascend impl. The sole caller never passes it today, so nothing
    # breaks; the parameter (and its semantics) belong with the follow-up
    # signature work.
    "vllm.v1.sample.ops.topk_topp_triton.apply_top_k_top_p_triton": apply_top_k_top_p,
    "vllm.v1.worker.gpu.structured_outputs._apply_grammar_bitmask_kernel": apply_grammar_bitmask_npu,
    "vllm.v1.worker.mamba_utils.precopy_mamba_align_fused_kernel": precopy_mamba_align_fused_kernel,
    # ---- vllm.third_party.flash_linear_attention.* ------------------------
    # chunk_gated_delta_rule is defined in ops/chunk.py; the ops package
    # re-exports it and the olmo/qwen GDN layers import that copy. The
    # identity sweep rebinds every copy (the retired patch reassigned the
    # package attribute directly).
    "vllm.third_party.flash_linear_attention.ops.chunk.chunk_gated_delta_rule": (  # noqa: E501
        chunk_gated_delta_rule
    ),
    # The KDA primitives are defined in ops/kda.py. The upstream
    # bailing_moe_v3 layer from-imports them directly from the module
    # and is their only consumer on this route: Kimi K3 ships a
    # self-contained vendored copy under models/kimi_k3/nvidia and
    # GLM-5 runs the AscendC kernels through
    # vllm_ascend/models/glm5next/ops/kda.py, so neither reaches these
    # keys. chunk_kda_with_fused_gate has no upstream caller today; it
    # stays for parity with the retired patch so a future
    # vllm.third_party consumer lands on the NPU impl.
    "vllm.third_party.flash_linear_attention.ops.kda.chunk_kda": chunk_kda,
    "vllm.third_party.flash_linear_attention.ops.kda.chunk_kda_with_fused_gate": (  # noqa: E501
        chunk_kda_with_fused_gate
    ),
    # Known-broken at its only launch site: bailing_moe_v3 passes
    # ``lower_bound=``, which the Ascend impl does not accept yet, so
    # the verbatim forward raises TypeError; upstream feeds
    # ``lower_bound or 0.0`` plus a ``USE_LOWER_BOUND`` constexpr. No
    # bailing e2e exists, so CI cannot catch it; the impl fix is a
    # separate follow-up. The other two bailing calls are safe:
    # ``safe_gate``/``lower_bound`` vanish into upstream's own
    # ``**kwargs`` on GPU (upstream never reads them) and the NPU
    # impls swallow them the same way; fused_recurrent_kda goes
    # further and implements ``lower_bound`` (an NPU extension with
    # its own default), a superset there, not a gap.
    "vllm.third_party.flash_linear_attention.ops.kda.fused_kda_gate": fused_kda_gate,
    "vllm.third_party.flash_linear_attention.ops.kda.fused_recurrent_kda": (  # noqa: E501
        fused_recurrent_kda
    ),
}


# One-way latch: vLLM invokes the platform hook from every
# WorkerWrapperBase.init_worker, one process may create several workers
# (in-process TP), and the e2e conftest registers directly before the
# in-process engine starts -- so both entry points must stay idempotent.
# Re-registering would double-wrap kernels, re-run the identity sweep on
# a possibly polluted module set, and corrupt the serve smoke signals.
_kernels_registered = False


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
