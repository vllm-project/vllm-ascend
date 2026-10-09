# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Startup warmup for compiled Ascend model paths.

The worker's regular ``_dummy_run`` is primarily shaped for CUDA-graph and
profile warmups.  STOCK_TORCH_COMPILE prefill needs a single synthetic request
with real attention metadata so Python branches in the model are traced before
the HTTP server accepts traffic.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import torch

from vllm.config import CompilationMode, CUDAGraphMode
from vllm.logger import logger


# Add every prefill token count that must be compiled before accepting traffic.
# Values are post-tokenizer token counts, not prompt character lengths.
PREFILL_WARMUP_TOKEN_SIZES: list[int] = [1,2,3,4,5]


@contextmanager
def _use_runtime_rope_tensor_kind():
    """Make dummy prefill RoPE tensors match execute_model tensors.

    DSA's full RoPE tables are regular tensors created while the model is
    loaded.  Indexing them from ``_dummy_run`` currently retains the
    ADInplaceOrView dispatch key, even though ``_dummy_run`` is in inference
    mode.  The scheduler-driven path produces inference tensors without that
    key.  Dynamo guards on the dispatch-key set, so normalize only the
    temporary, non-cached RoPE results used by this startup warmup.
    """
    import vllm_ascend.attention.context_parallel.dsa_cp as dsa_cp

    original_get_cos_and_sin_dsa = dsa_cp.get_cos_and_sin_dsa

    def get_cos_and_sin_dsa(*args, **kwargs):
        cos, sin = original_get_cos_and_sin_dsa(*args, **kwargs)
        if kwargs.get("use_cache", False):
            return cos, sin

        def as_inference_tensor(tensor):
            with torch.inference_mode():
                result = torch.empty_strided(
                    tensor.size(),
                    tensor.stride(),
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
                result.copy_(tensor)
            return result

        data = {
            config_key: {
                group_name: (
                    as_inference_tensor(cos_tensor),
                    as_inference_tensor(sin_tensor),
                )
                for group_name, (cos_tensor, sin_tensor) in groups.items()
            }
            for config_key, groups in cos._data.items()
        }
        proxy_type = type(cos)
        return proxy_type(data, is_cos=True), proxy_type(data, is_cos=False)

    dsa_cp.get_cos_and_sin_dsa = get_cos_and_sin_dsa
    try:
        yield
    finally:
        dsa_cp.get_cos_and_sin_dsa = original_get_cos_and_sin_dsa


def warmup_prefill(model_runner) -> None:
    """Compile STOCK_TORCH_COMPILE prefill graphs in the configured size list.

    ``profile_cpp=True`` makes ``_dummy_run`` model one request rather than
    distributing tokens over ``max_num_seqs`` requests.  ``force_attention``
    builds the same DSA/MLA metadata objects used by ``execute_model``.
    """
    compilation_config = model_runner.compilation_config
    if compilation_config.mode != CompilationMode.STOCK_TORCH_COMPILE:
        return
    if os.getenv("VLLM_ASCEND_PREFILL_WARMUP", "1").lower() in {
        "0",
        "false",
        "off",
        "no",
    }:
        logger.info("Prefill warmup disabled by VLLM_ASCEND_PREFILL_WARMUP")
        return
    if model_runner.model_config.enforce_eager:
        return
    if getattr(model_runner, "_stock_compiled_call", None) is None:
        logger.warning("Skipping prefill warmup: stock compiled call is unavailable")
        return

    max_tokens = model_runner.scheduler_config.max_num_batched_tokens
    if max_tokens < 1:
        return

    warmup_sizes: list[int] = []
    for num_tokens in PREFILL_WARMUP_TOKEN_SIZES:
        if not isinstance(num_tokens, int) or isinstance(num_tokens, bool):
            logger.warning("Skipping non-integer prefill warmup size: %r", num_tokens)
        elif num_tokens < 1 or num_tokens > max_tokens:
            logger.warning(
                "Skipping prefill warmup size %d; valid range is [1, %d]",
                num_tokens,
                max_tokens,
            )
        elif num_tokens not in warmup_sizes:
            warmup_sizes.append(num_tokens)

    if not warmup_sizes:
        logger.info("No valid prefill warmup sizes configured")
        return

    # ``_prepare_inputs`` sets this during normal scheduling.  ``_dummy_run``
    # constructs the same metadata, but does not persist the flag on the
    # runner, so set it explicitly to select ``_stock_compiled_call``.
    previous_with_prefill = getattr(model_runner, "with_prefill", False)
    model_runner.with_prefill = True
    try:
        with torch.inference_mode(), _use_runtime_rope_tensor_kind():
            for num_tokens in warmup_sizes:
                logger.info("Warming STOCK_TORCH_COMPILE prefill graph for %d tokens", num_tokens)
                model_runner._dummy_run(
                    num_tokens=num_tokens,
                    with_prefill=True,
                    cudagraph_runtime_mode=CUDAGraphMode.NONE,
                    force_attention=True,
                    is_profile=True,
                    profile_cpp=True,
                    force_prefill_attention_state=True,
                    in_profile_run=False,
                )

            # MTP schedules two target-model tokens immediately after the
            # prefill (the sampled token plus one speculative token).  The
            # scheduler invokes this step without attention metadata, so the
            # dummy run must do the same to satisfy Dynamo's type guards.
            if model_runner.speculative_config is not None:
                decode_tokens = int(model_runner.uniform_decode_query_len)
                model_runner._dummy_run(
                    num_tokens=decode_tokens,
                    with_prefill=True,
                    cudagraph_runtime_mode=CUDAGraphMode.NONE,
                    is_profile=True,
                    profile_cpp=True,
                    in_profile_run=False,
                )
    except Exception:
        # Warmup is an optimization.  A backend/compiler resource failure
        # must not make the API server unavailable; the first real request
        # can still compile (or report the same backend error) normally.
        logger.warning(
            "Prefill warmup failed; continuing without the remaining warmup sizes",
            exc_info=True,
        )
        return
    finally:
        model_runner.with_prefill = previous_with_prefill
    logger.info("Finished STOCK_TORCH_COMPILE prefill warmup for sizes %s", warmup_sizes)
