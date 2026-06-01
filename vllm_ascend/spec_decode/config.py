# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The vLLM team.
#
# This file is a part of the vllm-ascend project.

ASCEND_SPEC_DECODE_METHOD_CONFIG_KEY = "ascend_spec_decode_method"
ZIPF_CONFIG_KEY = "zipf_config"

DEFAULT_ZIPF_NGRAM_SIZE = 4
DEFAULT_ZIPF_MIN_WINDOW = 2
MAX_ZIPF_NGRAM_SIZE = 8
DEFAULT_ZIPF_SKIP_SHARED = False
DEFAULT_ZIPF_GENERALIZED_BEFORE_SHARED = True


def get_ascend_spec_decode_method(vllm_config):
    """Return the Ascend-specific speculative decoding method.

    Upstream vLLM validates speculative_config.method against its own schema.
    Ascend-only methods that are not in upstream yet are selected through
    additional_config while speculative_config still enables the vLLM spec
    decode pipeline and provides num_speculative_tokens.
    """
    speculative_config = getattr(vllm_config, "speculative_config", None)
    if speculative_config is None:
        return None

    additional_config = getattr(vllm_config, "additional_config", None) or {}
    return additional_config.get(
        ASCEND_SPEC_DECODE_METHOD_CONFIG_KEY,
        speculative_config.method,
    )


def get_zipf_config(vllm_config):
    speculative_config = vllm_config.speculative_config
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    zipf_config = additional_config.get(ZIPF_CONFIG_KEY, {})
    if not isinstance(zipf_config, dict):
        raise ValueError("additional_config.zipf_config must be a dictionary.")

    max_speculative_tokens = speculative_config.num_speculative_tokens
    initial_speculative_tokens = zipf_config.get(
        "zipf_initial_speculative_tokens",
        max_speculative_tokens,
    )
    min_window = zipf_config.get("zipf_min_window", DEFAULT_ZIPF_MIN_WINDOW)
    ngram_size = zipf_config.get("zipf_ngram_size", DEFAULT_ZIPF_NGRAM_SIZE)
    skip_shared = zipf_config.get("zipf_skip_shared", DEFAULT_ZIPF_SKIP_SHARED)
    generalized_before_shared = zipf_config.get(
        "zipf_generalized_before_shared",
        DEFAULT_ZIPF_GENERALIZED_BEFORE_SHARED,
    )

    if not isinstance(initial_speculative_tokens, int):
        raise ValueError("zipf_initial_speculative_tokens must be an integer.")
    if initial_speculative_tokens <= 0:
        raise ValueError("zipf_initial_speculative_tokens must be greater than 0.")
    if initial_speculative_tokens > max_speculative_tokens:
        raise ValueError(
            "zipf_initial_speculative_tokens must be less than or equal to speculative_config.num_speculative_tokens."
        )

    if not isinstance(min_window, int) or not isinstance(ngram_size, int):
        raise ValueError("zipf_min_window and zipf_ngram_size must be integers.")
    if min_window <= 0:
        raise ValueError("zipf_min_window must be greater than 0.")
    if ngram_size < min_window:
        raise ValueError("zipf_ngram_size must be greater than or equal to zipf_min_window.")
    if ngram_size > MAX_ZIPF_NGRAM_SIZE:
        raise ValueError(f"zipf_ngram_size must be less than or equal to {MAX_ZIPF_NGRAM_SIZE}.")

    if not isinstance(skip_shared, bool):
        raise ValueError("zipf_skip_shared must be a boolean.")
    if not isinstance(generalized_before_shared, bool):
        raise ValueError("zipf_generalized_before_shared must be a boolean.")

    return {
        "zipf_initial_speculative_tokens": initial_speculative_tokens,
        "zipf_min_window": min_window,
        "zipf_ngram_size": ngram_size,
        "zipf_skip_shared": skip_shared,
        "zipf_generalized_before_shared": generalized_before_shared,
    }
