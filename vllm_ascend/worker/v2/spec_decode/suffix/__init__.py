# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project


def validate_suffix_runtime(speculative_config) -> None:
    """Require the complete upstream protocol, rather than backporting core hooks."""
    # Lazy import: runner selection also calls this in frontend processes.
    from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator

    hooks = ("add_request", "on_requests_finished", "reset_state")
    if not hasattr(speculative_config, "suffix_decoding_corpus_tokens") or not all(
        callable(getattr(BaseSpeculator, hook, None)) for hook in hooks
    ):
        raise ValueError(
            "Ascend MRV2 suffix decoding requires vLLM #60339 "
            "(190df0b90007eead441d97d3abf6270f84f2c63e) or a descendant, "
            "including suffix configuration and request lifecycle hooks. "
            "Use VLLM_USE_V2_MODEL_RUNNER=0 with an older vLLM version."
        )
