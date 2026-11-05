# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from importlib import import_module


def validate_suffix_runtime(speculative_config) -> None:
    """Require the complete upstream protocol, rather than backporting core hooks."""
    # Lazy import: runner selection also calls this in frontend processes.
    from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator

    message = (
        "Ascend MRV2 suffix decoding requires vLLM #60339 "
        "(190df0b90007eead441d97d3abf6270f84f2c63e) or a descendant, "
        "including SuffixSpeculator, suffix configuration and request lifecycle hooks. "
        "Use VLLM_USE_V2_MODEL_RUNNER=0 with an older vLLM version."
    )
    hooks = ("add_request", "on_requests_finished", "reset_state")
    if not hasattr(speculative_config, "suffix_decoding_corpus_tokens") or not all(
        callable(getattr(BaseSpeculator, hook, None)) for hook in hooks
    ):
        raise ValueError(message)

    # The Ascend class now inherits this concrete implementation. Check it
    # before importing the subclass, while preserving unrelated import errors.
    module_name = "vllm.v1.worker.gpu.spec_decode.suffix.speculator"
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name and (module_name == exc.name or module_name.startswith(exc.name + ".")):
            raise ValueError(message) from exc
        raise
    if not isinstance(getattr(module, "SuffixSpeculator", None), type):
        raise ValueError(message)
