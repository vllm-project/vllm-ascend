# SPDX-License-Identifier: Apache-2.0
"""Validate the new KV label contract in vLLM's existing input validation step."""

import functools


def install_request_validation():
    from vllm.exceptions import VLLMValidationError
    from vllm.v1.engine.input_processor import InputProcessor

    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.qos import KvQosPolicy

    original = InputProcessor._validate_params
    if getattr(original, "_ascend_ai_qos_validated", False):
        return

    @functools.wraps(original)
    def validate(self, params, supported_tasks):
        transfer = self.vllm_config.kv_transfer_config
        config = transfer.get_from_extra_config("kv_qos", None) if transfer is not None else None
        if isinstance(config, dict) and config.get("level_names") is True:
            policy = KvQosPolicy.from_config(config)
            extra = getattr(params, "extra_args", None) or {}
            request_params = extra.get("kv_transfer_params")
            try:
                policy.resolve_priority(request_params)
            except ValueError as exc:
                # AsyncLLM wraps raw ValueError as EngineGenerateError (HTTP
                # 500). Classify only request input failures as client errors;
                # policy construction and original processing stay unchanged.
                parameter = "kv_transfer_params"
                if isinstance(request_params, dict):
                    parameter += ".kv_priority"
                raise VLLMValidationError(str(exc), parameter=parameter) from exc
        return original(self, params, supported_tasks)

    validate._ascend_ai_qos_validated = True
    InputProcessor._validate_params = validate
