# SPDX-License-Identifier: Apache-2.0
"""Explicit boundaries for GPU Worker facilities unavailable on Ascend."""


class UnsupportedElasticEPExecutor:
    """No thread pool is constructed when elastic EP is unsupported."""

    def execute(self, execute_method: str, *args, **kwargs):
        raise NotImplementedError("Ascend does not support elastic expert-parallel reconfiguration.")

    def shutdown(self) -> None:
        # There is deliberately no executor resource to release.
        pass


# These methods must continue resolving to vLLM after dependency upgrades.
INHERITED_WORKER_METHODS = (
    "execute_model",
    "_check_weight_transfer_engine",
    "init_weight_transfer_engine",
    "start_weight_update",
    "update_weights",
    "finish_weight_update",
    "shutdown_weight_transfer_engine",
    "get_model",
    "get_supported_tasks",
    "take_draft_token_ids",
    "add_lora",
    "remove_lora",
    "list_loras",
    "pin_lora",
    "reset_encoder_cache",
    "update_max_model_len",
    "reload_weights",
    "update_config",
)
