# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


def uses_prefill_ops(vllm_config) -> bool:
    """Select the optimized operator family for dedicated Prefill instances."""
    transfer = vllm_config.kv_transfer_config
    return transfer is not None and transfer.kv_role == "kv_producer"
