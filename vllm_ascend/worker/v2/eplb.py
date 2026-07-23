# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project


def is_eplb_load_scope_matched(load_scope: str, batch_has_prefill: bool) -> bool:
    """Return whether the whole batch belongs to the configured load scope."""
    if load_scope == "all":
        return True
    batch_scope = "prefill" if batch_has_prefill else "decode"
    return load_scope == batch_scope
