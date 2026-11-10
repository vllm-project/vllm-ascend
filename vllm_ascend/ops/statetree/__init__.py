# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Lossless StateTree weight compression for Ascend 950."""


def is_statetree_enabled() -> bool:
    """Return the typed feature switch without imposing init order on tests."""
    from vllm_ascend.ascend_config import get_ascend_config

    try:
        return get_ascend_config().enable_statetree
    except RuntimeError:
        # Layers created in isolated unit tests can precede platform config.
        return False
