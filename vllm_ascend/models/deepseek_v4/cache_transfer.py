# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

def transfer_cache_views(kv_caches):
    """Transfer A5 indexer data and scales once, preserving the kernel views."""
    return {
        name: views[:2]
        if name.endswith(".indexer.k_cache") and isinstance(views, (tuple, list)) and len(views) == 3
        else views
        for name, views in kv_caches.items()
    }
