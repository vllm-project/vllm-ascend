# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One compatibility shim for vLLM versions that lack ``get_all_blocks``.

Nothing else lives here: the Ascend port publishes no recovery-unit events and
applies no pending state gathers, so the earlier placeholder payload classes
were removed.
"""


def install_block_hash_map_compat() -> None:
    """Add ``BlockHashToBlockMap.get_all_blocks`` on vLLM 0.30.

    The frontier manager asks the prefix-cache map for every block that a hash
    maps to (a hash may be shared by several blocks).  That helper only exists
    in later vLLM releases, so on 0.30 every publish raised AttributeError and
    was dropped.  The 0.30 map uses the same ``_cache`` layout, so the shim is
    a straight re-implementation.
    """
    from vllm.v1.core.block_pool import BlockHashToBlockMap

    if hasattr(BlockHashToBlockMap, "get_all_blocks"):
        return

    def get_all_blocks(self, key):
        blocks = self._cache.get(key)
        if blocks is None:
            return []
        if isinstance(blocks, dict):
            return list(blocks.values())
        return [blocks]

    BlockHashToBlockMap.get_all_blocks = get_all_blocks  # type: ignore[attr-defined]
