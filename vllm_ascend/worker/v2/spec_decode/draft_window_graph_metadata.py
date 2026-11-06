# SPDX-License-Identifier: Apache-2.0
"""Private Draft metadata copies and the shared Window read-field binding."""

from __future__ import annotations

from collections.abc import Mapping
from copy import copy
from typing import Any


def copy_draft_attention_metadata(metadata: Mapping[str, Any], layers: tuple[str, ...]) -> dict[str, Any]:
    """Copy the admitted single Draft group's header, keeping its layer aliases.

    The real builder publishes one header for the already inventoried group.
    Both callers replace read fields; inherited Full/query operands stay read-only.
    """
    layer = copy(metadata[layers[0]])
    return dict.fromkeys(layers, layer)


def bind_draft_window_read(layer: Any, bank: Any) -> None:
    """FIA1 reads Host exact lengths; keep builder-owned Full GPU/shape fields."""
    layer.block_tables = bank.window_block_table
    layer.seq_lens_cpu = bank.cpu_window_exact
    layer.seq_lens_list = bank.cpu_window_exact.tolist()
