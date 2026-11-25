# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

MAX_RESIDENT_REQUESTS = 4
MAX_PACKED_OUTPUT_REQUESTS = 16
DECODE_CONTEXT_BUCKETS = (16384, 65536, 131072, 524288)
OUTPUT_PANEL_CHANNELS = 1024
BASE_WORKSPACE_PER_ROW = 224 << 10
QUERY_ROW_BYTES = 32768 * 2
WORKSPACE_PADDING_BYTES = 16384


# Segment flags follow the four-page run descriptors in the index score workspace.
FRAGMENTED_INDEX_SEGMENT = -1
SHARED_INDEX_SEGMENT = 2
LINKED_PAIR_LEADER = 3

# Pack the signed scheduling score above the shared count (at most 1024 rows times eight segments).
INDEX_VOTE_STRIDE = 1 << 14
INDEX_COVERAGE_SCALE = 4
SHARED_INDEX_WEIGHT = 4


def use_segmented_index_scan(batch, context):
    return batch >= 128 and context == 131072


def use_four_request_index_scan(batch, context):
    return batch >= 512 and use_segmented_index_scan(batch, context)


def head_panel_layout(batch):
    """Native FP32 projection K partition and its DMA-aligned panel width."""
    width = 147 if batch <= 128 else 293 if batch <= 448 else 4096
    return (4096 + width - 1) // width, width, (width + 31) // 32 * 32


def projection_rows(batch, prepacked_heads=True):
    """Choose a dense tile height that avoids rereading the full projection weights."""
    rows = (batch + 31) // 32 * 32
    if prepacked_heads and rows in (160, 224):
        return (rows + 63) // 64 * 64
    if prepacked_heads and rows > 256:
        tile = next(size for size in (128, 96, 64, 32) if rows % size == 0)
        panels = (rows + 127) // 128
        if panels < rows // tile:
            return panels * 128
    return rows


def query_workspace_rows(batch, rows, resident_workspace):
    if resident_workspace and 1 < batch <= MAX_RESIDENT_REQUESTS:
        # One FP32 [64 heads, 16 tiles, 512 channels] partial per request.
        return max(rows, batch * 32)
    return rows


def decode_context_bucket(limit):
    return next((bucket for bucket in DECODE_CONTEXT_BUCKETS if 0 < limit <= bucket), None)


def latent_workspace_elements(batch, rows, context):
    return max(rows * (QUERY_ROW_BYTES // 2), batch * (context // 4))


def resident_workspace_bytes(batch, context=131072):
    rows = projection_rows(batch)
    resident_extra = (query_workspace_rows(batch, rows, True) - rows) * QUERY_ROW_BYTES
    score_extra = latent_workspace_elements(batch, rows, context) * 2 - rows * QUERY_ROW_BYTES
    return rows * BASE_WORKSPACE_PER_ROW + resident_extra + score_extra
