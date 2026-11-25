# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


DENSE_PREFIX_TOKENS = 2048
MIN_BULK_QUERY_TOKENS = 120
Q6_MAX_REQUESTS = 60
Q6_MAX_CONTEXT = 1 << 20
TOKENS_PER_REQUEST = 6
PAGE_SIZE = 128
INDEX_HEAD_PANEL_SHAPE = (28, 32, 176)
WORKSPACE_PADDING_BYTES = 16384


def score_context(max_model_len):
    # Short radix tiles contain 2048 pools. Long tiles rank 32768 pools each.
    alignment = 8192 if max_model_len <= 139264 else 131072
    return (max_model_len + alignment - 1) // alignment * alignment


def mtp_projection_rows(tokens):
    rows = (tokens + 31) // 32 * 32
    # One extra 32-row block enables a wider Cube tile for the large odd extents.
    if rows >= 192 and all(rows % tile for tile in (96, 80, 64)):
        return rows + 32
    return rows


def mtp_scratch_bytes(requests, tokens, context):
    rows = mtp_projection_rows(tokens)
    return rows * (320 << 10) + tokens * (context // 4) * 2 + requests * (context // 1024) * 4
