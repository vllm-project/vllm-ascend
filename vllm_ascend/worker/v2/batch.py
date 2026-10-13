# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.


def sort_batch_req_ids(
    num_tokens_per_req: dict[str, int],
    draft_tokens: dict[str, list[int]],
    decode_query_len: int,
) -> list[str]:
    """Keep decode first without changing the relative order of prefills.

    FIA's floating-point output can depend on the packed prefill offsets.
    MRV1 preserves the scheduled prefill order, so MRV2 must preserve it too.
    Verification and short-extend ordering retain the upstream priorities.
    """

    def key(req_id: str) -> tuple[bool, bool, int]:
        num_tokens = num_tokens_per_req[req_id]
        has_drafts = bool(draft_tokens.get(req_id))
        priority = num_tokens if has_drafts or num_tokens <= decode_query_len else decode_query_len + 1
        return not has_drafts, num_tokens != decode_query_len, priority

    return sorted(num_tokens_per_req, key=key)
