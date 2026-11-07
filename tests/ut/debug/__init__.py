# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


def load_plan(guard, request="r", end=19, start=0, *, is_async=True):
    guard._test_load_id = getattr(guard, "_test_load_id", 0) + 1
    return {
        "pool_id": guard.pool_id,
        "load_id": guard._test_load_id,
        "request_id": request,
        "block_table": [list(refs) for refs in guard.requests[request]],
        "token_start": start,
        "token_end": end,
        "is_async": is_async,
    }


def complete_load(guard, request="r", end=19, start=0):
    meta = load_plan(guard, request, end, start)
    guard.begin_load(meta)
    guard.load(meta)
    return meta
