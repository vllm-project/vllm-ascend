# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Prefix reachability over shared object descriptors, independent of I/O."""

import json
from bisect import bisect_left, bisect_right

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.state_snapshot import MAMBA_FORMAT_PREFIX


def logical_cache_range(cache, spec):
    compression = getattr(spec, "tokens_per_state", 1)
    if cache.format_id.startswith("mla-page-v2:"):
        rows = json.loads(cache.format_id.split(":", 1)[1])["rows"]
        return cache.token_start * rows * compression, cache.token_end * rows * compression
    if cache.format_id.startswith(MAMBA_FORMAT_PREFIX) or '"family":"circular-v2"' in cache.format_id:
        return cache.token_start, cache.token_end
    return cache.token_start * compression, cache.token_end * compression


def needed_prefix_objects(refs, specs, start, end):
    selected = []
    for ref in refs:
        for part in ref.slices:
            cache = part.cache
            if cache.format_id.startswith(MAMBA_FORMAT_PREFIX):
                needed = cache.token_end == end
            else:
                left, right = logical_cache_range(cache, specs[cache.layer])
                window = getattr(specs[cache.layer], "sliding_window", None)
                cutoff = max(start, end - window) if isinstance(window, int) else start
                needed = left < end and right > cutoff
            if needed:
                selected.append(ref)
                break
    return tuple(selected)


def readable_prefix_ends(refs, readable, specs, *, start, end, granularity):
    """Intersect missing-object intervals without quadratic prefix rescans.

    Ordinary attention requires all reachable earlier pages. A window expires
    old requirements; aligned recurrent state is needed only at its checkpoint.
    Recurrent hits are therefore not assumed to be monotonic.
    """
    if len(refs) != len(readable) or any(type(hit) is not bool for hit in readable):
        raise ValueError("Prefix readability response is not aligned")
    candidates = list(range((start // granularity + 1) * granularity, end + 1, granularity))
    blocked = [0] * (len(candidates) + 1)
    for ref, hit in zip(refs, readable, strict=True):
        if hit:
            continue
        for part in ref.slices:
            cache = part.cache
            if cache.format_id.startswith(MAMBA_FORMAT_PREFIX):
                first, stop = bisect_left(candidates, cache.token_end), bisect_right(candidates, cache.token_end)
            else:
                left, right = logical_cache_range(cache, specs[cache.layer])
                if right <= start:
                    continue
                window = getattr(specs[cache.layer], "sliding_window", None)
                first = bisect_right(candidates, left)
                stop = bisect_left(candidates, right + window) if isinstance(window, int) else len(candidates)
            blocked[first] += 1
            blocked[stop] -= 1
    result, count = [], 0
    for index, boundary in enumerate(candidates):
        count += blocked[index]
        if count == 0:
            result.append(boundary)
    return result


def probe_pool_prefix(layout, backend, profile, worker_id, query):
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pool_metadata import PoolSaveRequest, PoolWorkerReport

    if not isinstance(query, dict) or query.keys() != {
        "op",
        "incarnation",
        "mode",
        "request",
        "group_ids",
        "granularity",
    }:
        raise ValueError("Invalid prefix query fields")
    if query["op"] != "prefix" or query["incarnation"] != profile.engine_incarnation:
        raise ValueError("Prefix query belongs to another worker incarnation")
    data = query["request"]
    expected = {"transfer_id", "attempt_id", "request_digest", "token_start", "token_end", "group_hashes"}
    if not isinstance(data, dict) or data.keys() != expected:
        raise ValueError("Invalid prefix request fields")
    groups = query["group_ids"]
    count = len(layout.cache_config.kv_cache_groups)
    if (
        not isinstance(groups, list)
        or not groups
        or any(type(group) is not int or not 0 <= group < count for group in groups)
        or len(set(groups)) != len(groups)
    ):
        raise ValueError("Invalid prefix cache groups")
    step = query["granularity"]
    if type(step) is not int or step <= 0 or query["mode"] not in ("probe", "manifest"):
        raise ValueError("Invalid prefix lookup mode or granularity")
    request = PoolSaveRequest(
        **(
            data
            | {
                "block_ids": tuple(() for _ in range(count)),
                "group_hashes": tuple(tuple(hashes) for hashes in data["group_hashes"]),
            }
        )
    )
    refs = layout.prefix_objects(request, profile, frozenset(groups), history=query["mode"] == "probe")
    if query["mode"] == "manifest":
        refs = needed_prefix_objects(refs, layout.specs, request.token_start, request.token_end)
    keys = [ref.resolve(profile.key_identity)[0].key for ref in refs]
    backend.set_device()
    readable = backend.batch_is_readable(keys) if keys else []
    if len(readable) != len(refs) or any(type(hit) is not bool for hit in readable):
        raise ValueError("Prefix backend returned invalid readability")
    if query["mode"] == "probe":
        return {
            "worker_id": worker_id,
            "ready_ends": readable_prefix_ends(
                refs,
                readable,
                layout.specs,
                start=request.token_start,
                end=request.token_end,
                granularity=step,
            ),
        }
    ready = all(readable)
    return PoolWorkerReport(
        request.transfer_id,
        request.attempt_id,
        worker_id,
        ready,
        refs if ready else (),
        "" if ready else "Prefix objects were evicted",
    ).to_wire()


def lookup_pool_prefix(workers, data, group_ids, step, executor, control_request):
    """Probe all source workers and recheck the chosen committed boundary."""
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pool_metadata import PoolTransferMetadata, PoolWorkerReport

    start, end = data["token_start"], data["token_end"]

    def query(worker, mode):
        return control_request(
            worker.control_endpoint,
            {
                "op": "prefix",
                "incarnation": worker.profile.engine_incarnation,
                "mode": mode,
                "request": dict(data),
                "group_ids": list(group_ids),
                "granularity": step,
            },
        )

    workers = list(workers)
    if not workers:
        return None
    futures = [executor.submit(query, worker, "probe") for worker in workers]
    common = set(range((start // step + 1) * step, end + 1, step))
    for worker, future in zip(workers, futures, strict=True):
        response = future.result()
        if (
            not isinstance(response, dict)
            or response.keys() != {"worker_id", "ready_ends"}
            or response["worker_id"] != worker.worker_id
        ):
            raise ValueError("Prefix response has a mismatched worker")
        values = response["ready_ends"]
        if not isinstance(values, list) or any(
            type(value) is not int or value <= start or value > end or value % step for value in values
        ):
            raise ValueError("Invalid prefix checkpoint candidates")
        common.intersection_update(values)
    if not common:
        return None
    data["token_end"] = max(common)
    futures = [executor.submit(query, worker, "manifest") for worker in workers]
    objects = []
    for worker, future in zip(workers, futures, strict=True):
        report = PoolWorkerReport.from_wire(future.result())
        if (report.worker_id, report.transfer_id, report.attempt_id) != (
            worker.worker_id,
            data["transfer_id"],
            data["attempt_id"],
        ):
            raise ValueError("Prefix manifest identity mismatch")
        if not report.committed:
            return None  # An eviction between probe and manifest is a miss.
        if any(ref.profile_id != worker.profile.profile_id or ref.group_id not in group_ids for ref in report.objects):
            raise ValueError("Prefix manifest contains another source layout or cache group")
        objects.extend(report.objects)
    return PoolTransferMetadata(
        data["transfer_id"],
        data["attempt_id"],
        data["request_digest"],
        start,
        data["token_end"],
        "prefix",
        tuple(worker.profile for worker in workers),
        tuple(objects),
    )
