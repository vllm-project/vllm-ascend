# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from argparse import Namespace

from vllm.entrypoints.launchers import dp_supervisor


def _build_device_ids(args: Namespace, local_rank: int) -> list[int | str]:
    """Give each DP child its TP × PP × PCP slice of the visible devices."""
    # v0.34 callers can omit PCP when using its default.
    pcp_size = getattr(args, "prefill_context_parallel_size", 1)
    devices_per_rank = args.tensor_parallel_size * args.pipeline_parallel_size * pcp_size
    start = local_rank * devices_per_rank
    stop = start + devices_per_rank
    device_ids = getattr(args, "device_ids", None)
    if device_ids is not None:
        if stop > len(device_ids):
            raise ValueError(
                f"--device-ids has {len(device_ids)} entries, but DP rank {local_rank} needs devices [{start}, {stop})"
            )
        return device_ids[start:stop]
    return list(range(start, stop))


# The release and verified-main pins advance independently. Only replace a
# supervisor that gives a two-rank PCP group a single device (vLLM #69664).
if dp_supervisor._build_device_ids(
    Namespace(tensor_parallel_size=1, pipeline_parallel_size=1, prefill_context_parallel_size=2, device_ids=None),
    local_rank=0,
) == [0]:
    dp_supervisor._build_device_ids = _build_device_ids
