# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure request-local AscendStore v1 projection stages without Backend I/O.

Run: NUMBA_DISABLE_JIT=1 PYTHONPATH=.:../vllm python benchmarks/ascendstore_v1_projection.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import timeit

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import ChunkedTokenDatabase, KeyMetadata
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.graph.coordinates import TokenRange
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.graph.projection import (
    ContiguousBindingProjection,
    KVProjection,
    StridedBindingProjection,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.graph.reachability import GroupSelection, KVSelection
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.graph.topology import (
    KVCacheGroupTopology,
    KVTopology,
    TPPartitionSpec,
)


def measure(operation, iterations: int) -> float:
    samples = timeit.repeat(operation, number=iterations, repeat=7)
    return statistics.median(samples) / iterations * 1_000_000


def make_projection(lookup_ranks: int, *, strided: bool) -> KVProjection:
    metadata = KeyMetadata("benchmark", 0, 0, 0)
    database = ChunkedTokenDatabase([metadata], [4], None, hash_block_size=4)
    database.set_group_buffers({0: [1_000]}, {0: [32]}, {0: [32]})
    partition = TPPartitionSpec(strided, lookup_ranks, lookup_ranks if strided else 1)
    topology = KVTopology(
        0,
        lookup_ranks,
        1,
        0,
        1,
        1,
        1,
        4,
        4,
        partition,
        (KVCacheGroupTopology(0, 4, ("layer",), metadata),),
        (0,),
        None,
    )
    binding_projection = (
        StridedBindingProjection(database, topology) if strided else ContiguousBindingProjection(database)
    )
    projection = KVProjection(database, topology, binding_projection)
    projection.compile_memory_mapping()
    return projection


def benchmark_case(chunk_count: int, lookup_ranks: int, *, strided: bool) -> None:
    projection = make_projection(lookup_ranks, strided=strided)
    hashes = tuple(index.to_bytes(8, byteorder="big") for index in range(chunk_count))
    selection = KVSelection(TokenRange(0, chunk_count * 4), hashes, (GroupSelection(0, None),))
    block_ids = (tuple(range(1, chunk_count + 1)),)
    projected = projection.project_chunks(selection)
    allocated = projection.assign_local_blocks(projected, block_ids)
    iterations = max(20, 20_000 // chunk_count)
    print(
        json.dumps(
            {
                "layout": "strided" if strided else "contiguous",
                "chunks": chunk_count,
                "lookup_ranks": lookup_ranks,
                "project_us": measure(lambda: projection.project_chunks(selection), iterations),
                "lookup_expand_us": measure(lambda: projection.project_remote_objects(projected), iterations),
                "allocate_us": measure(lambda: projection.assign_local_blocks(projected, block_ids), iterations),
                "bind_us": measure(lambda: projection.bind_representations(allocated), iterations),
            }
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=int, nargs="+", default=[32, 256, 2048])
    parser.add_argument("--lookup-ranks", type=int, nargs="+", default=[1, 8])
    args = parser.parse_args()
    for chunk_count in args.chunks:
        for lookup_ranks in args.lookup_ranks:
            benchmark_case(chunk_count, lookup_ranks, strided=False)
        benchmark_case(chunk_count, 2, strided=True)


if __name__ == "__main__":
    main()
