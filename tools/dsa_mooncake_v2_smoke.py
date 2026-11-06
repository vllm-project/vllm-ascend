# SPDX-License-Identifier: Apache-2.0
"""Two-device byte check for MooncakeConnectorV2 -> DSA Host KV landing.

Run producer and consumer on separate idle NPU devices. This exercises V2's
native TP transfer bucket and the shared staging reader, without model weights.
It does not establish model correctness or end-to-end performance.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch_npu  # noqa: F401
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import (
    MooncakePCPTransferMetadata,
    MooncakePPTransferMetadata,
    MooncakeTPTransferMetadata,
    MooncakeTransferMetadata,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker import MooncakePullRecvingThread
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_te
from vllm_ascend.kv_offload.dsa_pd_mooncake_staging import DSAPDMooncakeStagingReader
from vllm_ascend.worker.v2.attn_utils import (
    _allocate_dsa_pd_mooncake_kv_tensor,
    _register_dsa_pd_mooncake_final_kv_views,
)


def pattern(size: int, offset: int = 0) -> torch.Tensor:
    return ((torch.arange(size, dtype=torch.int32) + offset) % 127).to(torch.int8)


def producer(args: argparse.Namespace, engine: Any) -> None:
    source = pattern(args.nbytes + 2 * 1024 * 1024).to("npu")
    torch.npu.synchronize()
    global_te.register_buffer([source.data_ptr()], [source.nbytes])
    args.metadata.write_text(
        json.dumps(
            {
                "session": f"{args.hostname}:{engine.get_rpc_port()}",
                "ptr": source.data_ptr(),
                "nbytes": source.nbytes,
            }
        )
    )
    deadline = time.monotonic() + args.timeout
    while not args.done.exists():
        if time.monotonic() > deadline:
            raise TimeoutError("Consumer did not finish before producer deadline")
        time.sleep(0.2)
    print(json.dumps({"role": "producer", "ok": True}), flush=True)


def consumer(args: argparse.Namespace, engine: Any) -> None:
    remote = json.loads(args.metadata.read_text())
    block = 256
    if args.nbytes % block:
        raise ValueError("nbytes must be a multiple of 256")
    host = _allocate_dsa_pd_mooncake_kv_tensor(args.nbytes, 2 * 1024 * 1024)
    _register_dsa_pd_mooncake_final_kv_views({"host": host})
    hbm = torch.zeros(128 * 1024, dtype=torch.int8, device="npu")
    global_te.register_buffer([hbm.data_ptr()], [hbm.nbytes])
    reader = DSAPDMooncakeStagingReader(engine, {"host": host, "hbm": hbm}, [[block], [block]])
    thread = MooncakePullRecvingThread.__new__(MooncakePullRecvingThread)
    thread.engine, thread.staging_reader = engine, reader
    thread.tp_rank = thread.pcp_rank = thread.dcp_rank = 0
    thread.tp_size = thread.pcp_size = thread.dcp_size = 1
    thread.num_key_value_heads = 1
    thread.num_speculative_tokens = 0
    thread.layer_names = ["model.layers.0.self_attn", "model.layers.1.self_attn"]
    thread.kv_cache_specs = [FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float16)]
    thread.kv_caches_base_addr = [[host.data_ptr()], [hbm.data_ptr()]]
    thread.block_strides = thread.block_lens = [[block], [block]]
    thread.block_shapes = [[(1, 16, 8)], [(1, 16, 8)]]
    thread.block_size_scales = [[1], [1]]
    thread.local_metadata = MooncakeTransferMetadata(
        engine_id="smoke-consumer",
        te_rpc_port=engine.get_rpc_port(),
        block_size=16,
        num_blocks=args.nbytes // block,
        layer_names=thread.layer_names,
        layer_block_sizes=[16, 16],
        group_indices=[0, 0],
        kv_caches_base_addr=thread.kv_caches_base_addr,
        block_strides=thread.block_strides,
        block_lens=thread.block_lens,
        block_shapes=thread.block_shapes,
        block_size_scales=thread.block_size_scales,
        local_ip=args.hostname,
    )
    ip, port = remote["session"].rsplit(":", 1)
    pp = MooncakePPTransferMetadata(
        block_size=16,
        num_blocks=remote["nbytes"] // block,
        layer_names=thread.layer_names,
        layer_block_sizes=[16, 16],
        group_indices=[0, 0],
        block_strides=thread.block_strides,
        block_lens=thread.block_lens,
        block_shapes=thread.block_shapes,
        block_size_scales=thread.block_size_scales,
        metadata_by_pcp_rank={
            0: MooncakePCPTransferMetadata(
                metadata_by_tp_rank={
                    0: MooncakeTPTransferMetadata(
                        te_rpc_port=int(port),
                        layer_indices=[0, 1],
                        kv_caches_base_addr=[[remote["ptr"]], [remote["ptr"]]],
                        local_ip=ip,
                        handshake_port=0,
                    ),
                }
            )
        },
    )
    # Different request IDs allow the layers to use different block mappings.
    # Non-zero source offsets exercise ADXL's page-offset constraint.
    source_offset = 128 * 1024
    host_blocks = list(range(args.nbytes // block))
    hbm_blocks = list(range(hbm.nbytes // block))
    entries = {
        0: {
            (0, 0): [("host", host_blocks, [b + source_offset // block for b in host_blocks])],
            (1, 1): [("hbm", hbm_blocks, [b + 2 for b in hbm_blocks])],
        }
    }
    started = time.perf_counter()
    thread._execute_tp_transfer_bucket(0, 0, 0, 1, 1, 1, pp, entries)
    elapsed = time.perf_counter() - started
    scratch = torch.empty_like(host, device="npu")
    scratch.copy_(host, non_blocking=False)
    torch.npu.synchronize()
    host_ok = torch.equal(scratch.cpu(), pattern(args.nbytes, source_offset))
    hbm_ok = torch.equal(hbm.cpu(), pattern(hbm.nbytes, 2 * block))
    result = {
        "role": "consumer",
        "connector": "MooncakeConnectorV2",
        "allocator": "ModelRunnerV2 DSA Host KV",
        "host_ok": host_ok,
        "hbm_ok": hbm_ok,
        "host_bytes": host.nbytes,
        "hbm_bytes": hbm.nbytes,
        "staging_bytes": reader._dsa_pd_mooncake_staging_bytes,
        "pipeline": reader._pipeline_landing_enabled,
        "banks": reader._pipeline_banks,
        "transfer_seconds": elapsed,
    }
    print(json.dumps(result), flush=True)
    if not host_ok or not hbm_ok:
        raise RuntimeError("V2 Host/HBM byte verification failed")
    args.done.touch()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=["producer", "consumer"], required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--hostname", required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--done", type=Path, required=True)
    parser.add_argument("--nbytes", type=int, default=9 * 1024 * 1024 + 128 * 1024)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    engine = global_te.get_transfer_engine(args.hostname, None)
    if args.role == "producer":
        producer(args, engine)
    else:
        consumer(args, engine)
    # The native engine owns background threads; process exit releases devices.
    os._exit(0)


if __name__ == "__main__":
    main()
