# SPDX-License-Identifier: Apache-2.0
"""Persistent shared Engram owner: python -m vllm_ascend.shared_engram."""

import argparse
import json
import signal
import threading
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from safetensors import safe_open

# Initialize the normal operator registry before Engram's DeviceOperator import.
import vllm_ascend.ops  # noqa: F401
from vllm_ascend.engram_shard_layout import layout_identity, owner_set, row_ranges
from vllm_ascend.models.deepseek_v41.engram.shared_host import EngramReleaseError
from vllm_ascend.models.deepseek_v41.engram.shared_pool import SharedPoolOwner, checkpoint_identity


def load_tables(owner, model, stop, *, membership=None, owner_rank=None):
    root = Path(model)
    config = json.loads((root / "config.json").read_text())["text_config"]
    index = json.loads((root / "quant_model_weights.safetensors.index.json").read_text())["weight_map"]
    width = config["engram_head_dim"]
    if width != 256:
        raise ValueError("Shared Engram currently supports width256/group32 INT8 tables")
    shapes = list(zip(config["engram_layer_ids"], config["engram_num_embeddings"], strict=True))
    # Reject incompatible checkpoint layouts before the first physical allocation.
    for layer, rows in shapes:
        for suffix, columns, allowed in [("weight", width, {"I8", "INT8"}), ("scale", width // 32, {"F32", "F8_E8M0"})]:
            key = f"layers.{layer}.engram.embed.{suffix}"
            with safe_open(root / index[key], framework="pt", device="cpu") as file:
                source = file.get_slice(key)
                if source.get_shape() != [rows, columns] or source.get_dtype() not in allowed:
                    raise ValueError(f"Unsupported shared Engram checkpoint layout: {key}")
    for layer, rows in shapes:
        start_row, end_row = (
            (0, rows)
            if membership is None
            else row_ranges(rows, len(membership["owners"]), membership["chunk_rows"])[owner_rank]
        )
        local_rows = end_row - start_row
        shard = {} if membership is None else dict(global_rows=rows, row_start=start_row)
        weight_key = f"layers.{layer}.engram.embed.weight"
        scale_key = f"layers.{layer}.engram.embed.scale"
        codes = owner.add(weight_key, local_rows, width, torch.int8, **shard)
        scales = owner.add(scale_key, local_rows, width // 32, torch.float32, **shard)
        with (
            safe_open(root / index[weight_key], framework="pt", device="cpu") as weights,
            safe_open(root / index[scale_key], framework="pt", device="cpu") as scale_file,
        ):
            weight = weights.get_slice(weight_key)
            scale = scale_file.get_slice(scale_key)
            for start in range(start_row, end_row, 65536):
                if stop.is_set():
                    raise InterruptedError("Shared Engram loading cancelled")
                end = min(start + 65536, end_row)
                codes.publish(start - start_row, weight[start:end])
                scales.publish(start - start_row, scale[start:end].float())
        print(json.dumps(dict(stage="LAYER_LOADED", layer=layer, rows=rows)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--pool-dir", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--library-path")
    parser.add_argument("--owner-set", help="Static ordered owner membership JSON")
    parser.add_argument("--owner-rank", type=int)
    parser.add_argument("--listen-host", default="127.0.0.1", help="Reachable owner address for cross-node consumers")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--network-bootstrap", action="store_true", help="Allow bootstrap over the trusted network")
    args = parser.parse_args()
    membership = owner_set(args.owner_set) if args.owner_set else None
    if (membership is None) != (args.owner_rank is None):
        parser.error("--owner-set and --owner-rank must be provided together")
    if membership is not None and not 0 <= args.owner_rank < len(membership["owners"]):
        parser.error("Owner rank is outside membership")
    torch.npu.set_device(args.device)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    owner = SharedPoolOwner(
        args.pool_dir,
        checkpoint_identity(args.model),
        device=f"npu:{args.device}",
        library_path=args.library_path,
        host=args.listen_host,
        port=args.port,
        network_bootstrap=args.network_bootstrap,
    )
    try:
        if membership is not None:
            owner.manifest["sharding"] = dict(
                layout_id=layout_identity(membership, owner.manifest["identity"]),
                owner_rank=args.owner_rank,
                membership=membership,
            )
        load_tables(owner, args.model, stop, membership=membership, owner_rank=args.owner_rank)
        owner.ready()
        print(json.dumps(dict(stage="POOL_READY", root=args.pool_dir, identity=owner.manifest["identity"])), flush=True)
        stop.wait()
    finally:
        # Do not free memory while consumers still hold mappings. SIGKILL is
        # deliberately not a graceful-stop mechanism; use a new generation after it.
        while not owner.closed:
            try:
                owner.close(timeout=5)
            except TimeoutError:
                print("POOL_DRAINING waiting for consumers", flush=True)
            except EngramReleaseError as exc:
                print(f"POOL_DRAINING release failed, retrying: {exc}", flush=True)
                threading.Event().wait(1)
        print("POOL_CLOSED", flush=True)


if __name__ == "__main__":
    main()
