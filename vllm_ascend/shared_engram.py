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
from vllm_ascend.models.deepseek_v41.engram.shared_pool import SharedPoolOwner, checkpoint_identity


def load_tables(owner, model, stop):
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
        weight_key = f"layers.{layer}.engram.embed.weight"
        scale_key = f"layers.{layer}.engram.embed.scale"
        codes = owner.add(weight_key, rows, width, torch.int8)
        scales = owner.add(scale_key, rows, width // 32, torch.float32)
        with (
            safe_open(root / index[weight_key], framework="pt", device="cpu") as weights,
            safe_open(root / index[scale_key], framework="pt", device="cpu") as scale_file,
        ):
            weight = weights.get_slice(weight_key)
            scale = scale_file.get_slice(scale_key)
            for start in range(0, rows, 65536):
                if stop.is_set():
                    raise InterruptedError("Shared Engram loading cancelled")
                end = min(start + 65536, rows)
                codes.publish(start, weight[start:end])
                scales.publish(start, scale[start:end].float())
        print(json.dumps(dict(stage="LAYER_LOADED", layer=layer, rows=rows)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--pool-dir", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--store-url", help="MemFabric BM store URL; default: an available port on listen-host")
    parser.add_argument("--max-consumers", type=int, default=64, help="Maximum simultaneous consumer workers")
    parser.add_argument("--listen-host", default="127.0.0.1", help="Reachable owner address for cross-node consumers")
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    owner = SharedPoolOwner(
        args.pool_dir,
        checkpoint_identity(args.model),
        device=f"npu:{args.device}",
        host=args.listen_host,
        port=args.port,
        store_url=args.store_url,
        world_size=args.max_consumers + 1,
    )
    try:
        load_tables(owner, args.model, stop)
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
        print("POOL_CLOSED", flush=True)


if __name__ == "__main__":
    main()
