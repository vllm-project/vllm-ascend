# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vLLM 24420b7a341a examples/rl/rlhf_sharded_rdt_small_ep.py
"""Sharded-RDT weight sync on FOUR NPUs, one node: 2 FSDP2 trainer ranks -> 2
vLLM DP ranks with expert parallelism.

The reference example for the backend, sized to fit a CI runner:
``CheckpointNameSource`` over an FSDP2 model, one serve actor per rank, the
HTTP control plane, and assertions so it can run unattended.

Why a MoE model rather than a small dense one: the point of sharded RDT is that
each inference worker pulls only the slices it consumes, so the example is only
meaningful when no rank holds the whole model. That needs experts on both sides.
``Qwen/Qwen1.5-MoE-A2.7B`` is the reference shape, a real MoE of the same family
as the 30B model the big example uses. Its WEIGHTS are never read: the trainer
builds the config and the server runs ``--load-format dummy``, so the sync is
what is under test and nothing is downloaded but the config and tokenizer. A
smaller MoE via ``RDT_MODEL`` changes nothing about what is exercised, which is
what CI runs.

The trainer publishes CHECKPOINT names, not the names its own modules carry:
Transformers fuses each layer's experts into ``[E, ...]`` tensors
(``mlp.experts.gate_up_proj``/``down_proj``) while the checkpoint stores them per
expert, and vLLM's MoE loaders read the per-expert form. ``CheckpointNameSource``
splits them back, which is the same conversion a real trainer does when it maps
its internal layout onto checkpoint names. Publishing the fused names instead
works only for the Qwen families, whose loaders happen to accept them.

FSDP2 shards those fused tensors on dim 0 — the expert dimension — so after
sharding no rank stores every expert, and the inference side gets real EP from
``--enable-expert-parallel`` over DP2. Two simplifications an example can afford
and a real trainer cannot: each rank BUILDS the whole model before
``fully_shard`` splits it (fine for a tiny config, so nothing here streams a
checkpoint), and iteration all-gathers every parameter, so ownership stays
uniform. Serving only the experts a rank holds (``ParamMeta.owners``) is where a real
trainer earns its keep and is deliberately out of scope.

Model naming is load-bearing: names are mapped onto checkpoint keys one-to-one,
so a model whose HF module names have drifted from its checkpoint (as
GraniteMoe's router has) will not work here.

What it checks (a smoke test, not a numerical equivalence check):
  - generation CHANGES after the first sync — the weights really moved, so a
    silently-skipped sync fails instead of printing plausible text;
  - generation is UNCHANGED by a second sync — replay is stable.

Run:
    python examples/rl/rlhf_sharded_rdt_hixl_small_ep.py

Needs 4 NPUs on one node, each big enough for the whole model, since every rank
builds it before ``fully_shard`` splits it: the default is ~29 GiB in bf16, so
40 GiB cards and up. Set ``RDT_MODEL`` to a tiny MoE on smaller NPUs. Joins an
existing Ray cluster if there is one and otherwise starts its own.
Uses ND expert weights (weight_nz_mode=0) for layerwise replay. Set
RDT_RESULT_PATH to save the output comparisons and shutdown result as JSON.
"""

import json
import os
import sys
import time
from pathlib import Path

import ray
import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from ray.util.placement_group import placement_group, placement_group_table
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from rdt_hixl_vllm_serve import (
    http_generate,
    launch_vllm_serve,
    pause_generation,
    resume_generation,
    shutdown_server,
    wait_for_server,
)
from rdt_hixl_weight_source import CheckpointNameSource
from torch.distributed.fsdp import fully_shard
from transformers import AutoConfig, AutoModelForCausalLM
from vllm.distributed.weight_transfer import (
    HTTPVLLMWeightSyncClient,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.sharded_rdt_trainer import (
    ShardedRDTTrainerInitInfo,
)
from vllm.utils.network_utils import get_open_port

# Weights are built, never loaded, so this costs a config fetch -- but every rank
# materializes the model before sharding, so the default wants a 40 GiB card.
MODEL_NAME = os.environ.get("RDT_MODEL", "Qwen/Qwen1.5-MoE-A2.7B")
RAY_NAMESPACE = "sharded_rdt_hixl_small_ep_example"
VLLM_PORT = int(os.environ.get("RDT_VLLM_PORT", "8100"))
VLLM_ENDPOINT = f"http://127.0.0.1:{VLLM_PORT}"

FSDP_WORLD_SIZE = 2
INFERENCE_TP_SIZE = 1
INFERENCE_DP_SIZE = 2
NUM_INFERENCE_CONSUMERS = INFERENCE_TP_SIZE * INFERENCE_DP_SIZE

PROMPTS = ["The capital of France is", "The future of AI is"]


class FSDPTrainWorker:
    """One FSDP2 rank per NPU. Identical to the 8-NPU example's worker: the
    sharded-RDT engine owns the HIXL serve surface, so this stays a plain Ray
    actor with no producer mixin."""

    def __init__(self, model_name, rank, world_size, master_addr, master_port):
        self.rank = rank
        self.engine = None
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        assigned = ray.get_runtime_context().get_accelerator_ids()["NPU"][0]
        visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", str(assigned)).split(",")
        device_index = visible.index(str(assigned))
        torch.npu.set_device(device_index)
        print(f"[trainer] rank={rank} physical={assigned} logical={device_index}", flush=True)
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)

        # Same seed on every rank, so the shards compose into ONE model rather
        # than a blend of independent inits -- the second sync's stability check
        # is only meaningful against coherent weights.
        torch.manual_seed(0)
        config = AutoConfig.from_pretrained(model_name)
        with torch.device("npu"):
            model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
        for layer in model.model.layers:
            fully_shard(layer)
        fully_shard(model)
        self.model = model

    def get_rank(self):
        return self.rank

    def setup_engine(self, vllm_endpoint: str):
        from vllm_ascend.distributed.weight_transfer import _register_sharded_rdt

        _register_sharded_rdt()
        self.engine = WeightTransferTrainerFactory.trainer_init(
            ShardedRDTTrainerInitInfo(
                rank=self.rank,
                num_consumers=NUM_INFERENCE_CONSUMERS,
                trainer_actor_namespace=RAY_NAMESPACE,
                num_rdt_buffers=int(os.environ.get("NUM_RDT_BUFFERS", "2")),
                buffer_presize_gb=float(os.environ.get("RDT_BUFFER_PRESIZE_GB", "0")),
            ),
            client=HTTPVLLMWeightSyncClient(vllm_endpoint),
            source=CheckpointNameSource(self.model),
        )

    def sync_weights(self):
        if self.engine is None:
            raise RuntimeError("Set up the weight transfer engine before syncing")
        self.engine.send_weights()

    def close(self):
        if self.engine is not None:
            self.engine.shutdown()
        dist.destroy_process_group()


def main():
    assert torch.npu.device_count() >= 4, "Select four NPUs before starting this example"
    # Ship a minimal working_dir (this example dir) so Ray actors do NOT inherit
    # a workspace snapshot that shadows the editable vLLM install.
    runtime_env: dict[str, object] = {
        "py_executable": sys.executable,
        "working_dir": os.path.dirname(os.path.abspath(__file__)),
    }
    forwarded = {
        k: os.environ[k]
        for k in (
            "HCCL_CONNECT_TIMEOUT",
            "HCCL_EXEC_TIMEOUT",
            "LD_PRELOAD",
            "LD_LIBRARY_PATH",
        )
        if k in os.environ
    }
    if forwarded:
        runtime_env["env_vars"] = forwarded
    if not ray.is_initialized():
        try:
            ray.init(address="auto", runtime_env=runtime_env, namespace=RAY_NAMESPACE)
        except (ConnectionError, RuntimeError):
            # No cluster to join (a bare CI runner): start one on this node.
            # `launch_vllm_serve` hands the child this cluster's address, which
            # it needs -- vLLM's ray DP backend would otherwise start its own.
            ray.init(
                address="local",
                resources={"NPU": 4},
                num_cpus=12,
                object_store_memory=512 << 20,
                runtime_env=runtime_env,
                namespace=RAY_NAMESPACE,
            )

    assert ray.available_resources().get("NPU", 0) >= 4, "Four idle NPUs required"
    print(f"[init] model {MODEL_NAME}, {FSDP_WORLD_SIZE} trainer + {INFERENCE_DP_SIZE} inference NPUs")

    # Reserve the trainer's NPUs BEFORE launching the server, so the server's DP
    # workers take the node's remaining two. STRICT_PACK keeps the trainer's
    # all-gathers on one node's HCCL -- here that is the only node anyway.
    trainer_pg = placement_group([{"NPU": 1, "CPU": 1}] * FSDP_WORLD_SIZE, strategy="STRICT_PACK")
    ray.get(trainer_pg.ready())
    pg_node_id = next(iter(placement_group_table(trainer_pg)["bundles_to_node_id"].values()))
    master_addr = next(n["NodeManagerAddress"] for n in ray.nodes() if n["NodeID"] == pg_node_id)

    @ray.remote(
        num_cpus=0,
        scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=trainer_pg),
    )
    def _free_port():
        return get_open_port()

    master_port = ray.get(_free_port.remote())

    worker_cls = ray.remote(resources={"NPU": 1})(FSDPTrainWorker)
    workers = [
        worker_cls.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=trainer_pg, placement_group_bundle_index=rank
            ),
        ).remote(MODEL_NAME, rank, FSDP_WORLD_SIZE, master_addr, master_port)
        for rank in range(FSDP_WORLD_SIZE)
    ]
    ray.get([w.get_rank.remote() for w in workers])
    print(f"[init] {FSDP_WORLD_SIZE} FSDP trainer workers ready (seeded weights).")

    server = launch_vllm_serve(
        MODEL_NAME,
        tensor_parallel_size=INFERENCE_TP_SIZE,
        data_parallel_size=INFERENCE_DP_SIZE,
        enable_expert_parallel=True,
        port=VLLM_PORT,
        extra_args=[
            "--max-model-len",
            "128",
            "--max-num-seqs",
            "8",
            "--max-num-batched-tokens",
            "128",
            "--kv-cache-memory-bytes",
            str(256 << 20),
            "--all2all-backend",
            "allgather_reducescatter",
            "--additional-config",
            json.dumps({"weight_nz_mode": 0}),
        ],
    )
    try:
        wait_for_server(VLLM_ENDPOINT, server)

        before = http_generate(VLLM_ENDPOINT, MODEL_NAME, PROMPTS)
        print("[generate] BEFORE sync (vLLM's own dummy init):")
        for p, t in zip(PROMPTS, before):
            print(f"  {p!r} -> {t!r}")

        print("[sync] building trainer engines (bake on the sender)...")
        t0 = time.perf_counter()
        ray.get([w.setup_engine.remote(VLLM_ENDPOINT) for w in workers])
        print(f"[sync] engine setup (incl. bake) took {time.perf_counter() - t0:.2f}s")

        pause_generation(VLLM_ENDPOINT)
        t0 = time.perf_counter()
        ray.get([w.sync_weights.remote() for w in workers])
        print(f"[sync] iter 0 took {time.perf_counter() - t0:.3f}s")
        resume_generation(VLLM_ENDPOINT)

        after = http_generate(VLLM_ENDPOINT, MODEL_NAME, PROMPTS)
        print("[generate] AFTER sync (the trainer's weights):")
        for p, t in zip(PROMPTS, after):
            print(f"  {p!r} -> {t!r}")

        # Sync again; replay must be stable, so generation must not move.
        pause_generation(VLLM_ENDPOINT)
        t0 = time.perf_counter()
        ray.get([w.sync_weights.remote() for w in workers])
        print(f"[sync] iter 1 took {time.perf_counter() - t0:.3f}s")
        resume_generation(VLLM_ENDPOINT)
        again = http_generate(VLLM_ENDPOINT, MODEL_NAME, PROMPTS)

        # Greedy decoding, so both comparisons are exact.
        assert after != before, (
            f"generation did not change across the weight sync, so the sync moved nothing: {before!r}"
        )
        assert again == after, f"a second sync changed generation, so replay is not stable: {after!r} then {again!r}"
        print("[ok] weights moved on sync 0 and replay was stable on sync 1")
        # Disconnect consumers before deregistering the trainer's source rings.
        HTTPVLLMWeightSyncClient(VLLM_ENDPOINT).shutdown_weight_transfer_engine()
        ray.get([w.close.remote() for w in workers])
        print("[shutdown] consumers and producers released", flush=True)

        if result_path := os.environ.get("RDT_RESULT_PATH"):
            Path(result_path).write_text(
                json.dumps(
                    {
                        "before": before,
                        "after": after,
                        "again": again,
                        "changed": after != before,
                        "stable": again == after,
                        "shutdown": True,
                    },
                    indent=2,
                )
            )
    finally:
        shutdown_server(server)
        ray.shutdown()


if __name__ == "__main__":
    try:
        main()
    finally:
        ray.shutdown()
