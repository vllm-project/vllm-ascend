# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Isolated PCP hidden-state gather / local Q overlap experiment.

Uses production DSA projection and PCP gather/restore methods, with synthetic
weights. Does not patch a model runner or change serving defaults. Requires eight
idle Ascend NPUs and the serving image's model configuration/quant description.
See benchmarks/scripts/run_pcp_q_overlap.sh for the environment and invocation.
"""

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch_npu
from vllm.config import set_current_vllm_config
from vllm.distributed import (
    ensure_model_parallel_initialized,
    get_pcp_group,
    get_world_group,
    init_distributed_environment,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear

from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.attention.context_parallel.dsa_cp import AscendDSAPCPImpl
from vllm_ascend.distributed.parallel_state import init_ascend_model_parallel
from vllm_ascend.utils import adapt_patch, enable_custom_op, register_ascend_customop

WORLD_SIZE = 8
WEIGHT_SEED = 20261107
WARMUP_STEPS = 8
REPLAY_INPUT_SEEDS = (71, 83, 97)
PROFILE_STEPS = 20
ATOL = 0.02
RTOL = 0.02
RELATIVE_L2_LIMIT = 0.005


class PCPQueryExperiment:
    """Keep collectives on the caller stream; move only independent Q work.

    The auxiliary stream is owned by this experiment, not module-global state.
    Every invocation joins it before returning; concurrent calls on the same
    instance are not supported. No cross-invocation tensor cache is introduced.
    """

    def __init__(self, impl):
        self.impl = impl
        self.query_stream = torch.npu.Stream()

    def query(self, hidden, cos, sin, actual_tokens):
        if actual_tokens == 0:
            return ()
        return self.impl._mla_prolog_single_stream(
            hidden[:actual_tokens], cos[:actual_tokens], sin[:actual_tokens], None, None, write_swa_cache=False
        )

    def forward(self, hidden, metadata, cos, sin, actual_tokens, overlap):
        if not overlap or actual_tokens == 0:
            restored = self.impl._gather_and_restore_hidden_states(hidden, metadata)
            query = self.query(hidden, cos, sin, actual_tokens)
            return (restored, *query)
        caller = torch.npu.current_stream()
        # Inputs (including RoPE) may have been updated immediately before replay.
        self.query_stream.wait_stream(caller)
        with torch.npu.stream(self.query_stream):
            query = self.query(hidden, cos, sin, actual_tokens)
        restored = self.impl._gather_and_restore_hidden_states(hidden, metadata)
        # The gather output and all Q outputs are ready for downstream consumers.
        caller.wait_stream(self.query_stream)
        return (restored, *query)


def build_projection(config):
    hf = config.model_config.hf_config
    impl = AscendDSAPCPImpl.__new__(AscendDSAPCPImpl)
    impl.n_local_heads = hf.num_attention_heads
    impl.head_dim = hf.head_dim
    impl.nope_head_dim = hf.head_dim - hf.qk_rope_head_dim
    impl.eps = hf.rms_norm_eps
    prefix = "model.layers.3.self_attn"
    with torch.device("npu"):
        impl.wq_a = ReplicatedLinear(
            hf.hidden_size,
            hf.q_lora_rank,
            bias=False,
            quant_config=config.quant_config,
            prefix=f"{prefix}.wq_a",
            return_bias=False,
        )
        impl.wq_b = ColumnParallelLinear(
            hf.q_lora_rank,
            hf.num_attention_heads * hf.head_dim,
            bias=False,
            quant_config=config.quant_config,
            prefix=f"{prefix}.wq_b",
            return_bias=False,
        )
        impl.wkv = ReplicatedLinear(
            hf.hidden_size,
            hf.head_dim,
            bias=False,
            quant_config=config.quant_config,
            prefix=f"{prefix}.wkv",
            return_bias=False,
        )
        impl.q_norm = RMSNorm(hf.q_lora_rank, eps=hf.rms_norm_eps)
        impl.q_norm_without_weight = RMSNorm(hf.head_dim, eps=hf.rms_norm_eps, has_weight=False)
    for name in ("wq_a", "wq_b", "wkv", "q_norm", "q_norm_without_weight"):
        module = getattr(impl, name)
        for param_name, param in module.named_parameters():
            seed = WEIGHT_SEED + int.from_bytes(hashlib.sha256(f"{name}.{param_name}".encode()).digest()[:4], "little")
            torch.manual_seed(seed)
            torch.npu.manual_seed_all(seed)
            if param.dtype == torch.int8:
                param.random_(-32, 33)
            elif "offset" in param_name:
                param.zero_()
            elif "scale" in param_name:
                param.fill_(1 / (19 * hf.q_lora_rank**0.5))
            elif "norm" in name:
                param.fill_(1)
            else:
                param.normal_(0, hf.hidden_size**-0.5)
        method = getattr(module, "quant_method", None)
        if method is not None:
            method.process_weights_after_loading(module)
    assert impl.wq_a.weight.dtype == torch.bfloat16
    assert impl.wq_b.weight.dtype == torch.int8
    return impl


def synchronize_ranks():
    torch.npu.synchronize()
    dist.barrier(group=get_world_group().cpu_group)


def collect(value):
    values = [None] * WORLD_SIZE
    dist.all_gather_object(values, value, group=get_world_group().cpu_group)
    return values


def compare(actual, expected):
    stats = []
    for got, ref in zip(actual, expected, strict=True):
        if got is None or ref is None:
            assert got is ref
            continue
        got = got.detach().cpu()
        ref = ref.detach().cpu()
        if got.numel() == 0:
            assert got.shape == ref.shape
            continue
        diff = (got.float() - ref.float()).abs()
        relative = float(diff.norm() / ref.float().norm().clamp_min(1e-12))
        equal = torch.equal(got, ref)
        passed = (
            equal
            if not got.is_floating_point()
            else (torch.allclose(got, ref, atol=ATOL, rtol=RTOL) and relative < RELATIVE_L2_LIMIT)
        )
        stats.append({"max_abs": float(diff.max()), "relative_l2": relative, "equal": equal, "passed": passed})
    assert all(x["passed"] for x in stats), stats
    return stats


def make_case(tokens, kind, hidden_size, rope_dim, rank):
    if kind == "uneven":
        lengths = [max(1, tokens - 3 * r) for r in range(WORLD_SIZE)]
    elif kind == "empty_rank":
        lengths = [0] + [tokens] * (WORLD_SIZE - 1)
    elif kind == "all_empty":
        lengths = [0] * WORLD_SIZE
    else:
        lengths = [tokens] * WORLD_SIZE
    valid = torch.tensor([r * tokens + t for r, n in enumerate(lengths) for t in range(n)], dtype=torch.int64)
    # Nontrivial scheduler-order restoration, including omission of padding.
    generator = torch.Generator().manual_seed(123)
    indices = valid[torch.randperm(len(valid), generator=generator)]
    hidden = torch.empty((tokens, hidden_size * (2 if kind == "strided" else 1)), device="npu")
    if kind == "strided":
        hidden = hidden[:, ::2]
    phase = torch.arange(tokens * rope_dim, dtype=torch.float32).reshape(tokens, 1, 1, rope_dim) / 100
    cos, sin = phase.cos().to(device="npu", dtype=torch.bfloat16), phase.sin().to(device="npu", dtype=torch.bfloat16)
    metadata = SimpleNamespace(hidden_restore_idx=indices.to("npu"))

    def update(seed):
        shards = []
        for owner in range(WORLD_SIZE):
            g = torch.Generator().manual_seed(seed * WORLD_SIZE + owner)
            shard = torch.randn(tokens, hidden_size, generator=g, dtype=torch.bfloat16)
            shard[lengths[owner] :].fill_(9)  # Padding must not leak into restored data.
            shards.append(shard)
        hidden.copy_(shards[rank])
        restored_order = indices.roll(seed % len(indices)) if len(indices) else indices
        metadata.hidden_restore_idx.copy_(restored_order)
        cos.copy_((phase + seed / 7).cos())
        sin.copy_((phase + seed / 7).sin())
        return torch.cat(shards).index_select(0, restored_order)

    return hidden, metadata, cos, sin, lengths[rank], update, lengths


def benchmark_pair(functions, rounds, iterations):
    measurements = {name: [] for name in functions}
    names = list(functions)
    for repeat in range(rounds):
        # Reverse order every round to reduce drift bias.
        for name in names if repeat % 2 == 0 else names[::-1]:
            synchronize_ranks()
            start = time.perf_counter()
            for _ in range(iterations):
                functions[name]()
            torch.npu.synchronize()
            measurements[name].append(max(collect((time.perf_counter() - start) * 1000 / iterations)))
    return {name: {"round_ms": values, "median_ms": statistics.median(values)} for name, values in measurements.items()}


def profile_pair(functions, output, rank):
    for name, function in functions.items():
        synchronize_ranks()
        if rank == 0:
            with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                record_shapes=False,
                with_stack=False,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(output / name)),
            ) as prof:
                for _ in range(PROFILE_STEPS):
                    function()
                    prof.step()
                torch.npu.synchronize()
        else:
            for _ in range(PROFILE_STEPS):
                function()
        synchronize_ranks()


def run_case(experiment, config, args, tokens, kind, rank):
    hf = config.model_config.hf_config
    hidden, metadata, cos, sin, actual, update, lengths = make_case(
        tokens,
        kind,
        hf.hidden_size,
        hf.qk_rope_head_dim,
        rank,
    )
    expected_hidden = update(11)
    functions = {
        "serial": lambda: experiment.forward(hidden, metadata, cos, sin, actual, False),
        "overlap": lambda: experiment.forward(hidden, metadata, cos, sin, actual, True),
    }
    for function in functions.values():
        for _ in range(WARMUP_STEPS):
            function()
    synchronize_ranks()
    ref = functions["serial"]()
    torch.npu.synchronize()
    assert torch.equal(ref[0].cpu(), expected_hidden)
    candidate = functions["overlap"]()
    torch.npu.synchronize()
    correctness = [{"phase": "eager", "stats": compare(candidate, ref)}]
    timing = {"eager": benchmark_pair(functions, args.rounds, args.iterations)}
    graphs, outputs = {}, {}
    for name, function in functions.items():
        synchronize_ranks()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            result = function()
        graph.replay()
        torch.npu.synchronize()
        correctness.append({"phase": name + "_graph", "stats": compare(result, ref)})
        graphs[name], outputs[name] = graph, result
    graph_functions = {name: graph.replay for name, graph in graphs.items()}
    timing["graph"] = benchmark_pair(graph_functions, args.rounds, args.iterations)
    for seed in REPLAY_INPUT_SEEDS:
        expected_hidden = update(seed)
        ref = functions["serial"]()
        torch.npu.synchronize()
        reference = tuple(x.cpu().clone() if x is not None else None for x in ref)
        for name, graph in graphs.items():
            for tensor in outputs[name]:
                if tensor is not None:
                    tensor.fill_(float("nan") if tensor.is_floating_point() else 127)
            graph.replay()
            torch.npu.synchronize()
            assert torch.equal(outputs[name][0].cpu(), expected_hidden)
            correctness.append({"phase": name + "_dynamic", "seed": seed, "stats": compare(outputs[name], reference)})
    if kind == "balanced" and tokens == args.profile_tokens:
        profile_pair(graph_functions, args.output / "profiles", rank)
    row = {
        "tokens_per_rank": tokens,
        "case": kind,
        "actual_lengths": lengths,
        "rank": rank,
        "correctness": correctness,
        "timing": timing,
    }
    (args.output / f"{kind}-{tokens}-rank{rank}.json").write_text(json.dumps(row, indent=2))
    if rank == 0:
        print(json.dumps({"case": kind, "tokens": tokens, "timing": timing}), flush=True)
    synchronize_ranks()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/data/models")
    parser.add_argument("--output", type=Path, default=Path("/bench/results"))
    parser.add_argument("--tokens", type=int, nargs="+", default=[128, 512, 1024])
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--profile-tokens", type=int, default=1024)
    parser.add_argument("--skip-edge-cases", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # torchrun supplies rank variables; vLLM central envs do not own these.
    rank = int(os.environ["RANK"])
    assert int(os.environ["WORLD_SIZE"]) == WORLD_SIZE
    torch.npu.set_device(rank)
    torch.npu.config.allow_internal_format = True
    torch.set_default_dtype(torch.bfloat16)
    torch.set_grad_enabled(False)
    config = EngineArgs(
        model=args.model,
        trust_remote_code=True,
        quantization="ascend",
        tensor_parallel_size=1,
        prefill_context_parallel_size=WORLD_SIZE,
        enable_expert_parallel=True,
        max_model_len=200000,
        max_num_batched_tokens=8192,
        max_num_seqs=16,
        disable_custom_all_reduce=True,
        enforce_eager=True,
        additional_config={"enable_cpu_binding": False},
    ).create_engine_config()
    adapt_patch()
    register_ascend_customop(config)
    init_ascend_config(config)
    assert enable_custom_op()
    with set_current_vllm_config(config):
        init_distributed_environment(WORLD_SIZE, rank, "env://", rank, "hccl")
        ensure_model_parallel_initialized(1, 1, WORLD_SIZE, 1)
        init_ascend_model_parallel(config.parallel_config, config.engram_config)
        assert get_pcp_group().world_size == WORLD_SIZE
        impl = build_projection(config)
        experiment = PCPQueryExperiment(impl)
        if rank == 0:
            print(
                "PROJECTION",
                [
                    (name, str(getattr(impl, name).weight.dtype), list(getattr(impl, name).weight.shape))
                    for name in ("wq_a", "wq_b", "wkv")
                ],
                flush=True,
            )
        for tokens in args.tokens:
            run_case(experiment, config, args, tokens, "balanced", rank)
        if not args.skip_edge_cases:
            for kind in ("uneven", "empty_rank", "all_empty", "strided"):
                run_case(experiment, config, args, 1 if kind == "all_empty" else 128, kind, rank)
    synchronize_ranks()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
