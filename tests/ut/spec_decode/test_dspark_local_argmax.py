# SPDX-License-Identifier: Apache-2.0
"""CPU checks for Markov-corrected, vocabulary-sharded greedy drafting."""

import importlib.util
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Load the pure-Torch helper without initializing the NPU plugin. This also
# lets the Gloo checks run independently of the full vLLM test environment.
_HELPER_PATH = Path(__file__).resolve().parents[3] / "vllm_ascend/spec_decode/dspark_local_argmax.py"


def _load_sampler():
    spec = importlib.util.spec_from_file_location("dspark_local_argmax_under_test", _HELPER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.sample_local_draft_tokens


def _check_case(rank, world_size, dtype, steps, mode, vocab_size):
    generator = torch.Generator().manual_seed(41)
    batch = 3
    full = torch.randn(batch, steps, vocab_size, generator=generator, dtype=dtype)
    transitions = torch.randn(vocab_size, generator=generator, dtype=dtype)
    if mode in ("ties", "negative_infinity"):
        full.fill_(0 if mode == "ties" else -float("inf"))
    if mode == "markov":
        full.zero_()
        full[:, :, 0] = 9
    if mode == "nan":
        full[:, :, vocab_size - 1] = float("nan")
        full[:, :, 1] = float("nan")
    seed = torch.tensor([0, vocab_size - 1, vocab_size // 2], dtype=torch.int64)

    def bias(tokens):
        values = transitions.expand(batch, -1).clone()
        if mode in ("ties", "negative_infinity", "nan"):
            values.zero_()
        elif mode == "markov":
            values.zero_()
            values.scatter_(1, ((tokens + 3) % vocab_size).unsqueeze(-1), 20)
        else:
            values.scatter_add_(1, tokens.unsqueeze(-1), torch.ones(batch, 1, dtype=dtype))
        return values

    reference = torch.empty(batch, steps, dtype=torch.int64)
    previous = seed
    for position in range(steps):
        reference[:, position] = (full[:, position] + bias(previous)).argmax(-1)
        previous = reference[:, position]

    shard_size = (vocab_size + world_size - 1) // world_size
    start = rank * shard_size
    valid = max(0, min(shard_size, vocab_size - start))
    # Padding must never become a candidate, even if its original score is high.
    local = torch.full((batch, steps, shard_size), 1000, dtype=dtype)
    local[:, :, :valid] = full[:, :, start : start + valid]
    calls = []

    def gather(pairs):
        calls.append(tuple(pairs.shape))
        outputs = [torch.empty_like(pairs) for _ in range(world_size)]
        dist.all_gather(outputs, pairs)
        return torch.cat(outputs, dim=-1)

    actual = _load_sampler()(local, seed, bias, start, vocab_size, world_size, gather)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert calls == ([(batch, 2)] * steps if world_size > 1 else [])


def _distributed_worker(rank, world_size, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=world_size, timeout=timedelta(seconds=90)
    )
    try:
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for steps in (5, 7):
                for mode in ("random", "ties", "markov", "negative_infinity", "nan"):
                    _check_case(rank, world_size, dtype, steps, mode, 33)
                # More ranks than valid vocabulary entries exercises empty shards.
                _check_case(rank, world_size, dtype, steps, "ties", 3)
            _check_case(rank, world_size, dtype, 5, "markov", 131077)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [1, 2, 8])
def test_distributed_markov_argmax(world_size, tmp_path):
    rendezvous = "file://" + str(Path(tmp_path) / "store")
    mp.spawn(_distributed_worker, args=(world_size, rendezvous), nprocs=world_size, join=True)


def test_reject_inexact_packed_token_ids():
    def unexpected_call(*args):
        raise AssertionError("No projection or collective should run for an unsupported vocabulary")

    with pytest.raises(ValueError, match="exact FP32"):
        _load_sampler()(
            torch.zeros(1, 1, 1),
            torch.zeros(1, dtype=torch.int64),
            unexpected_call,
            0,
            (1 << 24) + 1,
            1,
            unexpected_call,
        )
