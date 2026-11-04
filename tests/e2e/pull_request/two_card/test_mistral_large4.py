# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Mistral Large 4 on two NPUs: the two MoE layouts, with dummy weights.

Reuses the reduced config and harness of the one-card test. Both cases check
the layout the ranks actually ended up in and that the decode still replays;
neither compares output against a one-card run, because dummy weights shard
differently per layout, so there is no baseline to compare against.
"""

import pytest

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from tests.e2e.pull_request.one_card.test_mistral_large4 import (
    CONFIG,
    SHARING_LAYERS,
    MistralLarge4Worker,
    _assert_replay_matches,
    _generate,
    _write_model,
)
from vllm_ascend.models import _upstream_has_module

pytestmark = pytest.mark.skipif(
    not _upstream_has_module("vllm.models.mistral.mistral_large4"),
    reason="vLLM without Mistral Large 4 (vllm-project/vllm#64534)",
)

TP_SIZE = 2
NUM_EXPERTS = CONFIG["n_routed_experts"]


class MistralLarge4ParallelWorker(MistralLarge4Worker):
    """The one-card checks, plus what the parallel layout did to the MoE."""

    def check_expert_parallel_shards(self) -> bool:
        assert self.check_mistral_large4_paths()
        experts = self._routed_experts()
        # Expert parallelism shards the experts; each rank owns half.
        assert experts.ep_size == TP_SIZE, experts.ep_size
        local = experts.n_local_physical_experts
        assert local == NUM_EXPERTS // TP_SIZE, local
        return True

    def check_tensor_parallel_does_not_shard_experts(self) -> bool:
        assert self.check_mistral_large4_paths()
        experts = self._routed_experts()
        # With expert parallelism off every rank holds every expert, and the
        # MoE is sharded along the intermediate size instead.
        assert experts.ep_size == 1, experts.ep_size
        assert experts.n_local_physical_experts == NUM_EXPERTS
        return True

    def _routed_experts(self):
        model = self.model_runner.model
        for layer in model.model.layers:
            experts = getattr(layer.mlp, "experts", None)
            if experts is not None:
                return experts
        raise AssertionError("no MoE layer found in the reduced model")


def _run(tmp_path, enable_expert_parallel: bool, check: str):
    model = tmp_path / "model"
    _write_model(model)
    with VllmRunner(
        str(model),
        load_format="dummy",
        skip_tokenizer_init=True,
        worker_extension_cls=f"{__name__}.MistralLarge4ParallelWorker",
        dtype="bfloat16",
        tensor_parallel_size=TP_SIZE,
        enable_expert_parallel=enable_expert_parallel,
        max_model_len=1024,
        max_num_seqs=2,
        max_num_batched_tokens=1024,
        enforce_eager=True,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.5,
        seed=0,
    ) as runner:
        assert runner.model.collective_rpc(check) == [True] * TP_SIZE
        baseline = _generate(runner)
        replay = _generate(runner)
    _assert_replay_matches(baseline, replay)


@wait_until_npu_memory_free()
def test_mistral_large4_expert_parallel_shards_and_replays(tmp_path):
    """TP=2 with expert parallelism: the experts really shard across ranks."""
    _run(tmp_path, enable_expert_parallel=True, check="check_expert_parallel_shards")


@wait_until_npu_memory_free()
def test_mistral_large4_tensor_parallel_keeps_every_expert_and_replays(tmp_path):
    """TP=2 without it: every rank keeps every expert.

    This is also the layout where the KV-sharing write skip has to survive
    sharding — the worker check runs on both ranks, so a skip that landed on
    only one of them fails here rather than in a long-context run.
    """
    assert SHARING_LAYERS, "the reduced config must keep a KV-sharing layer"
    _run(tmp_path, enable_expert_parallel=False, check="check_tensor_parallel_does_not_shard_experts")
