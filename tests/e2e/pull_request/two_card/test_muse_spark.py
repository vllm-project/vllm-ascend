# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Muse Spark Open under expert parallelism on two NPUs.

The reduced config and the dummy-weight harness come from the one-card test;
the parallel layout differs, and the expert hidden size is widened to what the
MC2 dispatch kernel accepts. What this adds over that test:

* the experts are really sharded (``ep_size == 2``, half the experts per rank),
* the MoE communication method the selector lands on is recorded, and the one
  path that cannot carry Muse Spark's per-route norm is rejected,
* the decode still replays, so the sharded routing is deterministic.

What it does not check is the *value* of the per-route norm under expert
parallelism. Dummy weights are sharded differently per layout, so there is no
one-card output to compare against. That the norm is still correct rests on it
being a single shared row-wise norm, which commutes with the dispatch
permutation; ``tests/ut/models/test_muse_spark.py`` pins that property.
"""

from typing import Any, cast

import pytest

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from tests.e2e.pull_request.one_card.test_muse_spark import (
    TEXT_CONFIG,
    _assert_replay_matches,
    _generate,
    _write_model,
)
from vllm_ascend.models import _upstream_has_module

pytestmark = pytest.mark.skipif(
    not _upstream_has_module("vllm.models.muse_spark.amd.model"),
    reason="vLLM without Muse Spark",
)

TENSOR_PARALLEL_SIZE = 2
# The MC2 dispatch kernel wants the expert hidden size in [1024, 8192]
# (``npu_moe_distribute_dispatch_v2`` fails tiling below it), so the latent
# width the one-card test uses is too narrow here. The released model sits at
# 4096. Widening it makes the pooler expand rather than reduce, which the code
# does not care about and no assertion here reads.
EP_TEXT_CONFIG = TEXT_CONFIG | {"moe_hidden_size": 1024, "moe_intermediate_size": 1024}
# TEXT_CONFIG is the on-disk config dict, so its values are untyped.
NUM_EXPERTS = cast(int, TEXT_CONFIG["num_local_experts"])
EXPERTS_PER_RANK = NUM_EXPERTS // TENSOR_PARALLEL_SIZE
# The comm selector is a function of the token count, so ask it about a batch
# the size of the decodes this test actually runs.
DECODE_TOKENS = 2


class MuseSparkExpertParallelWorker:
    """Read-only checks of how the MoE is laid out on this rank."""

    model_runner: Any  # Supplied by the worker extension host at runtime.
    vllm_config: Any

    def check_muse_spark_expert_parallel(self) -> str:
        from vllm_ascend.ascend_forward_context import select_moe_comm_method
        from vllm_ascend.models.muse_spark import AscendMuseSpark1xMoE

        moes = [module for module in self.model_runner.model.modules() if isinstance(module, AscendMuseSpark1xMoE)]
        assert moes, "no Muse Spark MoE layer on this rank"
        for moe in moes:
            assert moe.use_ep
            assert moe.ep_size == TENSOR_PARALLEL_SIZE
            assert moe.n_physical_experts == NUM_EXPERTS
            assert moe.n_local_physical_experts == EXPERTS_PER_RANK
            assert moe.physical_expert_end - moe.physical_expert_start == EXPERTS_PER_RANK
            # The norm has to be reachable; the MLP stage raises without it.
            assert moe.experts.routed_experts.quant_method.post_expert_norm_fn is not None

        comm = select_moe_comm_method(num_tokens=DECODE_TOKENS, vllm_config=self.vllm_config)
        return comm.name


@wait_until_npu_memory_free()
def test_muse_spark_expert_parallel_shards_and_replays(tmp_path):
    model = tmp_path / "model"
    _write_model(model, EP_TEXT_CONFIG)
    with VllmRunner(
        str(model),
        load_format="dummy",
        skip_tokenizer_init=True,
        worker_extension_cls=f"{__name__}.MuseSparkExpertParallelWorker",
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        enable_expert_parallel=True,
        dtype="bfloat16",
        max_model_len=1024,
        max_num_seqs=2,
        max_num_batched_tokens=1024,
        enforce_eager=True,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.5,
        seed=0,
    ) as runner:
        comm_methods = runner.model.collective_rpc("check_muse_spark_expert_parallel")
        baseline = _generate(runner)
        replay = _generate(runner)

    assert len(comm_methods) == TENSOR_PARALLEL_SIZE
    assert len(set(comm_methods)) == 1, f"ranks disagree on the MoE comm method: {comm_methods}"
    # Which of these the selector picks depends on the device policy: A5 and A3
    # reach MC2 at two ranks, A2 requires a 16-rank EP group and stays on
    # ALLGATHER. Both keep the MLP stage, and therefore the per-route norm.
    # FUSED_MC2 fuses dispatch, MLP and combine into one operator and has no
    # place for that norm, which is why the model rejects it at construction.
    assert comm_methods[0] in {"MC2", "ALLGATHER"}, f"unexpected MoE comm method {comm_methods[0]}"
    print(f"MoE comm method under EP=2: {comm_methods[0]}")

    _assert_replay_matches(baseline, replay)
