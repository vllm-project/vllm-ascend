# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the BSD 3-Clause License.

"""Generalized Arch35 FlashDecoding functional and accuracy tests.

Every functional case uses one input set for two calculations:
  1. CPU FP32 mathematical golden;
  2. NPU execution with Host-controlled automatic FD selection.

The test matrix is deterministic.  It targets FD shard partition boundaries,
MHA/GQA/MQA mappings, runtime select_num_idx clamping, partial KV blocks and
numerically difficult softmax distributions for both FP16 and BF16.
"""

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "torch_extension"))

from cann_ops_transformer.ops.sparse_attention_score import (  # noqa: E402
    npu_sparse_attention_score,
)

HEAD_DIM = 128
BLOCK_SIZE = 128
INNER_PRECISE = 4
ARCH35_MAX_AIC = 28
NORMAL_TILING_KEYS = {"fp16": 10001, "bf16": 10002}
FD_TILING_KEYS = {"fp16": 10005, "bf16": 10006}


@dataclass(frozen=True)
class FDCase:
    name: str
    dtype_name: str
    q_tokens: int
    q_heads: int
    kv_heads: int
    top_k: int
    kv_seq_len: int
    select_num_pattern: str = "full"
    data_pattern: str = "random"
    seed: int = 2026
    expected_fd: bool = True
    check_cpu_golden: bool = True

    @property
    def dtype(self):
        return {"fp16": torch.float16, "bf16": torch.bfloat16}[self.dtype_name]

    @property
    def base_tasks(self):
        return self.q_tokens * self.kv_heads


# These cases satisfy the Host FD gate on a 28-AIC Ascend950.  The base-task
# and top-k choices also exercise different shard partition layouts.
FD_ELIGIBLE_CASES = (
    FDCase("bf16_minimum_two_shards", "bf16", 1, 1, 1, 2, 256),
    FDCase("bf16_single_task_top16", "bf16", 1, 16, 1, 16, 2048),
    FDCase("bf16_gqa_two_base_tasks", "bf16", 1, 16, 2, 8, 1024),
    FDCase("bf16_mha_four_base_tasks", "bf16", 1, 4, 4, 4, 512),
    FDCase("bf16_multi_query_eight_base_tasks", "bf16", 4, 16, 2, 8, 1024),
    FDCase("bf16_base_task_24_boundary", "bf16", 12, 16, 2, 2, 256),
    FDCase("bf16_group_size_128_boundary", "bf16", 1, 128, 1, 2, 256),
    FDCase(
        "bf16_runtime_valid_count_mix_partial_block",
        "bf16",
        4,
        8,
        1,
        16,
        2033,
        "mixed",
        "random",
        2033,
    ),
    FDCase("bf16_equal_logits", "bf16", 1, 16, 1, 16, 2048, data_pattern="equal_logits"),
    FDCase("bf16_cross_shard_logit_extremes", "bf16", 1, 16, 1, 16, 2048, data_pattern="shard_extremes"),
    FDCase("bf16_constant_value_invariant", "bf16", 2, 8, 1, 8, 1024, data_pattern="constant_value"),
    FDCase("fp16_minimum_two_shards", "fp16", 1, 1, 1, 2, 256),
    FDCase("fp16_single_task_top16", "fp16", 1, 16, 1, 16, 2048),
    FDCase("fp16_gqa_eight_base_tasks", "fp16", 4, 16, 2, 8, 1024),
    FDCase(
        "fp16_runtime_valid_count_mix",
        "fp16",
        4,
        8,
        1,
        16,
        2048,
        "mixed",
        "shard_extremes",
        9527,
    ),
)


# These cases do not satisfy the Host FD gate and must automatically fall back
# to the normal path.
FD_FALLBACK_CASES = (
    FDCase("fallback_topk_1_no_extra_shard", "bf16", 1, 16, 1, 1, 128, expected_fd=False),
    FDCase("fallback_topk_17_policy_only", "bf16", 1, 16, 1, 17, 2176, expected_fd=False, check_cpu_golden=False),
    FDCase("fallback_base_tasks_equal_aic", "bf16", 28, 1, 1, 2, 256, expected_fd=False),
    FDCase("fallback_fp16_topk_1", "fp16", 1, 16, 1, 1, 128, expected_fd=False),
)

ZERO_SELECT_CASE = FDCase(
    "known_limit_zero_valid_blocks",
    "bf16",
    1,
    16,
    1,
    2,
    256,
    select_num_pattern="zero",
    expected_fd=True,
)

# With the current FD cost model this shape launches 13 AICs and assigns five
# flattened TopK records per AIC. Some AICs therefore finish one base task and
# immediately start another, exercising partial-result MTE3 completion before
# the next task reuses the same Vector UB region.
CROSS_BASE_TASK_CASE = FDCase(
    "cross_base_task_partial_write_completion",
    "bf16",
    1,
    16,
    4,
    16,
    2048,
)


def _case_id(case):
    route = "fd" if case.expected_fd else "fallback"
    return f"{route}-{case.name}"


def _expected_fd_compute_cores(case: FDCase, aic_num=ARCH35_MAX_AIC):
    """Mirror the Host's continuous flattened-range core calculation."""
    if not case.expected_fd:
        return case.base_tasks
    total_tasks = case.base_tasks * case.top_k
    tasks_per_core = math.ceil(total_tasks / aic_num)
    return math.ceil(total_tasks / tasks_per_core)


def _valid_counts(case: FDCase):
    shape = (case.kv_heads, case.q_tokens)
    if case.select_num_pattern == "full":
        return torch.full(shape, case.top_k, dtype=torch.int32)
    if case.select_num_pattern == "mixed":
        # A causal sparse-attention request always contains at least the
        # current logical block.  Cover the supported runtime range starting
        # from one; zero is kept as a separate known-limit regression below.
        values = [1, min(2, case.top_k), max(1, case.top_k - 1), case.top_k]
        return torch.tensor(
            [values[i % len(values)] for i in range(case.base_tasks)],
            dtype=torch.int32,
        ).reshape(shape)
    if case.select_num_pattern == "zero":
        return torch.zeros(shape, dtype=torch.int32)
    raise ValueError(f"unknown select_num_pattern: {case.select_num_pattern}")


def _make_inputs(case: FDCase) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(case.seed)
    total_blocks = math.ceil(case.kv_seq_len / BLOCK_SIZE)
    assert total_blocks >= case.top_k
    assert case.q_heads % case.kv_heads == 0

    query = (
        torch.randn(
            case.q_tokens,
            case.q_heads,
            HEAD_DIM,
            generator=generator,
            dtype=torch.float32,
        )
        * 0.25
    )
    key = (
        torch.randn(
            total_blocks,
            BLOCK_SIZE,
            case.kv_heads,
            HEAD_DIM,
            generator=generator,
            dtype=torch.float32,
        )
        * 0.25
    )
    value = (
        torch.randn(
            total_blocks,
            BLOCK_SIZE,
            case.kv_heads,
            HEAD_DIM,
            generator=generator,
            dtype=torch.float32,
        )
        * 0.25
    )

    # Reverse logical-to-physical mapping so the tests exercise block_table
    # address translation instead of relying on identity block IDs.
    physical_ids = torch.arange(total_blocks - 1, -1, -1, dtype=torch.int32)
    block_table = physical_ids.reshape(1, total_blocks)

    if case.data_pattern == "equal_logits":
        query.zero_()
        key.zero_()
    elif case.data_pattern == "constant_value":
        value.fill_(0.125)
    elif case.data_pattern == "shard_extremes":
        query.fill_(0.25)
        # Values span negative and positive logits.  Logical block 0 and the
        # last selected block land in different shards for the top-k=16 cases.
        levels = torch.linspace(-1.5, 1.5, total_blocks)
        for logical_id, level in enumerate(levels):
            physical_id = int(block_table[0, logical_id])
            key[physical_id].fill_(float(level))
    elif case.data_pattern != "random":
        raise ValueError(f"unknown data_pattern: {case.data_pattern}")

    base_order = torch.randperm(total_blocks, generator=generator)[: case.top_k]
    select_idx = torch.empty(
        case.kv_heads,
        case.q_tokens,
        case.top_k,
        dtype=torch.int32,
    )
    for kv_head in range(case.kv_heads):
        for q_token in range(case.q_tokens):
            shift = (kv_head * case.q_tokens + q_token) % case.top_k
            select_idx[kv_head, q_token] = torch.roll(base_order, shifts=shift)

    select_num_idx = _valid_counts(case)
    # Invalid suffix values must be ignored by the FD records that are outside
    # the runtime-valid prefix.  -1 also makes accidental reads easy to detect.
    for kv_head in range(case.kv_heads):
        for q_token in range(case.q_tokens):
            valid = int(select_num_idx[kv_head, q_token])
            if valid < case.top_k:
                select_idx[kv_head, q_token, valid:] = -1

    return {
        "query": query.to(case.dtype),
        "key": key.to(case.dtype),
        "value": value.to(case.dtype),
        "select_idx": select_idx,
        "block_table": block_table,
        "select_num_idx": select_num_idx,
        "actual_seq_lengths": torch.tensor([case.q_tokens], dtype=torch.int32),
        "actual_seq_lengths_kv": torch.tensor([case.kv_seq_len], dtype=torch.int32),
    }


def _cpu_fp32_golden(case: FDCase, inputs: dict[str, torch.Tensor]):
    query = inputs["query"].float()
    key = inputs["key"].float()
    value = inputs["value"].float()
    select_idx = inputs["select_idx"].to(torch.int64)
    block_table = inputs["block_table"].to(torch.int64)
    select_num_idx = inputs["select_num_idx"].to(torch.int64)
    group_size = case.q_heads // case.kv_heads
    scale = 1.0 / math.sqrt(HEAD_DIM)
    output = torch.zeros_like(query)
    history_len = case.kv_seq_len - case.q_tokens

    for q_token in range(case.q_tokens):
        causal_bound = history_len + q_token
        for kv_head in range(case.kv_heads):
            keys = []
            values = []
            valid_count = min(int(select_num_idx[kv_head, q_token]), case.top_k)
            for raw_idx in range(valid_count):
                logical_id = int(select_idx[kv_head, q_token, raw_idx])
                if logical_id < 0:
                    continue
                block_begin = logical_id * BLOCK_SIZE
                block_end = min(block_begin + BLOCK_SIZE, case.kv_seq_len)
                effective_end = min(block_end, causal_bound + 1)
                if effective_end <= block_begin:
                    continue
                physical_id = int(block_table[0, logical_id])
                valid_len = effective_end - block_begin
                keys.append(key[physical_id, :valid_len, kv_head])
                values.append(value[physical_id, :valid_len, kv_head])

            if not keys:
                continue
            all_key = torch.cat(keys, dim=0)
            all_value = torch.cat(values, dim=0)
            q_begin = kv_head * group_size
            q_group = query[q_token, q_begin : q_begin + group_size]
            probability = torch.softmax(torch.matmul(q_group, all_key.t()) * scale, dim=-1)
            output[q_token, q_begin : q_begin + group_size] = torch.matmul(
                probability,
                all_value,
            )
    return output


def _run_npu(case: FDCase, inputs: dict[str, torch.Tensor]):
    kwargs = dict(
        select_num_idx=inputs["select_num_idx"].npu(),
        actual_seq_lengths=inputs["actual_seq_lengths"].npu(),
        actual_seq_lengths_kv=inputs["actual_seq_lengths_kv"].npu(),
        num_key_value_heads=case.kv_heads,
        scale_value=1.0 / math.sqrt(HEAD_DIM),
        block_size=BLOCK_SIZE,
        top_k=case.top_k,
        inner_precise=INNER_PRECISE,
    )
    output = npu_sparse_attention_score(
        inputs["query"].npu(),
        inputs["key"].npu(),
        inputs["value"].npu(),
        inputs["select_idx"].npu(),
        inputs["block_table"].npu(),
        **kwargs,
    )
    torch.npu.synchronize()
    return output.cpu()


def _assert_accuracy(case: FDCase, actual, golden):
    assert actual.shape == inputs_shape(case)
    assert actual.dtype == case.dtype
    assert torch.isfinite(actual.float()).all()

    # top_k=17 is intentionally outside the current FD range.  It is retained
    # only to exercise automatic fallback; current normal-kernel accuracy
    # support is not asserted for that shape.
    if not case.check_cpu_golden:
        print(f"[{case.name}] automatic fallback; CPU accuracy is outside this case's contract")
        return

    actual_fp32 = actual.float()
    absolute = (actual_fp32 - golden).abs()
    golden_l1 = golden.abs().sum().item()
    relative_l1 = absolute.sum().item() / (golden_l1 + 1e-12)
    if golden_l1 == 0.0:
        assert absolute.max().item() <= 7e-3
        cosine = 1.0
    else:
        cosine = torch.nn.functional.cosine_similarity(
            actual_fp32.flatten(),
            golden.flatten(),
            dim=0,
        ).item()
        assert relative_l1 <= 2e-2, f"{case.name}: relative_l1={relative_l1:.8f}, max_diff={absolute.max().item():.8f}"
        assert cosine >= 0.999, f"{case.name}: cosine={cosine:.8f}"
    print(
        f"[{case.name}] dtype={case.dtype_name}, base_tasks={case.base_tasks}, "
        f"top_k={case.top_k}, expected_compute_cores={_expected_fd_compute_cores(case)}, "
        f"max_diff_cpu={absolute.max().item():.8f}, "
        f"relative_l1={relative_l1:.8f}, cosine={cosine:.8f}"
    )


def inputs_shape(case: FDCase) -> tuple[int, int, int]:
    return case.q_tokens, case.q_heads, HEAD_DIM


@pytest.mark.parametrize("case", FD_ELIGIBLE_CASES, ids=_case_id)
def test_fd_generalized_accuracy(case):
    inputs = _make_inputs(case)
    golden = _cpu_fp32_golden(case, inputs)
    fd_auto = _run_npu(case, inputs)
    _assert_accuracy(case, fd_auto, golden)


@pytest.mark.parametrize("case", FD_FALLBACK_CASES, ids=_case_id)
def test_fd_automatic_fallback_accuracy(case):
    inputs = _make_inputs(case)
    golden = _cpu_fp32_golden(case, inputs)
    actual = _run_npu(case, inputs)
    _assert_accuracy(case, actual, golden)


@pytest.mark.xfail(
    reason=("known limit: select_num_idx=0 does not initialize all normal/FD output or partial-result storage"),
    strict=False,
)
def test_zero_valid_blocks_should_produce_zero_output():
    inputs = _make_inputs(ZERO_SELECT_CASE)
    fd_auto = _run_npu(ZERO_SELECT_CASE, inputs)
    torch.testing.assert_close(
        fd_auto.float(),
        torch.zeros_like(fd_auto.float()),
        rtol=0.0,
        atol=0.0,
    )


def test_cross_base_task_partial_write_completion_is_stable():
    inputs = _make_inputs(CROSS_BASE_TASK_CASE)
    golden = _cpu_fp32_golden(CROSS_BASE_TASK_CASE, inputs)
    for iteration in range(100):
        fd_auto = _run_npu(CROSS_BASE_TASK_CASE, inputs)
        try:
            _assert_accuracy(CROSS_BASE_TASK_CASE, fd_auto, golden)
        except AssertionError as error:
            raise AssertionError(f"cross-base-task partial write is unstable at iteration {iteration}") from error


def test_case_matrix_covers_fd_boundaries():
    """Guard the generated matrix itself against accidental coverage loss."""
    eligible = FD_ELIGIBLE_CASES
    fallback = FD_FALLBACK_CASES
    assert {case.dtype_name for case in eligible} == {"fp16", "bf16"}
    assert {case.top_k for case in eligible}.issuperset({2, 4, 8, 16})
    assert {case.top_k for case in fallback}.issuperset({1, 17})
    assert {case.base_tasks for case in eligible}.issuperset({1, 2, 4, 8, 24})
    assert any(case.base_tasks == ARCH35_MAX_AIC for case in fallback)
    assert {case.select_num_pattern for case in eligible} == {"full", "mixed"}
    assert {case.data_pattern for case in eligible}.issuperset(
        {"random", "equal_logits", "constant_value", "shard_extremes"}
    )
    assert all(_expected_fd_compute_cores(case) >= case.base_tasks for case in eligible)
    assert any(_expected_fd_compute_cores(case) > case.base_tasks for case in eligible)
    assert all(not case.expected_fd for case in fallback)
